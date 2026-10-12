"""Harvest public metadata changes, independently of version submission dates.

OAI exposes the latest version of each changed article. A paper first made public
today is harvested today even if its version was submitted months ago.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
import time
import xml.etree.ElementTree as ET

import requests

ENDPOINT = 'https://oaipmh.arxiv.org/oai'
DEFAULT_CATEGORIES = ('cs.AI', 'cs.LG', 'cs.CL', 'stat.ML')
OAI = '{http://www.openarchives.org/OAI/2.0/}'
RAW = '{http://arxiv.org/OAI/arXivRaw/}'


def category_set(category):
    if not re.fullmatch(r'[a-z][a-z-]*\.[A-Za-z][A-Za-z-]*', category):
        raise ValueError(f'Invalid arXiv category: {category}')
    archive, subject = category.split('.')
    # cs and stat are both their own OAI group and archive.
    group = archive if archive in ('cs', 'stat', 'math', 'q-bio', 'q-fin', 'econ', 'eess') else 'physics'
    return f'{group}:{archive}:{subject}'


def parse_page(content, categories):
    """Strict parsing: a malformed/missing record must never advance a cursor."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise RuntimeError('Malformed OAI XML; sync cursor not advanced') from exc
    if root.tag != OAI + 'OAI-PMH':
        raise RuntimeError('Response is not an OAI-PMH document')
    response = root.findtext(OAI + 'responseDate')
    if not response:
        raise RuntimeError('OAI responseDate is missing')
    response_day = datetime.fromisoformat(response.replace('Z', '+00:00')).astimezone(timezone.utc).date().isoformat()
    errors = root.findall(OAI + 'error')
    if errors:
        if len(errors) == 1 and errors[0].get('code') == 'noRecordsMatch':
            return [], [], None, None, response_day, 0
        raise RuntimeError('OAI error: ' + '; '.join(f"{e.get('code')}: {e.text}" for e in errors))
    listing = root.find(OAI + 'ListRecords')
    if listing is None:
        raise RuntimeError('OAI ListRecords is missing')
    records = listing.findall(OAI + 'record')
    rows, deleted = [], []
    for record in records:
        header = record.find(OAI + 'header')
        if header is None or not header.findtext(OAI + 'identifier'):
            raise RuntimeError('OAI record header/identifier is missing')
        if header.get('status') == 'deleted':
            deleted.append(header.findtext(OAI + 'identifier'))
            continue
        datestamp = header.findtext(OAI + 'datestamp')
        if not datestamp:
            raise RuntimeError('OAI record datestamp is missing')
        datetime.strptime(datestamp, '%Y-%m-%d')
        raw = record.find(OAI + 'metadata/' + RAW + 'arXivRaw')
        if raw is None:
            raise RuntimeError('OAI arXivRaw metadata is missing')
        identifier = raw.findtext(RAW + 'id')
        paper_categories = (raw.findtext(RAW + 'categories') or '').split()
        if not identifier or not paper_categories:
            raise RuntimeError('OAI article id/categories are missing')
        if not set(paper_categories).intersection(categories):
            continue
        versions = []
        for version in raw.findall(RAW + 'version'):
            label = version.get('version', '')
            if not re.fullmatch(r'v[1-9]\d*', label):
                raise RuntimeError(f'Invalid version for {identifier}: {label}')
            submitted = parsedate_to_datetime(version.findtext(RAW + 'date') or '')
            if submitted.tzinfo is None:
                raise RuntimeError(f'Version date lacks timezone: {identifier}')
            versions.append((int(label[1:]), submitted.astimezone(timezone.utc)))
        if not versions:
            raise RuntimeError(f'Version history missing: {identifier}')
        latest, updated = max(versions)
        published = min(versions)[1]
        title = (raw.findtext(RAW + 'title') or '').strip()
        abstract = (raw.findtext(RAW + 'abstract') or '').strip()
        if not title or not abstract:
            raise RuntimeError(f'Title/abstract missing: {identifier}')
        rows.append({
            'title': title, 'abstract': abstract.replace('\n', ' '),
            'authors': [(raw.findtext(RAW + 'authors') or '').strip()],
            'url': f'https://arxiv.org/abs/{identifier}v{latest}',
            'date': updated.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'published': published.isoformat(), 'updated': updated.isoformat(),
            'status': 'v1 (New)' if latest == 1 else f'v{latest} Update',
            'arxiv_categories': paper_categories, 'oai_datestamp': datestamp,
            'version_history': [{'version': n, 'submitted_at': d.isoformat()} for n, d in sorted(versions)],
        })
    token_element = listing.find(OAI + 'resumptionToken')
    token = (token_element.text or '').strip() if token_element is not None else ''
    total = token_element.get('completeListSize') if token_element is not None else None
    if not records and token_element is None:
        raise RuntimeError('Empty OAI listing without noRecordsMatch or pagination status')
    return rows, deleted, token or None, int(total) if total is not None else None, response_day, len(records)


@dataclass
class Harvest:
    papers: list
    response_day: str
    record_count: int
    deleted_identifiers: list


def fetch_metadata_updates(start, until=None, categories=DEFAULT_CATEGORIES, *, max_retries=5,
                           retry_delay=3, timeout=60):
    """Follow every resumption token, across the union of category sets.

    No result cap: callers split outputs into bounded batches, never truncate a
    harvest. An expired token or any incomplete stream fails the whole harvest.
    """
    datetime.strptime(start, '%Y-%m-%d')
    if max_retries < 1 or retry_delay < 0 or timeout <= 0:
        raise ValueError('Retries and timeout must be positive; retry delay must be nonnegative')
    if until:
        datetime.strptime(until, '%Y-%m-%d')
        if until < start:
            raise ValueError('OAI until must be on or after from')
    categories = tuple(dict.fromkeys(categories))
    if not categories:
        raise ValueError('At least one arXiv category is required')
    rows, deleted, response_days = {}, set(), []
    record_count = 0
    last_request = None
    with requests.Session() as session:
        session.headers['User-Agent'] = 'llm-paper-filter/1.0 (arXiv OAI-PMH metadata harvester)'
        for category in categories:
            params = {'verb': 'ListRecords', 'metadataPrefix': 'arXivRaw',
                      'set': category_set(category), 'from': start}
            if until:
                params['until'] = until
            seen_tokens, stream_count, expected_total = set(), 0, None
            while True:
                for attempt in range(max_retries):
                    if last_request is not None:
                        time.sleep(max(0, 3 - (time.monotonic() - last_request)))
                    last_request = time.monotonic()
                    try:
                        response = session.get(ENDPOINT, params=params, timeout=timeout)
                        response.raise_for_status()
                        break
                    except requests.RequestException as exc:
                        if attempt + 1 == max_retries:
                            raise RuntimeError('OAI fetch failed; sync cursor not advanced') from exc
                        delay = retry_delay * (2 ** attempt)
                        failed = getattr(exc, 'response', None)
                        retry_after = failed.headers.get('Retry-After') if failed is not None else None
                        if retry_after:
                            try:
                                delay = max(delay, float(retry_after))
                            except ValueError:
                                delay = max(delay, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                        print(f'OAI request retry {attempt + 1}/{max_retries} in {delay:g}s', flush=True)
                        time.sleep(delay)
                page_rows, page_deleted, token, total, response_day, count = parse_page(response.content, categories)
                response_days.append(response_day)
                if response_day < start:
                    raise RuntimeError('OAI response predates requested start; cursor not advanced')
                stream_count += count
                record_count += count
                if total is not None:
                    if expected_total is not None and total != expected_total:
                        raise RuntimeError('OAI result total changed during pagination; retry harvest')
                    expected_total = total
                deleted.update(page_deleted)
                for row in page_rows:
                    previous = rows.get(row['url'])
                    if previous is None or row['oai_datestamp'] >= previous['oai_datestamp']:
                        rows[row['url']] = row
                print(f'OAI {category}: {stream_count} records read', flush=True)
                if not token:
                    if expected_total is not None and stream_count != expected_total:
                        raise RuntimeError('OAI pagination ended before the advertised total')
                    break
                if token in seen_tokens:
                    raise RuntimeError('OAI repeated a resumption token; cursor not advanced')
                seen_tokens.add(token)
                params = {'verb': 'ListRecords', 'resumptionToken': token}
    return Harvest(sorted(rows.values(), key=lambda r: (r['oai_datestamp'], r['url'])),
                   min(response_days), record_count, sorted(deleted))

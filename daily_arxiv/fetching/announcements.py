"""Read official daily announcements; pubDate is not a submission timestamp."""
from email.utils import parsedate_to_datetime
import re
import time
import xml.etree.ElementTree as ET

import requests

from daily_arxiv.fetching.oai import DEFAULT_CATEGORIES, category_set

ARXIV = '{http://arxiv.org/schemas/atom}'
DC = '{http://purl.org/dc/elements/1.1/}'


def parse_announcements(content):
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise RuntimeError('Malformed arXiv RSS; discovery not committed') from exc
    channel = root.find('channel') if root.tag == 'rss' else None
    if channel is None:
        raise RuntimeError('arXiv RSS channel is missing')
    items = channel.findall('item')
    # Fetch separate categories instead of the capped multi-category feed. Fail
    # closed at the documented feed cap rather than certify a truncated day.
    if len(items) >= 2000:
        raise RuntimeError('arXiv RSS reached 2000 entries; cannot certify a complete announcement feed')
    rows = []
    for item in items:
        kind = item.findtext(ARXIV + 'announce_type')
        if kind not in ('new', 'replace', 'cross', 'replace-cross'):
            raise RuntimeError(f'Unknown arXiv announcement type: {kind}')
        identifier = re.fullmatch(r'oai:arXiv.org:(.+v[1-9]\d*)', item.findtext('guid') or '')
        title = (item.findtext('title') or '').strip()
        description = item.findtext('description') or ''
        abstract = description.partition('Abstract:')[2].strip()
        pub_date = parsedate_to_datetime(item.findtext('pubDate') or '')
        if not identifier or not title or not abstract or pub_date.tzinfo is None:
            raise RuntimeError('arXiv RSS entry lacks version id, title, abstract or announcement date')
        rows.append({'url': 'https://arxiv.org/abs/' + identifier.group(1),
                     'title': title, 'abstract': abstract.replace('\n', ' '),
                     'authors': [(item.findtext(DC + 'creator') or '').strip()],
                     'arxiv_categories': [c.text for c in item.findall('category')],
                     # Keep the official listing date as supplied, without
                     # interpreting its midnight date label as a release instant.
                     'announcement_date': pub_date.date().isoformat(),
                     'announcement_pub_date': pub_date.isoformat(),
                     'announce_type': kind})
    return rows


def fetch_announcements(categories=DEFAULT_CATEGORIES):
    rows = {}
    with requests.Session() as session:
        session.headers['User-Agent'] = 'llm-paper-filter/1.0 (arXiv daily announcements)'
        for index, category in enumerate(dict.fromkeys(categories)):
            category_set(category)
            if index:
                time.sleep(3)
            for attempt in range(5):
                try:
                    response = session.get('https://rss.arxiv.org/rss/' + category, timeout=60)
                    response.raise_for_status()
                    break
                except requests.RequestException as exc:
                    if attempt == 4:
                        raise RuntimeError('arXiv RSS failed; discovery not committed') from exc
                    delay = 3 * (2 ** attempt)
                    failed = getattr(exc, 'response', None)
                    retry_after = failed.headers.get('Retry-After') if failed is not None else None
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            from datetime import datetime, timezone
                            delay = max(delay, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                    print(f'arXiv RSS retry in {delay:g}s', flush=True)
                    time.sleep(delay)
            for row in parse_announcements(response.content):
                # A cross-list may duplicate a primary new/replacement listing.
                previous = rows.get(row['url'])
                if previous and previous['announcement_date'] != row['announcement_date']:
                    raise RuntimeError('Category feeds disagree on announcement date; retry discovery')
                if previous is None or row['announce_type'] in ('new', 'replace'):
                    rows[row['url']] = row
    return sorted(rows.values(), key=lambda r: (r['announcement_date'], r['url']))


def enrich_announcements(announcements, metadata):
    """Keep version submission dates separate from official announcement dates."""
    by_id = {re.sub(r'^https?://arxiv.org/abs/', '', p['url']): p for p in metadata}
    missing = [re.sub(r'^https?://arxiv.org/abs/', '', p['url']) for p in announcements
               if re.sub(r'^https?://arxiv.org/abs/', '', p['url']) not in by_id]
    if missing:
        # RSS can be ahead of OAI indexing. Request exact versions, never substitute
        # a newer version whose announcement date belongs to another batch.
        import arxiv
        client = arxiv.Client(page_size=100)
        for offset in range(0, len(missing), 100):
            search = arxiv.Search(id_list=missing[offset:offset + 100], max_results=None)
            for p in client.results(search):
                identifier = p.get_short_id()
                version = int(re.search(r'v(\d+)$', identifier).group(1))
                by_id[identifier] = {'url': p.entry_id, 'title': p.title,
                    'abstract': p.summary.replace('\n', ' '),
                    'authors': [a.name for a in p.authors],
                    'date': p.updated.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'published': p.published.isoformat(), 'updated': p.updated.isoformat(),
                    'status': 'v1 (New)' if version == 1 else f'v{version} Update',
                    'arxiv_categories': p.categories}
    rows = []
    for announcement in announcements:
        identifier = re.sub(r'^https?://arxiv.org/abs/', '', announcement['url'])
        if identifier not in by_id:
            raise RuntimeError(f'Exact announced version not available yet: {identifier}; retry discovery')
        row = dict(by_id[identifier])
        row.update({key: announcement[key] for key in ('announcement_date', 'announcement_pub_date', 'announce_type')})
        rows.append(row)
    return rows

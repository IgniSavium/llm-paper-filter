"""Fetch accepted main-track metadata from NeurIPS when OpenReview is not public.

The snapshot uses the existing conference corpus format. Provenance identifies
the official website; acceptance is verified from its exact main-track source.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import threading
import time

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from daily_arxiv.cli.common import run
from daily_arxiv.fetching.snapshots import save_snapshot
from daily_arxiv.filtering.pipeline import write_json
from openreview.fetching import BASE_DIR, is_accepted_paper

ORIGIN = 'https://neurips.cc'
LIST_URL = ORIGIN + '/static/virtual/data/neurips-{year}-orals-posters.json'
ABSTRACTS_URL = ORIGIN + '/static/virtual/data/neurips-{year}-abstracts.json'
ABSTRACT_PARSER_VERSION = 1
VOID_TAGS = frozenset(('area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'))


class AbstractParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts = []
        self.found = 0

    def handle_starttag(self, tag, attrs):
        if dict(attrs).get('id') == 'abstractText':
            self.found += 1
            self.depth = 1
        elif self.depth and tag not in VOID_TAGS:
            self.depth += 1
        if self.depth and tag in ('p', 'br'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return
        if self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)


def parse_abstract(html):
    parser = AbstractParser()
    parser.feed(html)
    abstract = ''.join(parser.parts).strip()
    if parser.found != 1 or not abstract:
        raise ValueError('Official paper page has no unique, nonempty abstract')
    return abstract


def main_track_papers(payload, abstracts, year):
    """Deduplicate oral/poster/location events by their OpenReview forum ID."""
    rows = payload.get('results')
    if (not isinstance(rows, list) or not rows or payload.get('next')
            or payload.get('count') != len(rows) or not isinstance(abstracts, dict)):
        raise ValueError('Incomplete official event listing or invalid abstract mapping')
    venueid = f'NeurIPS.cc/{year}/Conference'
    source = 'https://openreview.net/group?id=' + venueid
    papers = {}
    ranks = {'poster': 0, 'spotlight': 1, 'oral': 2}
    for row in rows:
        if row.get('sourceurl') != source:
            continue
        match = re.fullmatch(r'https://openreview.net/forum\?id=([A-Za-z0-9_-]+)', row.get('paper_url') or '')
        decision = re.fullmatch(r'Accept \((poster|spotlight|oral)\)', row.get('decision') or '')
        detail = row.get('virtualsite_url', '')
        title = row.get('name')
        authors = row.get('authors')
        if (not match or not decision or not isinstance(title, str) or not title.strip()
                or not isinstance(authors, list) or not authors
                or any(not isinstance(a, dict) or not isinstance(a.get('fullname'), str)
                       or not a['fullname'].strip() for a in authors)
                or not re.fullmatch(rf'/virtual/{year}/(?:poster|oral)/[0-9]+', detail)):
            raise ValueError(f'Invalid main-track event: {row.get("id")}')
        forum = match[1]
        label = decision[1]
        abstract = abstracts.get(str(row['id']))
        if abstract is not None and (not isinstance(abstract, str) or not abstract.strip()):
            raise ValueError(f'Invalid official abstract: {row["id"]}')
        if forum in papers:
            old = papers[forum]
            if ' '.join(old['title'].split()) != ' '.join(title.split()):
                raise ValueError(f'Conflicting titles for forum {forum}')
            if ranks[label] > ranks[old['venue'].split()[-1]]:
                old['venue'] = f'NeurIPS {year} {label}'
            if not old['abstract'] and abstract:
                old['abstract'] = abstract.strip()
                old['abstract_source_url'] = ABSTRACTS_URL.format(year=year)
            continue
        paper = dict(title=title.strip(), abstract=abstract.strip() if abstract else '',
                     authors=[a['fullname'].strip() for a in authors],
                     url=row['paper_url'], openreview_id=forum, conference='NeurIPS', year=year,
                     venueid=venueid, venue=f'NeurIPS {year} {label}', status=f'NeurIPS {year} Accepted',
                     official_url=ORIGIN + detail, metadata_source='neurips_official_website',
                     abstract_source_url=ABSTRACTS_URL.format(year=year) if abstract else None)
        if row.get('paper_pdf_url'):
            paper['pdf_url'] = row['paper_pdf_url']
        papers[forum] = paper
    if not papers:
        raise ValueError('No explicitly accepted main-track papers in the official list')
    return sorted(papers.values(), key=lambda p: p['openreview_id'])


class WebsiteClient:
    def __init__(self, timeout=30, retries=3, interval=0.2):
        self.timeout, self.retries, self.interval = timeout, retries, interval
        self.lock = threading.Lock()
        self.next_request = 0
        self.local = threading.local()

    def get(self, url):
        if not url.startswith(ORIGIN + '/'):
            raise ValueError('Only official NeurIPS URLs are supported')
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
            self.local.session.headers['User-Agent'] = 'llm-paper-filter/1.0 (conference metadata)'
        for attempt in range(self.retries):
            response = None
            with self.lock:
                time.sleep(max(0, self.next_request - time.monotonic()))
                self.next_request = time.monotonic() + self.interval
            try:
                response = self.local.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt + 1 == self.retries:
                    raise
                delay = 2 ** (attempt + 1)
                if response is not None and response.status_code == 429:
                    try:
                        delay = max(delay, float(response.headers.get('Retry-After', 30)))
                    except ValueError:
                        delay = max(delay, 30)
                    with self.lock:
                        self.next_request = max(self.next_request, time.monotonic() + delay)
                time.sleep(delay)


def fetch_snapshot(year, directory, *, workers=4, client=None):
    client = client or WebsiteClient()
    directory = Path(directory) / f'NeurIPS_{year}'
    cache = directory / '.official_cache'
    cache.mkdir(parents=True, exist_ok=True)
    listing = client.get(LIST_URL.format(year=year)).json()
    abstracts = client.get(ABSTRACTS_URL.format(year=year)).json()
    write_json(cache / 'events.json', listing)
    write_json(cache / 'abstracts.json', abstracts)
    papers = main_track_papers(listing, abstracts, year)
    pending = [p for p in papers if not p['abstract']]
    print(f'Official main track: {len(papers)} unique papers; {len(pending)} abstracts need paper pages', flush=True)

    def recover(paper):
        path = cache / f'{paper["openreview_id"]}.json'
        saved = json.loads(path.read_text()) if path.exists() else {}
        if (saved.get('parser_version') == ABSTRACT_PARSER_VERSION
                and saved.get('title') == paper['title'] and saved.get('url') == paper['official_url']
                and isinstance(saved.get('abstract'), str) and saved['abstract'].strip()):
            abstract = saved['abstract']
        else:
            html = client.get(paper['official_url']).text
            if paper['url'] not in html:
                raise ValueError('Official page does not link to the expected paper')
            abstract = parse_abstract(html)
            write_json(path, dict(title=paper['title'], url=paper['official_url'], abstract=abstract,
                                  parser_version=ABSTRACT_PARSER_VERSION))
        paper.update(abstract=abstract, abstract_source_url=paper['official_url'])

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(recover, paper): paper for paper in pending}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                errors.append((futures[future]['openreview_id'], type(exc).__name__))
            if count % 50 == 0 or count == len(pending):
                print(f'Paper abstracts: {count}/{len(pending)} checked; {len(errors)} errors', flush=True)
    if errors:
        raise RuntimeError(f'{len(errors)} abstracts unavailable; no partial snapshot saved. Rerun to resume. Examples: {errors[:5]}')
    assert all(is_accepted_paper(p, 'NeurIPS', year) for p in papers)
    return save_snapshot(papers, f'{year}-01-01', directory, replace=True, source='openreview',
                         metadata_extra=dict(conference='NeurIPS', conference_year=year,
                                             venueid=f'NeurIPS.cc/{year}/Conference',
                                             paper_status='accepted', track='main', date_basis='conference_year',
                                             discovery_method='neurips_official_website',
                                             api_url=LIST_URL.format(year=year),
                                             coverage='public_official_main_track_listing',
                                             note='Official NeurIPS website metadata, not an OpenReview API harvest; complete for the current public listing, not a final proceedings count.',
                                             fetched_at=datetime.now(timezone.utc).isoformat()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--paper_info_dir', type=Path, default=BASE_DIR / 'paper_info')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1 or not 2000 <= args.year <= datetime.now(timezone.utc).year:
        parser.error('Invalid year or workers')
    print(fetch_snapshot(args.year, args.paper_info_dir, workers=args.workers))


if __name__ == '__main__':
    run(main)

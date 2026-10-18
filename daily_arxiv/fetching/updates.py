"""Discover recently public versions and keep immutable, model-independent batches."""
import fcntl
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from daily_arxiv.fetching.oai import DEFAULT_CATEGORIES, fetch_metadata_updates
from daily_arxiv.fetching.announcements import fetch_announcements, enrich_announcements
from daily_arxiv.fetching.snapshots import load_paper_input, paper_info_digest, save_snapshot
from daily_arxiv.filtering.prompts import PROMPT_SHA256, evaluation_score


def version_key(paper):
    match = re.search(r'/abs/(.+?)/?$', paper['url'])
    if not match:
        raise ValueError(f"Not an arXiv paper URL: {paper['url']}")
    identifier = match.group(1)
    if not re.search(r'v\d+$', identifier):
        version = re.search(r'v(\d+)', paper.get('status', ''))
        if not version:
            raise ValueError(f"Missing arXiv version: {paper['url']}")
        identifier += 'v' + version.group(1)
    return identifier


def snapshot_paths(directory):
    return sorted(Path(directory).glob('papers_*.json')) + sorted((Path(directory) / 'updates').glob('*/papers_*.json'))


def metadata_key(paper):
    """A version can receive bibliographic/administrative updates in place."""
    return version_key(paper), paper_info_digest([paper])


def announcement_target_date(collection_date=None):
    collection_date = collection_date or datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    return (datetime.strptime(collection_date, '%Y-%m-%d').date() - timedelta(days=1)).isoformat()


def _save_sync_json(path, payload):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def discover_announcements(directory, max_results=5000, lookback_days=7, now=None, *,
                     sync_from=None, categories=DEFAULT_CATEGORIES):
    """Checkpoint only complete OAI harvests; submission age never filters papers.

    max_results bounds each immutable output batch, not the harvest. Saved batches
    remain resumable if the cursor commit or later model inference fails.
    """
    if max_results < 1 or lookback_days < 1:
        raise ValueError('Batch size and overlap days must be positive')
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()
    end = announcement_target_date(today)
    start = (datetime.strptime(end, '%Y-%m-%d').date() - timedelta(days=lookback_days - 1)).isoformat()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.updates.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        categories = tuple(dict.fromkeys(categories))
        state_path = directory / 'oai_sync_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        origin_path = directory / 'oai_sync_origin.json'
        if origin_path.exists():
            origin = json.loads(origin_path.read_text())['from']
            datetime.strptime(origin, '%Y-%m-%d')
            if not state or origin < state.get('initial_from', origin):
                # A first harvest or an explicitly extended backfill can fail.
                # Its requested boundary must survive the next default retry.
                start = min(start, origin)
        if state:
            if state.get('schema_version') != 1:
                raise ValueError('Unknown OAI sync state schema; refusing to discard progress')
            previous_day = state.get('last_successful_until', state['last_successful_response_day'])
            datetime.strptime(previous_day, '%Y-%m-%d')
            # Includes the last successful day, with an overlap for boundary/cache delays.
            start = min(start, (datetime.strptime(previous_day, '%Y-%m-%d')
                               - timedelta(days=lookback_days - 1)).date().isoformat())
            if set(categories) - set(state['categories']):
                start = min(start, state['initial_from'])
        if sync_from:
            datetime.strptime(sync_from, '%Y-%m-%d')
            if sync_from > end:
                raise ValueError('--sync_from cannot be after the target announcement date')
            start = min(start, sync_from)
        if not origin_path.exists() or start < origin:
            _save_sync_json(origin_path, {'from': start})
        known, known_announcements, saved_metadata = set(), set(), []
        for path in snapshot_paths(directory):
            papers, _ = load_paper_input(path)
            known.update(version_key(p) for p in papers)
            saved_metadata.extend(papers)
            for paper in papers:
                if paper.get('announcement_date'):
                    known_announcements.add((paper['announcement_date'], version_key(paper)))
        # Capture official announcements before harvesting/enrichment can fail.
        # The live RSS rotates; an archived announcement keeps its original date.
        feed_dir = directory / 'announcement_feeds'
        feed_dir.mkdir(exist_ok=True)
        live = fetch_announcements(categories)
        feed_groups = {}
        for row in live:
            feed_groups.setdefault(row['announcement_date'], []).append(row)
        for day, items in feed_groups.items():
            feed_path = feed_dir / f'feed_{day}.json'
            existing = json.loads(feed_path.read_text())['items'] if feed_path.exists() else []
            merged = {version_key(p): p for p in existing + items}
            _save_sync_json(feed_path, {'announcement_date': day, 'items': list(merged.values())})
        announced = {}
        for feed_path in sorted(feed_dir.glob('feed_*.json')):
            for row in json.loads(feed_path.read_text())['items']:
                identity = (row['announcement_date'], version_key(row))
                if (row['announcement_date'] <= end
                        and set(row['arxiv_categories']).intersection(categories)
                        and identity not in known_announcements):
                    announced[version_key(row)] = row
        print(f'Target daily announcement: {end}; metadata catch-up: {start} through {end}; categories: {", ".join(categories)}', flush=True)
        harvest = fetch_metadata_updates(start, until=end, categories=categories)
        enriched = enrich_announcements(list(announced.values()), saved_metadata + harvest.papers)
        groups = {}
        for row in enriched:
            day = row['announcement_date']
            kind = 'daily_announcements' if day == end else 'catch_up_announcements'
            groups.setdefault((day, kind), []).append(row)
        catch_up = {version_key(p): p for p in harvest.papers
                    if version_key(p) not in known and version_key(p) not in announced}
        if catch_up:
            groups[(today, 'catch_up')] = list(catch_up.values())
        for (date, kind), rows in sorted(groups.items()):
            for offset in range(0, len(rows), max_results):
                batch_rows = rows[offset:offset + max_results]
                digest = paper_info_digest(batch_rows)
                batch = date + '_' + kind + '_' + digest[:12]
                target = directory / 'updates' / batch
                extra = {'batch_id': batch, 'batch_kind': kind,
                    'discovered_at': now.isoformat(), 'collected_on': today,
                    'report_timezone': 'America/New_York' if kind.endswith('announcements') else 'Asia/Shanghai',
                    'query_start_utc': start, 'query_end_utc': end,
                    'discovery_method': 'rss_announcements' if kind.endswith('announcements') else 'oai_pmh',
                    'arxiv_categories': list(categories),
                    'submission_date_range': [min(p['date'] for p in batch_rows), max(p['date'] for p in batch_rows)]}
                stamps = [p['oai_datestamp'] for p in batch_rows if p.get('oai_datestamp')]
                if stamps:
                    extra['metadata_datestamp_range'] = [min(stamps), max(stamps)]
                if kind.endswith('announcements'):
                    extra['announcement_date'] = date
                    extra['date_basis'] = 'official_announcement'
                else:
                    extra['date_basis'] = 'collection_date'
                path = save_snapshot(batch_rows, date, target, source='recently_public',
                                     max_results=max_results, metadata_extra=extra)
                print(f'{kind}: {path} ({len(batch_rows)} versions)', flush=True)
        if not groups:
            print('No new daily announcements or catch-up versions; no empty snapshot written.', flush=True)
        # All category streams and every batch must be durable before advancing.
        response_day = min(now.astimezone(timezone.utc).date().isoformat(), harvest.response_day)
        if state and response_day < state['last_successful_response_day']:
            raise RuntimeError('OAI response is older than saved progress; cursor not advanced')
        next_state = {'schema_version': 1, 'last_successful_response_day': response_day,
                      'last_successful_until': min(end, harvest.response_day),
                      'initial_from': min(start, state.get('initial_from', start)),
                      'categories': list(categories), 'last_completed_at': now.isoformat(),
                      'last_record_count': harvest.record_count,
                      'last_deleted_identifiers': harvest.deleted_identifiers}
        _save_sync_json(state_path, next_state)
    return today, start


def latest_papers(rows):
    """Collapse cross-category races and multiple versions to one latest article."""
    latest = {}
    for paper in rows:
        key = version_key(paper)
        identifier, number = re.fullmatch(r'(.+)v(\d+)', key).groups()
        rank = (int(number), paper.get('oai_datestamp', ''))
        if identifier not in latest or rank > latest[identifier][0]:
            latest[identifier] = (rank, paper)
    return sorted((entry[1] for entry in latest.values()),
                  key=lambda p: (p['oai_datestamp'], version_key(p)))


def discover_updates(directory, max_results=5000, lookback_days=7, now=None, *,
                     sync_from=None, categories=DEFAULT_CATEGORIES, through_date=None):
    """Recover unseen latest versions and batch them by metadata modification day.

    A saved cursor widens the query after outages. Explicit dates elsewhere query
    the provider independently, without excluding versions collected on other days.
    """
    if max_results < 1 or lookback_days < 1:
        raise ValueError('Batch size and overlap days must be positive')
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()
    end = through_date or now.astimezone(timezone.utc).date().isoformat()
    datetime.strptime(end, '%Y-%m-%d')
    if end > now.astimezone(timezone.utc).date().isoformat():
        raise ValueError('Cannot fetch a future metadata modification date')
    start = (datetime.strptime(end, '%Y-%m-%d').date() - timedelta(days=lookback_days - 1)).isoformat()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    categories = tuple(dict.fromkeys(categories))
    with (directory / '.updates.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = directory / 'oai_sync_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        origin_path = directory / 'oai_sync_origin.json'
        origin = json.loads(origin_path.read_text())['from'] if origin_path.exists() else start
        datetime.strptime(origin, '%Y-%m-%d')
        if state:
            if state.get('schema_version') != 1:
                raise ValueError('Unknown OAI sync state schema')
            previous = state.get('last_successful_until', state['last_successful_response_day'])
            start = min(start, (datetime.strptime(previous, '%Y-%m-%d').date()
                                - timedelta(days=lookback_days - 1)).isoformat())
            if set(categories) - set(state['categories']):
                start = min(start, state['initial_from'])
        if not state or origin < state.get('initial_from', origin):
            start = min(start, origin)
        if sync_from:
            datetime.strptime(sync_from, '%Y-%m-%d')
            if sync_from > end:
                raise ValueError('--sync_from must not be after --through_date')
            start = min(start, sync_from)
        if not origin_path.exists() or start < origin:
            _save_sync_json(origin_path, {'from': start})
        known = set()
        for path in snapshot_paths(directory):
            rows, _ = load_paper_input(path)
            known.update(metadata_key(p) for p in rows)
        print(f'Latest public metadata versions: {start} through {end}', flush=True)
        harvest = fetch_metadata_updates(start, until=end, categories=categories)
        groups = {}
        for paper in latest_papers(harvest.papers):
            day = paper['oai_datestamp']
            if not start <= day <= end:
                raise RuntimeError('Provider record is outside requested metadata dates; cursor not advanced')
            if metadata_key(paper) not in known:
                groups.setdefault(day, []).append(paper)
        for day, rows in sorted(groups.items()):
            for offset in range(0, len(rows), max_results):
                batch_rows = rows[offset:offset + max_results]
                batch = day + '_latest_updates_' + paper_info_digest(batch_rows)[:12]
                extra = {'batch_id': batch, 'batch_kind': 'latest_updates',
                         'discovered_at': now.isoformat(), 'collected_on': today,
                         'report_timezone': 'OAI datestamp', 'date_basis': 'metadata_datestamp',
                         'discovery_method': 'oai_pmh', 'arxiv_categories': list(categories),
                         'query_start_utc': start, 'query_end_utc': end,
                         'metadata_datestamp_range': [day, day],
                         'submission_date_range': [min(p['date'] for p in batch_rows), max(p['date'] for p in batch_rows)]}
                path = save_snapshot(batch_rows, day, directory / 'updates' / batch,
                                     source='recently_public', max_results=max_results, metadata_extra=extra)
                print(f'Latest updates for {day}: {path} ({len(batch_rows)} papers)', flush=True)
        if not groups:
            print('No unseen latest versions; pending model batches remain resumable.', flush=True)
        response_day = min(now.astimezone(timezone.utc).date().isoformat(), harvest.response_day)
        if state and response_day < state['last_successful_response_day']:
            raise RuntimeError('Provider response predates saved progress; cursor not advanced')
        _save_sync_json(state_path, {'schema_version': 1,
                       'last_successful_response_day': response_day,
                       'last_successful_until': min(end, harvest.response_day),
                       'initial_from': min(start, state.get('initial_from', start)),
                       'categories': list(categories), 'last_completed_at': now.isoformat(),
                       'last_record_count': harvest.record_count,
                       'last_deleted_identifiers': harvest.deleted_identifiers})
    return today, start


def completed_versions(model_dir, threshold=None, *, with_metadata=False):
    """Completed content stays complete after model, prompt or threshold changes."""
    from daily_arxiv.reporting.artifacts import load_completed_report
    done = set()
    for path in Path(model_dir).rglob('all_papers_*.json'):
        if any(part in path.parts for part in ('archives', '.history')) or any(part.startswith('.report-stage-') for part in path.parts):
            continue
        try:
            date = path.stem.removeprefix('all_papers_')
            reports = load_completed_report(path.parent.parent, date)
            done.update((metadata_key(p) if with_metadata else version_key(p))
                        for p in reports['all_papers']['all_papers_evaluated'])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return done


def pending_batches(directory, model_dir, threshold, start, *, announcement_through=None):
    done = completed_versions(model_dir, threshold, with_metadata=True)
    for path in snapshot_paths(directory):
        rows, date = load_paper_input(path)
        # Old explicit-date snapshots are backfilled only within the discovery window.
        if path.parent == Path(directory) and date < start:
            continue
        payload = json.loads(path.read_text())
        metadata = payload.get('metadata', {}) if isinstance(payload, dict) else {}
        if announcement_through and metadata.get('announcement_date', '') > announcement_through:
            continue
        if metadata.get('discovery_method') == 'rss_announcements':
            # A version reviewed in an OAI catch-up still needs its official
            # daily view. Only that batch's own complete report satisfies it.
            local_done = completed_versions(Path(model_dir) / 'updates' / path.parent.name, threshold, with_metadata=True)
            if rows and any(metadata_key(p) not in local_done for p in rows):
                yield path, date
            continue
        if rows and any(metadata_key(p) not in done for p in rows):
            yield path, date

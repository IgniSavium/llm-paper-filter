"""Model-independent arXiv snapshots shared by every filtering backend."""

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone, timedelta

from daily_arxiv.paths import PAPER_INFO_DIR

DEFAULT_PAPER_INFO_DIR = PAPER_INFO_DIR


def resolve_date(date=None):
    if date is None:
        return str((datetime.now(timezone.utc) - timedelta(days=1)).date())
    datetime.strptime(date, "%Y-%m-%d")
    return date


def raw_papers(papers):
    if not isinstance(papers, list):
        raise ValueError("papers must be a list")
    output = []
    urls = set()
    for paper in papers:
        if not isinstance(paper, dict) or any(not isinstance(paper.get(key), str) or not paper[key].strip()
                                             for key in ("title", "abstract", "url")):
            raise ValueError("Each paper needs nonempty title, abstract, url strings")
        if paper["url"] in urls:
            raise ValueError(f"Duplicate paper URL: {paper['url']}")
        urls.add(paper["url"])
        output.append(copy.deepcopy({key: value for key, value in paper.items()
                                     if key != "eval" and not key.startswith("zh_")}))
    return output


def paper_info_digest(papers):
    serialized = json.dumps(raw_papers(papers), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_paper_input(path, target_date=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stored_date = None
    if isinstance(payload, list):
        papers = payload
    elif isinstance(payload, dict):
        stored_date = payload.get("metadata", {}).get("target_date")
        papers = payload.get("papers", payload.get("all_papers_evaluated"))
    else:
        raise ValueError("Input must be a raw list, papers snapshot, or all_papers_evaluated JSON")
    if target_date and stored_date and target_date != stored_date:
        raise ValueError(f"Input date {stored_date} does not match requested date {target_date}")
    papers = raw_papers(papers)
    if isinstance(payload, dict) and payload.get("metadata", {}).get("paper_info_sha256"):
        if payload["metadata"]["paper_info_sha256"] != paper_info_digest(papers):
            raise ValueError("Paper snapshot checksum does not match its content")
    return papers, resolve_date(target_date or stored_date)


def save_snapshot(papers, date, paper_info_dir=DEFAULT_PAPER_INFO_DIR, *, replace=False, source="arxiv", max_results=None, metadata_extra=None):
    date = resolve_date(date)
    papers = raw_papers(papers)
    path = Path(paper_info_dir) / f"papers_{date}.json"
    digest = paper_info_digest(papers)
    if path.exists() and not replace:
        existing, _ = load_paper_input(path, date)
        if paper_info_digest(existing) != digest:
            raise ValueError(f"Conflicting paper snapshot for {date}: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": {"target_date": date, "total_papers": len(papers),
                            "paper_info_sha256": digest, "source": source,
                            "max_results": max_results}, "papers": papers}
    payload["metadata"].update(metadata_extra or {})
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def get_papers(date=None, paper_info_dir=DEFAULT_PAPER_INFO_DIR, max_results=2000, *, refresh=False):
    """Always query a metadata modification date, independently of local snapshots."""
    date = resolve_date(date)
    directory = Path(paper_info_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"papers_{date}.json"
    # Serialize concurrent first-time fetches so models cannot capture different snapshots.
    with (directory / f".papers_{date}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Explicit-date inputs are independent provider queries. Saved inputs
        # are artifacts, not filters that suppress or freeze future queries.
        papers, date = fetch_papers_by_date(date, max_results=max_results, require_complete=True)
        save_snapshot(papers, date, directory, replace=True, max_results=max_results,
                      source='arxiv_oai_datestamp', metadata_extra={
                          'discovery_method': 'oai_pmh', 'date_basis': 'metadata_datestamp',
                          'arxiv_categories': ['cs.AI', 'cs.LG', 'cs.CL', 'stat.ML']})
        print(f"Shared paper snapshot saved: {path}", flush=True)
        return raw_papers(papers), date


# Shared arXiv query and version labeling are defined below.
def _paper_version_status(paper) -> str:
    """Use the actual version number, including same-day replacements."""
    version_match = re.search(r'v(\d+)$', paper.entry_id)
    if version_match:
        version = int(version_match.group(1))
        return "v1 (New)" if version == 1 else f"v{version} Update"
    return "v1 (New)" if paper.published == paper.updated else "Update"


def fetch_papers_by_date(target_date_str=None, max_results=2000, max_retries=5, retry_delay=3, *, start_date=None, require_complete=True) -> tuple:
    """Fetch latest public versions whose OAI metadata changed on these UTC dates.

    Existing snapshots can reproduce old inputs; newly fetched dates use metadata
    datestamps, never the unsupported lastUpdatedDate search filter. The cap is a
    fail-closed guard for explicit snapshots, regardless of require_complete.
    """
    from daily_arxiv.fetching.oai import fetch_metadata_updates
    target_date = resolve_date(target_date_str)
    if max_results < 1:
        raise ValueError('max_results must be positive')
    harvest = fetch_metadata_updates(start_date or target_date, until=target_date,
                                     max_retries=max_retries, retry_delay=retry_delay)
    from daily_arxiv.fetching.updates import latest_papers
    rows = latest_papers(harvest.papers)
    if len(rows) > max_results:
        raise RuntimeError('Fetch exceeds --max_results; increase it. No partial snapshot saved.')
    print(f'Fetch complete: {len(rows)} latest versions with public metadata changes through {target_date}')
    return rows, target_date


def fetch_date_range(start, end, paper_info_dir=DEFAULT_PAPER_INFO_DIR, max_results=5000, categories=None):
    """Independently query a range, keeping each article only on its latest day."""
    from daily_arxiv.fetching.oai import DEFAULT_CATEGORIES, fetch_metadata_updates
    from daily_arxiv.fetching.updates import latest_papers
    start, end = resolve_date(start), resolve_date(end)
    if start > end or max_results < 1:
        raise ValueError('Invalid date range or max_results')
    selected = tuple(categories or DEFAULT_CATEGORIES)
    harvest = fetch_metadata_updates(start, until=end, categories=selected)
    grouped = {}
    for paper in latest_papers(harvest.papers):
        day = paper['oai_datestamp']
        if not start <= day <= end:
            raise RuntimeError('Provider record is outside the requested range')
        grouped.setdefault(day, []).append(paper)
    if any(len(rows) > max_results for rows in grouped.values()):
        raise RuntimeError('A date exceeds --max_results; no partial date inputs saved')
    paths = []
    day = datetime.strptime(start, '%Y-%m-%d').date()
    final_day = datetime.strptime(end, '%Y-%m-%d').date()
    while day <= final_day:
        date = day.isoformat()
        rows = grouped.get(date, [])
        path = save_snapshot(rows, date, paper_info_dir, replace=True,
                            source='arxiv_oai_datestamp', max_results=max_results,
                            metadata_extra={'discovery_method': 'oai_pmh', 'date_basis': 'metadata_datestamp',
                                            'arxiv_categories': list(selected),
                                            'query_start_utc': start, 'query_end_utc': end,
                                            'metadata_datestamp_range': [date, date]})
        paths.append((path, date))
        day += timedelta(days=1)
    return paths

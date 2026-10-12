"""Merge complete evaluations into one latest-version reading view per day."""
import copy
import json
import re
from collections import defaultdict
from pathlib import Path

from daily_arxiv.fetching.updates import version_key
from daily_arxiv.filtering.pipeline import save_reports
from daily_arxiv.filtering.prompts import PROMPT_SHA256, evaluation_score
from daily_arxiv.fetching.snapshots import paper_info_digest


def merge_latest_reports(model_dir, start, end, threshold=4):
    model_dir = Path(model_dir)
    sources = sorted((model_dir / 'all_papers').glob('all_papers_*.json'))
    sources += sorted((model_dir / 'updates').glob('*/all_papers/all_papers_*.json'))
    latest, dates, categories = {}, set(), []
    for path in sources:
        payload = json.loads(path.read_text())
        meta = payload['metadata']
        day = meta['target_date']
        if (not start <= day <= end or meta.get('date_basis') != 'metadata_datestamp'
                or meta.get('filter_prompt_sha256') != PROMPT_SHA256
                or meta.get('threshold_used') != threshold or meta.get('scoring_mode') != 'per_direction'):
            continue
        rows = payload['all_papers_evaluated']
        digest = paper_info_digest(rows)
        if meta.get('paper_info_sha256') != digest:
            continue
        output = path.parent.parent
        translated_path = output / 'hits_zh' / f'hits_zh_{day}.json'
        if not translated_path.exists():
            continue
        zh = json.loads(translated_path.read_text())
        if zh['hits_zh'] and not (output / 'html_reports' / f'report_{day}.html').exists():
            continue
        if (zh.get('metadata', {}).get('paper_info_sha256') != digest
                or zh['metadata'].get('filter_prompt_sha256') != PROMPT_SHA256
                or zh['metadata'].get('threshold_used') != threshold):
            continue
        translated = {version_key(p): p for p in zh['hits_zh']
                      if p.get('zh_title') and p.get('zh_abstract') and p.get('eval', {}).get('zh_reason')}
        required = {version_key(p) for p in rows if evaluation_score(p['eval']) >= threshold}
        if not required <= translated.keys():
            continue
        dates.add(day)
        for category in meta.get('arxiv_categories', []):
            if category not in categories:
                categories.append(category)
        for original in rows:
            if not start <= original.get('oai_datestamp', '') <= end:
                continue
            key = version_key(original)
            identifier, version = re.fullmatch(r'(.+)v(\d+)', key).groups()
            rank = (int(version), original['oai_datestamp'], path.stat().st_mtime_ns)
            if identifier not in latest or rank > latest[identifier][0]:
                latest[identifier] = (rank, copy.deepcopy(translated.get(key, original)))
    grouped = defaultdict(list)
    for _, row in latest.values():
        grouped[row['oai_datestamp']].append(row)
    reports = []
    for day in sorted(dates):
        rows = sorted(grouped[day], key=version_key)
        hits = [p for p in rows if evaluation_score(p['eval']) >= threshold]
        save_reports(rows, hits, day, threshold, model_dir, input_digest=paper_info_digest(rows),
                     source_metadata={'source': 'latest_daily_view', 'date_basis': 'metadata_datestamp',
                                      'discovery_method': 'oai_pmh', 'arxiv_categories': categories,
                                      'metadata_datestamp_range': [day, day]})
        if hits:
            reports.append(model_dir / 'html_reports' / f'report_{day}.html')
    return reports

"""Merge complete evaluations into one latest-version reading view per day."""
import copy
import re
from collections import defaultdict
from pathlib import Path

from daily_arxiv.fetching.updates import version_key
from daily_arxiv.filtering.pipeline import save_reports
from daily_arxiv.filtering.prompts import evaluation_score
from daily_arxiv.fetching.snapshots import paper_info_digest
from daily_arxiv.reporting.artifacts import load_completed_report, content_key


def merge_latest_reports(model_dir, start, end, threshold=5):
    model_dir = Path(model_dir)
    sources = sorted((model_dir / 'all_papers').glob('all_papers_*.json'))
    sources += sorted((model_dir / 'updates').glob('*/all_papers/all_papers_*.json'))
    latest, dates, categories, prompt_hashes, thresholds = {}, set(), [], defaultdict(set), {}
    translations_by_content = {}
    for path in sources:
        try:
            complete = load_completed_report(path.parent.parent, path.stem.removeprefix('all_papers_'))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        payload = complete['all_papers']
        meta = payload['metadata']
        day = meta['target_date']
        if (not start <= day <= end or meta.get('date_basis') != 'metadata_datestamp'
                or meta.get('scoring_mode') != 'per_direction'):
            continue
        rows = payload['all_papers_evaluated']
        zh = complete['hits_zh']
        translated = {version_key(p): p for p in zh['hits_zh']
                      if p.get('zh_title') and p.get('zh_abstract') and p.get('eval', {}).get('zh_reason')}
        for paper in translated.values():
            translations_by_content[(content_key(paper), paper['eval']['reason'])] = paper
        required = {version_key(p) for p in rows if evaluation_score(p['eval']) >= meta['threshold_used']}
        if not required <= translated.keys():
            continue
        dates.add(day)
        prompt_hashes[day].add(meta.get('filter_prompt_sha256'))
        thresholds.setdefault(day, 5)
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
        previous = translations_by_content.get((content_key(row), row['eval']['reason']))
        if previous is not None and not row.get('zh_abstract'):
            row.update(zh_title=previous['zh_title'], zh_abstract=previous['zh_abstract'])
            row['eval']['zh_reason'] = previous['eval']['zh_reason']
        grouped[row['oai_datestamp']].append(row)
    reports = []
    for day in sorted(dates):
        rows = sorted(grouped[day], key=version_key)
        day_threshold = thresholds[day]
        hits = [p for p in rows if evaluation_score(p['eval']) >= day_threshold]
        save_reports(rows, hits, day, day_threshold, model_dir, input_digest=paper_info_digest(rows),
                     source_metadata={'source': 'latest_daily_view', 'date_basis': 'metadata_datestamp',
                                      'filter_prompt_sha256': next(iter(prompt_hashes[day])) if len(prompt_hashes[day]) == 1 else 'mixed',
                                      'discovery_method': 'oai_pmh', 'arxiv_categories': categories,
                                      'metadata_datestamp_range': [day, day]})
        if hits:
            reports.append(model_dir / 'html_reports' / f'report_{day}.html')
    return reports

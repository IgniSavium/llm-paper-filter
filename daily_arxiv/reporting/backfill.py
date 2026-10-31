"""One-time historical views for the first incremental catch-up."""
import copy
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from daily_arxiv.fetching.updates import version_key
from daily_arxiv.filtering.pipeline import save_reports
from daily_arxiv.filtering.prompts import PROMPT_SHA256, evaluation_score
from daily_arxiv.fetching.snapshots import paper_info_digest, save_snapshot


def backfill_daily_reports(model_dir, paper_info_dir, start, end, today, threshold=5):
    model_dir = Path(model_dir)
    sources = sorted((model_dir / 'all_papers').glob('all_papers_*.json'))
    sources += sorted((model_dir / 'updates').glob('*/all_papers/all_papers_*.json'))
    latest = {}
    for source in sources:
        payload = json.loads(source.read_text())
        meta = payload['metadata']
        if meta.get('filter_prompt_sha256') != PROMPT_SHA256 or meta.get('threshold_used') != threshold:
            continue
        if not start <= meta['target_date'] <= end:
            continue
        zh_path = source.parent.parent / 'hits_zh' / f"hits_zh_{meta['target_date']}.json"
        if not zh_path.exists():
            continue
        translated = {version_key(p): p for p in json.loads(zh_path.read_text())['hits_zh']}
        for original in payload['all_papers_evaluated']:
            key = version_key(original)
            row = copy.deepcopy(translated.get(key, original))
            match = re.fullmatch(r'(.*)v(\d+)', key)
            identifier, version = match.group(1), int(match.group(2))
            row['status'] = 'v1 (New)' if version == 1 else f'v{version} Update'
            if identifier not in latest or version >= latest[identifier][0]:
                latest[identifier] = (version, row)
    days = defaultdict(list)
    for _, row in latest.values():
        date = row['date'][:10]
        if start <= date <= end:
            days[date].append(row)
    if not days:
        print('No completed data available for initial backfill.', flush=True)
        return []
    # Validate everything before replacing any existing reading view.
    for rows in days.values():
        for row in rows:
            if evaluation_score(row['eval']) >= threshold:
                if not (row.get('zh_title') and row.get('zh_abstract') and row['eval'].get('zh_reason')):
                    raise RuntimeError(f"Backfill translation missing: {row['url']}")
    history = model_dir / '.history' / f'initial-backfill-{today}'
    reports = []
    for date, rows in sorted(days.items()):
        rows.sort(key=lambda p: (p['date'], version_key(p)), reverse=True)
        for folder, filename in [('all_papers', f'all_papers_{date}.json'),
                                 ('hits', f'hits_{date}.json'), ('hits_zh', f'hits_zh_{date}.json'),
                                 ('html_reports', f'report_{date}.html')]:
            old = model_dir / folder / filename
            backup = history / folder / filename
            if old.exists() and not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, backup)
        extra = {'source': 'initial_backfill', 'backfilled_on': today,
                 'report_timezone': 'UTC', 'collection_timezone': 'Asia/Shanghai',
                 'submission_date_range': [min(p['date'] for p in rows), max(p['date'] for p in rows)]}
        raw = save_snapshot(rows, date, Path(paper_info_dir) / 'backfills' / today,
                            source='initial_backfill', metadata_extra=extra)
        hits = [p for p in rows if evaluation_score(p['eval']) >= threshold]
        save_reports(rows, hits, date, threshold, model_dir,
                     input_digest=paper_info_digest(rows), source_metadata=extra)
        report = model_dir / 'html_reports' / f'report_{date}.html'
        if hits:
            reports.append(report)
        print(f'Initial backfill: {date}, {len(rows)} versions, {len(hits)} selected; input {raw}', flush=True)
    # Keep the full batch data/cache for recovery, and put its large HTML out of the reading list.
    for path in (model_dir / 'updates').glob(f'{today}_*/html_reports/report_{today}.html'):
        batch = path.parent.parent
        for folder in ('all_papers', 'hits', 'hits_zh'):
            source = batch / folder / f'{folder}_{today}.json'
            if source.exists():
                destination = batch / '.history' / folder / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        destination = batch / '.history' / 'html_reports' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.replace(destination)
    return reports

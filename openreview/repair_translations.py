"""Repair saved translations without fetching papers or recomputing evaluations."""
import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from daily_arxiv.filtering.codex import CodexClient
from daily_arxiv.filtering.pipeline import (
    process_batches, save_reports, translation_contract_sha256,
    validate_translated_paper, write_json,
)
from openreview.fetching import BASE_DIR
from openreview.process_snapshots import reports_complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs_dir', type=Path, default=BASE_DIR / 'runs')
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--workers', type=int, default=1, help='Concurrent GPT requests (default: 1)')
    parser.add_argument('--codex_bin', default='codex')
    parser.add_argument('--include_audit', type=Path,
                        help='Also retranslate the corpus/id pairs in a prior translation_failures.json')
    parser.add_argument('--repair_dir', type=Path, help='Use an existing directory to resume cached repairs')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    forced = set()
    if args.include_audit:
        forced = {(r['corpus'], r['id']) for r in json.loads(args.include_audit.read_text())}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    repair_dir = args.repair_dir or args.runs_dir / '.translation-repairs' / stamp
    corpora, jobs, before = [], defaultdict(list), []
    for output in sorted((args.runs_dir / f'{args.model}_runs').iterdir()):
        if not output.is_dir():
            continue
        files = sorted((output / 'hits_zh').glob('hits_zh_*.json'))
        for file in files:
            payload = json.loads(file.read_text())
            date = payload['metadata']['target_date']
            all_path = output / 'all_papers' / f'all_papers_{date}.json'
            evaluated = json.loads(all_path.read_text())
            marker_path = output / '.queue_complete.json'
            marker = json.loads(marker_path.read_text())
            if marker['model'] != args.model:
                raise ValueError(f'Model mismatch: {marker_path}')
            effort = marker['reasoning_effort']
            corpus = dict(output=output, date=date, payload=payload, evaluated=evaluated, marker=marker)
            corpora.append(corpus)
            for index, paper in enumerate(payload['hits_zh']):
                error = None
                try:
                    validate_translated_paper(paper)
                except ValueError as exc:
                    error = str(exc)
                if error or (output.name, paper['openreview_id']) in forced:
                    jobs[effort].append((corpus, index, paper))
                    before.append(dict(corpus=output.name, id=paper['openreview_id'],
                                       error=error or 'Included by prior audit'))
    print(f'Checked {len(corpora)} corpora; {len(before)} papers require translation repair.', flush=True)
    for item in before:
        print(f"{item['corpus']} {item['id']}: {item['error']}", flush=True)
    if args.dry_run:
        return 0
    # Cache every new result before touching published local reports. A failed run
    # can be resumed with --repair_dir, leaving the old reports intact.
    repair_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = repair_dir / 'targets.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest = {(item['corpus'], item['id']): item for item in previous}
    for item in before:
        manifest.setdefault((item['corpus'], item['id']), item)
    write_json(manifest_path, list(manifest.values()))
    pending_updates = []
    cache_dirs = []
    for effort, items in jobs.items():
        client = CodexClient(args.codex_bin, effort=effort, model=args.model)
        client.check_login()
        cache_dir = repair_dir / 'cache' / effort
        repaired = process_batches([paper for _, _, paper in items], client, cache_dir,
                                   batch_size=1, translation=True, workers=args.workers)
        cache_dirs.append(cache_dir)
        for (corpus, index, original), paper in zip(items, repaired):
            validate_translated_paper(paper)
            pending_updates.append((corpus, index, paper))
    for corpus, index, paper in pending_updates:
        corpus['payload']['hits_zh'][index] = paper
    # Validate the full corpus before replacing any files, not just repaired rows.
    for corpus in corpora:
        for paper in corpus['payload']['hits_zh']:
            validate_translated_paper(paper)
    backup_root = repair_dir / 'backup'
    for corpus in corpora:
        output, date = corpus['output'], corpus['date']
        for file in [output / folder / f'{folder}_{date}.json'
                     for folder in ('all_papers', 'hits', 'hits_zh')] + [
                         output / 'html_reports' / f'report_{date}.html', output / '.queue_complete.json']:
            if file.exists():
                destination = backup_root / output.name / file.relative_to(output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(file, destination)
        metadata = copy.deepcopy(corpus['evaluated']['metadata'])
        for key in ('total_papers_fetched', 'accepted_papers', 'submitted_papers', 'translation_contract_sha256'):
            metadata.pop(key, None)
        save_reports(corpus['evaluated']['all_papers_evaluated'], corpus['payload']['hits_zh'], date,
                     corpus['marker']['threshold'], output,
                     input_digest=corpus['marker']['paper_info_sha256'], source_metadata=metadata)
        identity = dict(corpus['marker'], translation_contract_sha256=translation_contract_sha256())
        if not reports_complete(output, date, identity, check_marker=False):
            raise ValueError(f'Repaired reports failed completion validation: {output}')
        write_json(output / '.queue_complete.json', identity)
        print(f'Validated and rebuilt {output.name}', flush=True)
    # Repair cache keys exclude input positions and include model/effort, so the
    # same entries can safely replace their corresponding conference cache files.
    repaired_corpora = {str(corpus['output']): corpus['output'] for corpus, _, _ in pending_updates}
    for output in repaired_corpora.values():
        destination = output / '.codex_cache' / 'translation'
        destination.mkdir(parents=True, exist_ok=True)
        # Copying only already-owned keys prevents unrelated conference entries.
        for cache_dir in cache_dirs:
            for file in (cache_dir / 'translation').glob('*.json'):
                if (destination / file.name).exists():
                    original = backup_root / output.name / '.codex_cache' / 'translation' / file.name
                    original.parent.mkdir(parents=True, exist_ok=True)
                    if not original.exists():
                        shutil.copy2(destination / file.name, original)
                    shutil.copy2(file, destination / file.name)
    write_json(repair_dir / 'complete.json', {'repaired_papers': len(manifest),
                                            'selected_this_pass': len(before),
                                            'corpora': len(corpora), 'finished_at': datetime.now(timezone.utc).isoformat()})
    print(f'Repaired {len(before)} papers. Backups and manifest: {repair_dir}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

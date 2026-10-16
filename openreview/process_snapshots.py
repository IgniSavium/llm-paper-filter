"""Process saved conference snapshots newest first, retrying from inference caches."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from daily_arxiv.cli.common import run
from daily_arxiv.fetching.snapshots import load_paper_input, paper_info_digest
from daily_arxiv.filtering.pipeline import write_json, translation_contract_sha256, validate_translated_paper
from daily_arxiv.filtering.prompts import PROMPT_SHA256
from openreview.fetching import BASE_DIR, CONFERENCES, PAPER_STATUSES, corpus_name, is_accepted_paper, is_submitted_paper, venue_id
from openreview.run_openreview import build_command


def snapshot_queue(directory, paper_status='accepted'):
    jobs = []
    for path in Path(directory).glob('*/papers_*.json'):
        metadata = json.loads(path.read_text(encoding='utf-8')).get('metadata', {})
        conference, year = metadata.get('conference'), metadata.get('conference_year')
        if metadata.get('source') != 'openreview' or conference not in CONFERENCES or type(year) is not int:
            raise ValueError(f'Invalid OpenReview snapshot metadata: {path}')
        status = metadata.get('paper_status', 'accepted')
        if status not in PAPER_STATUSES:
            raise ValueError(f'Invalid paper status: {path}')
        if status != paper_status:
            continue
        papers, date = load_paper_input(path, f'{year}-01-01')
        if not papers:
            continue
        if metadata.get('venueid') != venue_id(conference, year):
            raise ValueError(f'Mismatched conference snapshot: {path}')
        predicate = is_accepted_paper if status == 'accepted' else is_submitted_paper
        if any(not predicate(p, conference, year) for p in papers):
            raise ValueError(f'Snapshot contains papers without explicit {status} status: {path}; clean or refresh it')
        jobs.append({'conference': conference, 'year': year, 'papers': len(papers),
                     'source': str(path.resolve()), 'date': date,
                     'paper_info_sha256': paper_info_digest(papers)})
    # ICLR n+1 > NeurIPS n > ICML n. Missing venues simply do not enter the queue.
    jobs.sort(key=lambda job: (job['year'], {'ICLR': 0, 'ICML': 1, 'NeurIPS': 2}[job['conference']]), reverse=True)
    identities = [(job['conference'], job['year']) for job in jobs]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate conference/year snapshots')
    return jobs


def retry_delay(attempt, initial, maximum):
    return min(initial * 2 ** min(attempt - 1, 20), maximum)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def reports_complete(output, date, identity, check_marker=True):
    """A completion marker is valid only while its translated artifacts validate."""
    try:
        if check_marker and json.loads((output / '.queue_complete.json').read_text()) != identity:
            return False
        reports = {folder: json.loads((output / folder / f'{folder}_{date}.json').read_text())
                   for folder in ('all_papers', 'hits', 'hits_zh')}
        for report in reports.values():
            if any(report['metadata'].get(key) != identity[key] for key in
                   ('paper_info_sha256', 'filter_prompt_sha256', 'translation_contract_sha256')):
                return False
        hits, translated = reports['hits']['hits'], reports['hits_zh']['hits_zh']
        if len(hits) != len(translated):
            return False
        for paper, original in zip(translated, hits):
            english = {k: v for k, v in paper.items() if not k.startswith('zh_')}
            english['eval'] = {k: v for k, v in english['eval'].items() if k != 'zh_reason'}
            if english != original:
                return False
            validate_translated_paper(paper)
        return not hits or (output / 'html_reports' / f'report_{date}.html').is_file()
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper_info_dir', type=Path, default=BASE_DIR / 'paper_info')
    parser.add_argument('--runs_dir', type=Path, default=BASE_DIR / 'runs')
    parser.add_argument('--paper_status', choices=PAPER_STATUSES, default='accepted')
    parser.add_argument('--model', choices=('gpt-6-luna', 'gpt-5.6-luna'), default='gpt-6-luna')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threshold', type=int, choices=range(1, 6), default=4)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--translation_batch_size', type=int, default=10)
    parser.add_argument('--reasoning_effort', choices=('low', 'medium', 'high', 'xhigh', 'max'), default='medium')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--max_retries', type=int, default=3)
    parser.add_argument('--codex_bin', default='codex')
    parser.add_argument('--retry_delay', type=int, default=300)
    parser.add_argument('--max_retry_delay', type=int, default=1800)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args(argv)
    args.limit = None
    for option in ('workers', 'batch_size', 'translation_batch_size', 'timeout', 'max_retries',
                   'retry_delay', 'max_retry_delay'):
        if getattr(args, option) < 1:
            parser.error(f'--{option} must be positive')
    jobs = snapshot_queue(args.paper_info_dir, args.paper_status)
    if not jobs:
        raise ValueError('No saved conference snapshots found')
    print('Queue: ' + ' → '.join(f"{j['conference']} {j['year']} ({j['papers']})" for j in jobs), flush=True)
    print(f"Total: {sum(j['papers'] for j in jobs)} papers; model={args.model}, workers={args.workers}, "
          f"threshold={args.threshold}, effort={args.reasoning_effort}", flush=True)
    if args.dry_run:
        return 0
    state_dir = args.runs_dir.resolve() / '.queue' / args.model
    if args.paper_status == 'submitted':
        state_dir = state_dir / 'submitted'
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / 'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'An OpenReview {args.model} queue is already running') from None
        state_path = state_dir / 'state.json'
        state = {'status': 'running', 'started_at': stamp(), 'model': args.model, 'queue': jobs,
                 'completed': [], 'total_papers': sum(j['papers'] for j in jobs)}
        for job in jobs:
            name = corpus_name(job['conference'], job['year'], args.paper_status)
            output = args.runs_dir.resolve() / f'{args.model}_runs' / name
            identity = {key: getattr(args, key) for key in ('model', 'threshold', 'reasoning_effort')}
            identity.update(paper_info_sha256=job['paper_info_sha256'], filter_prompt_sha256=PROMPT_SHA256)
            identity['translation_contract_sha256'] = translation_contract_sha256()
            marker = output / '.queue_complete.json'
            if reports_complete(output, job['date'], identity):
                print(f'Already complete: {name}', flush=True)
                state['completed'].append(name)
                continue
            # Freeze the input so a simultaneous provider refresh cannot change this job.
            source = state_dir / 'inputs' / f"{name}_{job['paper_info_sha256']}.json"
            payload = json.loads(Path(job['source']).read_text(encoding='utf-8'))
            papers, _ = load_paper_input(job['source'], job['date'])
            if paper_info_digest(papers) != job['paper_info_sha256']:
                raise RuntimeError('A snapshot changed while building the queue; rerun to rebuild it')
            payload['papers'] = papers
            write_json(source, payload)
            command = build_command(args, job['conference'], job['year'], source, args.model)
            log_path = state_dir / f'{name}.log'
            attempt = 0
            while True:
                attempt += 1
                state.update(status='running', current_job=name, attempt=attempt, job_log=str(log_path), updated_at=stamp())
                state.pop('retry_at', None)
                write_json(state_path, state)
                print(f'[{stamp()}] Start {name}, attempt {attempt}; log={log_path}', flush=True)
                with log_path.open('a', encoding='utf-8') as log:
                    log.write(f'\n[{stamp()}] Start {name}, attempt {attempt}\n')
                    log.flush()
                    result = subprocess.run(command, cwd=BASE_DIR.parent, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode == 0 and not reports_complete(output, job['date'], identity, check_marker=False):
                    print(f'{name}: generated reports failed validation; retrying from caches', flush=True)
                    result.returncode = 1
                if result.returncode == 0:
                    write_json(marker, identity)
                    state['completed'].append(name)
                    print(f'[{stamp()}] Completed {name}', flush=True)
                    break
                delay = retry_delay(attempt, args.retry_delay, args.max_retry_delay)
                state.update(status='retry_wait', last_exit_code=result.returncode,
                             retry_at=datetime.fromtimestamp(time.time() + delay, timezone.utc).isoformat(), updated_at=stamp())
                write_json(state_path, state)
                print(f'[{stamp()}] {name} exited {result.returncode}; retry in {delay}s using cached batches', flush=True)
                time.sleep(delay)
        state.update(status='complete', current_job=None, finished_at=stamp())
        write_json(state_path, state)
        print(f'[{stamp()}] All {len(jobs)} conference snapshots completed.', flush=True)
    return 0


if __name__ == '__main__':
    run(main)

"""Discover new public versions or reproduce a date, then run selected model filters."""

import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from daily_arxiv.fetching.updates import discover_updates, pending_batches
from daily_arxiv.fetching.oai import DEFAULT_CATEGORIES, category_set
from daily_arxiv.fetching.snapshots import DEFAULT_PAPER_INFO_DIR, get_papers, paper_info_digest, resolve_date, fetch_date_range

from daily_arxiv.paths import DAILY_DIR

BASE_DIR = DAILY_DIR
LOCAL_MODELS = {
    "Llama-3.3-70B-Instruct": ("daily_arxiv_filter.py", "llama_model_path"),
    "Meta-Llama-3.1-70B-Instruct-AWQ-INT4": (
        "other_local_models/daily_arxiv_filter_Llama3.1_70B_INT4.py", "llama31_model_path"),
    "Qwen2.5-72B-Instruct": (
        "other_local_models/daily_arxiv_filter_Qwen2.5_72B_Instruct.py", "qwen_model_path"),
}
MODELS = ("gpt-6-luna", "gpt-5.6-luna", *LOCAL_MODELS)


def build_commands(args, input_path, date, batch=None):
    commands = []
    for model in dict.fromkeys(args.models):
        output = Path(args.runs_dir).resolve() / f"{model}_runs"
        if batch:
            output = output / "updates" / batch
        common = ["--date", date, "--input_json", str(input_path),
                  "--threshold", str(args.threshold), "--save_dir", str(output)]
        if model.startswith("gpt-"):
            command = [sys.executable, str(BASE_DIR / "daily_arxiv_filter_codex.py"),
                       "--model", model, "--codex_bin", args.codex_bin,
                       "--workers", str(args.workers),
                       "--batch_size", str(args.batch_size),
                       "--translation_batch_size", str(args.translation_batch_size),
                       "--reasoning_effort", args.reasoning_effort,
                       "--timeout", str(args.timeout), "--max_retries", str(args.max_retries)]
            if batch:
                # Daily and catch-up views can contain the same version. Content,
                # prompt and model are already part of the cache identity.
                command += ['--cache_dir', str(Path(args.runs_dir).resolve() / f'{model}_runs' / '.codex_cache')]
        else:
            entry_point, path_option = LOCAL_MODELS[model]
            command = [sys.executable, str(BASE_DIR / entry_point),
                       "--model_path", str(Path(getattr(args, path_option)).resolve()),
                       "--tp_size", str(args.tp_size), "--gpu_util", str(args.gpu_util),
                       "--max_num_seqs", str(args.max_num_seqs),
                       "--batch_size", str(args.batch_size),
                       "--translation_batch_size", str(args.translation_batch_size),
                       "--max_retries", str(args.max_retries),
                       "--max_model_len", str(args.max_model_len),
                       "--evaluation_tokens", str(args.evaluation_tokens),
                       "--translation_tokens", str(args.translation_tokens),
                       "--dtype", args.dtype]
            if batch:
                command += ['--cache_dir', str(Path(args.runs_dir).resolve() / f'{model}_runs' / '.local_cache')]
        commands.append((model, command + common))
    return commands



def run_updates(args):
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    with (args.runs_dir / '.update_runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        today, start = discover_updates(args.paper_info_dir, args.max_results, args.lookback_days,
                                        sync_from=args.sync_from, categories=args.categories, through_date=args.through_date)
        ran = False
        selected = args.models
        for model in dict.fromkeys(selected):
            model_dir = args.runs_dir / f'{model}_runs'
            for source, date in pending_batches(args.paper_info_dir, model_dir, args.threshold, start):
                batch = source.parent.name if source.parent.parent.name == 'updates' else None
                args.models = [model]
                _, command = build_commands(args, source, date, batch)[0]
                batch_meta = json.loads(source.read_text()).get('metadata', {})
                print(f'[{today}] {batch_meta.get("batch_kind", "legacy_batch")} — resume/run {model}: {source}', flush=True)
                result = subprocess.run(command, cwd=BASE_DIR.parent)
                if result.returncode:
                    print('Update interrupted; rerun to resume the saved shared batch and inference caches.', flush=True)
                    return result.returncode if result.returncode > 0 else 1
                output = model_dir / 'updates' / batch if batch else model_dir
                ran = True
                print(f'Completed report: {output / "html_reports" / ("report_" + date + ".html")}', flush=True)
            if args.initial_backfill:
                from daily_arxiv.reporting.backfill import backfill_daily_reports
                backfill_daily_reports(model_dir, args.paper_info_dir, start, today, today, args.threshold)
            from daily_arxiv.reporting.latest import merge_latest_reports
            merge_latest_reports(model_dir, start, args.through_date or today, args.threshold)
            reports = sorted((p for p in model_dir.rglob('report_*.html') if 'archives' not in p.parts and '.history' not in p.parts),
                             key=lambda p: (p.stem.removeprefix("report_"), p.stat().st_mtime))
            if reports:
                daily = [p for p in reports if '_latest_updates_' in str(p)]
                if daily:
                    print(f'Latest metadata update {model} report: {daily[-1]}', flush=True)
                catch_up = [p for p in reports if '_catch_up' in str(p)]
                if catch_up:
                    print(f'Latest catch-up {model} report: {catch_up[-1]}', flush=True)
                if not daily and not catch_up:
                    print(f'Latest readable {model} report: {reports[-1]}', flush=True)
        if not ran:
            print('No pending model work. Inference and empty reports skipped.', flush=True)
        args.models = selected
    return 0

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Independently query a metadata modification date; default: recover latest versions through today")
    parser.add_argument("--initial_backfill", action="store_true", help="One-time: merge completed recent versions into their UTC submission-date reading reports")
    parser.add_argument("--lookback_days", type=int, default=7, help="Overlap days for metadata synchronization; outages are automatically caught up")
    parser.add_argument("--from_date", help="Independently recover a date range, with one HTML per metadata date")
    parser.add_argument("--through_date", help="Last metadata date to query; defaults to today UTC")
    parser.add_argument("--sync_from", help="Extend discovery back to this UTC metadata date (YYYY-MM-DD)")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES), help="arXiv categories to synchronize")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["gpt-6-luna"])
    parser.add_argument("--threshold", type=int, choices=(5,), default=5,
                        help="Only five-point arXiv papers are translated and retained as hits")
    parser.add_argument("--paper_info_dir", type=Path, default=DEFAULT_PAPER_INFO_DIR)
    parser.add_argument("--runs_dir", type=Path, default=BASE_DIR / "runs")
    parser.add_argument("--max_results", type=int, default=5000, help="Papers per automatic batch; fail-closed cap for explicit dates")
    parser.add_argument("--refresh", action="store_true", help="Explicitly refresh the shared arXiv snapshot")
    parser.add_argument("--dry_run", action="store_true", help="Print planned commands without fetching or inference")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent Luna requests (each keeps its own small input batch)")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--translation_batch_size", type=int, default=10)
    parser.add_argument("--reasoning_effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--codex_bin", default="codex")
    parser.add_argument("--llama_model_path", type=Path,
                        default=BASE_DIR.parent.parent / "weights" / "Llama-3.3-70B-Instruct")
    parser.add_argument("--llama31_model_path", type=Path,
                        default=BASE_DIR.parent.parent / "weights" / "Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    parser.add_argument("--qwen_model_path", type=Path,
                        default=BASE_DIR.parent.parent / "weights" / "Qwen2.5-72B-Instruct")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--evaluation_tokens", type=int, default=2048)
    parser.add_argument("--translation_tokens", type=int, default=8192)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--gpu_util", type=float, default=0.90)
    parser.add_argument("--max_num_seqs", type=int, default=128)
    args = parser.parse_args(argv)
    for option in ("workers", "lookback_days", "max_results", "batch_size", "translation_batch_size", "timeout",
                   "max_retries", "tp_size", "max_num_seqs", "max_model_len", "evaluation_tokens", "translation_tokens"):
        if getattr(args, option) < 1:
            parser.error(f"--{option} must be positive")
    if not 0 < args.gpu_util <= 1:
        parser.error("--gpu_util must be greater than 0 and at most 1")
    try:
        date = resolve_date(args.date) if args.date is not None else None
        if args.sync_from:
            resolve_date(args.sync_from)
        if args.from_date:
            resolve_date(args.from_date)
        if args.through_date:
            resolve_date(args.through_date)
        for category in args.categories:
            category_set(category)
    except ValueError:
        parser.error("Dates must be YYYY-MM-DD and categories must be valid arXiv categories")
    if args.date is not None and (args.from_date or args.through_date):
        parser.error("--date cannot be combined with a date range")
    if args.from_date and (args.sync_from or args.initial_backfill):
        parser.error("Independent date ranges cannot be combined with incremental backfill options")
    if args.date is not None and (args.sync_from or tuple(args.categories) != DEFAULT_CATEGORIES):
        parser.error("--sync_from and custom --categories apply to automatic discovery only")
    if args.initial_backfill and args.date is not None:
        parser.error("--initial_backfill is only for automatic updates")
    args.paper_info_dir = args.paper_info_dir.resolve()
    if args.from_date:
        from datetime import datetime, timezone
        end = args.through_date or datetime.now(timezone.utc).date().isoformat()
        if args.from_date > end:
            parser.error('--from_date must not be after --through_date')
        if args.dry_run:
            print(f'Independent latest-version query: {args.from_date} through {end}; one report per metadata day')
            return 0
        inputs = fetch_date_range(args.from_date, end, args.paper_info_dir, args.max_results, args.categories)
        for source, day in inputs:
            for model, command in build_commands(args, source, day):
                print(f'Run {model} latest metadata updates for {day}: {source}', flush=True)
                result = subprocess.run(command, cwd=BASE_DIR.parent)
                if result.returncode:
                    return result.returncode if result.returncode > 0 else 1
        return 0
    if args.date is None:
        if args.refresh:
            parser.error("--refresh applies only to explicit --date snapshots")
        if args.dry_run:
            print(f"Recover unseen latest versions through {args.through_date or 'today UTC'} with {args.lookback_days}-day overlap, sync_from={args.sync_from}, categories={args.categories}; group reports by metadata modification day")
            return 0
        return run_updates(args)
    source = args.paper_info_dir / f"papers_{date}.json"
    if args.dry_run:
        print(f"Shared snapshot: {source}; refresh={args.refresh}")
        for model, command in build_commands(args, source, date):
            print(f"[{model}] {shlex.join(command)}")
        return 0

    papers, date = get_papers(date, args.paper_info_dir, args.max_results, refresh=args.refresh)
    digest = paper_info_digest(papers)
    print(f"Runner: {date}, {len(papers)} papers, input SHA-256 {digest}", flush=True)
    # Freeze this run's corpus: an external refresh between model runs cannot change the input.
    with tempfile.TemporaryDirectory(prefix="paper-filter-run-") as temporary:
        input_path = Path(temporary) / f"papers_{date}.json"
        input_metadata = json.loads(source.read_text()).get("metadata", {}) if source.exists() else {}
        input_metadata.update({"target_date": date, "paper_info_sha256": digest})
        input_path.write_text(json.dumps({"metadata": input_metadata,
                                         "papers": papers}, ensure_ascii=False), encoding="utf-8")
        for index, (model, command) in enumerate(build_commands(args, input_path, date), 1):
            print(f"[{index}/{len(set(args.models))}] Running {model}", flush=True)
            result = subprocess.run(command, cwd=BASE_DIR.parent)
            if result.returncode != 0:
                print(f"Runner stopped: {model} exited with code {result.returncode}. "
                      "Rerun the command to reuse completed inference caches.", file=sys.stderr)
                return result.returncode if result.returncode > 0 else 1
            print(f"Finished {model}: {args.runs_dir.resolve() / (model + '_runs')}", flush=True)
    print("All selected model runs completed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Runner error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)

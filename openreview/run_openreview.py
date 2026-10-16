"""Fetch conference papers, then reuse the daily arXiv Codex filter unchanged."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from daily_arxiv.cli.common import run
from openreview.fetching import BASE_DIR, corpus_name, get_snapshot
from openreview.fetch_openreview_papers import add_fetch_arguments, make_client, selections


def build_command(args, conference, year, source, model):
    command = [sys.executable, "-m", "daily_arxiv.filtering.codex", "--input_json", str(source),
               "--date", f"{year}-01-01", "--model", model,
               "--save_dir", str(args.runs_dir.resolve() / f"{model}_runs" / corpus_name(
                   conference, year, getattr(args, 'paper_status', 'accepted')))]
    for name in ("threshold", "workers", "batch_size", "translation_batch_size", "reasoning_effort",
                 "timeout", "max_retries", "codex_bin", "limit"):
        if getattr(args, name) is not None:
            command.extend(["--" + name, str(getattr(args, name))])
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_fetch_arguments(parser)
    parser.add_argument("--models", nargs="+", choices=("gpt-6-luna", "gpt-5.6-luna"), default=["gpt-6-luna"])
    parser.add_argument("--runs_dir", type=Path, default=BASE_DIR / "runs")
    parser.add_argument("--threshold", type=int, choices=range(1, 6), default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--translation_batch_size", type=int, default=10)
    parser.add_argument("--reasoning_effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--codex_bin", default="codex")
    parser.add_argument("--limit", type=int, help="Filter only the first N papers; fetch remains complete")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without network or inference")
    args = parser.parse_args(argv)
    args.paper_info_dir = args.paper_info_dir.resolve()
    for option in ("workers", "batch_size", "translation_batch_size", "timeout", "max_retries", "limit"):
        if getattr(args, option) is not None and getattr(args, option) < 1:
            parser.error(f"--{option} must be positive")
    selected, client = list(selections(args)), make_client(args)
    for conference, year in selected:
        source = args.paper_info_dir.resolve() / corpus_name(conference, year, args.paper_status) / f"papers_{year}-01-01.json"
        if not args.dry_run:
            source = get_snapshot(conference, year, args.paper_info_dir, refresh=args.refresh, client=client,
                                  paper_status=args.paper_status)
            if source is None:
                continue
        for model in dict.fromkeys(args.models):
            command = build_command(args, conference, year, source, model)
            print(f"[{conference} {year}] {shlex.join(command)}", flush=True)
            if not args.dry_run:
                result = subprocess.run(command, cwd=BASE_DIR.parent)
                if result.returncode:
                    return result.returncode if result.returncode > 0 else 1
    return 0


if __name__ == "__main__":
    run(main)

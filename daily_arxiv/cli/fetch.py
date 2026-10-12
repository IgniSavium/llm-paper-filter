"""Fetch the model-independent daily paper snapshot without running an LLM."""
import argparse
import sys
from daily_arxiv.fetching.updates import discover_updates
from daily_arxiv.fetching.oai import DEFAULT_CATEGORIES, category_set
from daily_arxiv.fetching.snapshots import DEFAULT_PAPER_INFO_DIR, get_papers, fetch_date_range


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Independently query a metadata modification date; default: recover latest versions through today")
    parser.add_argument("--lookback_days", type=int, default=7, help="Metadata synchronization overlap in days")
    parser.add_argument("--from_date", help="Independently fetch a range, with one input per metadata date")
    parser.add_argument("--through_date", help="Last metadata date to query; defaults to today UTC")
    parser.add_argument("--sync_from", help="Extend discovery back to this UTC metadata date (YYYY-MM-DD)")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--paper_info_dir", default=str(DEFAULT_PAPER_INFO_DIR))
    parser.add_argument("--max_results", type=int, default=5000,
                        help="Papers per automatic batch; fail-closed cap for explicit dates")
    parser.add_argument("--refresh", action="store_true", help="Explicitly replace an existing date snapshot")
    args = parser.parse_args()
    if args.max_results < 1 or args.lookback_days < 1:
        parser.error("--max_results and --lookback_days must be positive")
    if args.from_date:
        if args.date or args.sync_from:
            parser.error('--from_date cannot be combined with --date or --sync_from')
        from datetime import datetime, timezone
        end = args.through_date or datetime.now(timezone.utc).date().isoformat()
        paths = fetch_date_range(args.from_date, end, args.paper_info_dir, args.max_results, args.categories)
        print(f'Ready: {len(paths)} independent date inputs; no inference performed.')
        return
    if args.date is None:
        if args.refresh:
            parser.error("--refresh requires --date")
        for category in args.categories:
            category_set(category)
        discover_updates(args.paper_info_dir, args.max_results, args.lookback_days,
                         sync_from=args.sync_from, categories=args.categories, through_date=args.through_date)
        print("Discovery complete. Shared batches are ready; no model inference performed.")
        return
    if args.sync_from or tuple(args.categories) != DEFAULT_CATEGORIES:
        parser.error("--sync_from and custom --categories apply to automatic discovery only")
    papers, date = get_papers(args.date, args.paper_info_dir, args.max_results, refresh=args.refresh)
    print(f"Ready: {date}, {len(papers)} papers. No model inference performed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

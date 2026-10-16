"""Fetch accepted-paper or public-submission snapshots without LLM inference."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from daily_arxiv.cli.common import run
from openreview.fetching import BASE_DIR, CONFERENCES, DEFAULT_CREDENTIALS_FILE, PAPER_STATUSES, OpenReviewClient, default_years, get_snapshot, load_credentials, venue_id


def add_fetch_arguments(parser):
    parser.add_argument("--conferences", nargs="+", choices=CONFERENCES, default=list(CONFERENCES))
    parser.add_argument('--paper_status', choices=PAPER_STATUSES, default='accepted',
                        help='accepted: recorded acceptances; submitted: public active submissions, excluding decisions/withdrawals')
    parser.add_argument("--years", nargs="+", type=int,
                        help='Default: previous three years plus current for accepted; current and next for submitted')
    parser.add_argument("--paper_info_dir", type=Path, default=BASE_DIR / "paper_info")
    parser.add_argument("--refresh", action="store_true", help="Replace saved snapshots after a complete fetch")
    parser.add_argument("--fetch_timeout", type=int, default=60)
    parser.add_argument("--fetch_retries", type=int, default=5)
    parser.add_argument("--page_size", type=int, default=1000)
    parser.add_argument("--credentials_file", type=Path,
                        help=f"Local credentials JSON; defaults to {DEFAULT_CREDENTIALS_FILE} if present")


def selections(args):
    years = args.years if args.years is not None else (
        [default_years()[-1], default_years()[-1] + 1] if args.paper_status == 'submitted' else default_years())
    for conference in dict.fromkeys(args.conferences):
        for year in sorted(set(years)):
            venue_id(conference, year)
            yield conference, year


def make_client(args):
    credentials = {} if getattr(args, 'dry_run', False) else load_credentials(
        args.credentials_file or DEFAULT_CREDENTIALS_FILE, required=args.credentials_file is not None)
    return OpenReviewClient(timeout=args.fetch_timeout, retries=args.fetch_retries,
                           page_size=args.page_size, credentials=credentials)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_fetch_arguments(parser)
    args = parser.parse_args(argv)
    selected, client = list(selections(args)), make_client(args)
    for conference, year in selected:
        path = get_snapshot(conference, year, args.paper_info_dir, refresh=args.refresh, client=client,
                            paper_status=args.paper_status)
        if path is not None:
            print(f"{conference} {year}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    run(main)

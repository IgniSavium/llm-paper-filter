"""Command-line entry point; implementation lives in daily_arxiv.filtering.local."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from daily_arxiv.cli.common import run
from daily_arxiv.filtering.local import main

if __name__ == '__main__':
    run(main)

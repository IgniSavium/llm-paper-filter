"""Project paths shared by CLI wrappers and internal packages."""
from pathlib import Path

DAILY_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DAILY_DIR.parent
PAPER_INFO_DIR = DAILY_DIR / 'paper_info'
RUNS_DIR = DAILY_DIR / 'runs'
TEMPLATE_PATH = DAILY_DIR / 'reporting' / 'templates' / 'report.html'
KATEX_DIR = DAILY_DIR / 'vendor' / 'katex'
SCHEDULER_ENTRY = DAILY_DIR / 'scheduler.py'

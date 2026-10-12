"""Compatibility entry point for Qwen2.5-72B-Instruct using the shared local pipeline."""
from pathlib import Path
import sys
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from daily_arxiv.filtering.local import main
from daily_arxiv.cli.common import run

if __name__ == '__main__':
    run(partial(main, default_model_path='./weights/Qwen2.5-72B-Instruct'))

#!/usr/bin/env bash
# Compatibility wrapper: forwards arguments to the shared GPT/local-model runner.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$script_dir/run_daily_arxiv.py" "$@"

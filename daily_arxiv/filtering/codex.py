"""Filter and translate arXiv papers using ChatGPT-authenticated Codex CLI."""

import argparse
from daily_arxiv.fetching.snapshots import DEFAULT_PAPER_INFO_DIR, get_papers, load_paper_input, paper_info_digest
from daily_arxiv.filtering.prompts import CATEGORIES, PROMPT_SHA256, SYSTEM_PROMPT, validate_evaluation, normalize_evaluation, evaluation_score, format_categories
from daily_arxiv.reporting.html import generate_mobile_html_report
import json
import os
from pathlib import Path
from daily_arxiv.paths import RUNS_DIR
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta


MODEL = "gpt-6-luna"
from daily_arxiv.filtering.pipeline import (
    MATH_PATTERN, TRANSLATION_PROMPT, DATA_BOUNDARY, TranslationContentError,
    result_schema, validate_results, write_json, validate_translation_math,
    split_translation_text, assemble_translation, decode_translation_parts,
    process_batches, save_reports, load_filter_input, paper_prompt, run_filter, reuse_completed_input, needs_inference,
)


class ModelAccessError(RuntimeError):
    """A permanent model-access failure that retrying will not resolve."""


class CodexClient:
    def __init__(self, executable="codex", effort="medium", timeout=300, retries=3, model=MODEL):
        self.executable = shutil.which(executable)
        if self.executable is None:
            raise RuntimeError("Codex CLI not found. Install it with npm install -g @openai/codex")
        self.effort = effort
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.env = os.environ.copy()
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"):
            self.env.pop(key, None)

    def check_login(self):
        status = subprocess.run([self.executable, "login", "status"],
                                capture_output=True, text=True, env=self.env, timeout=30)
        message = status.stdout + status.stderr
        if status.returncode != 0 or "logged in using chatgpt" not in message.lower():
            raise RuntimeError("ChatGPT subscription login required. Run codex login (or "
                               "codex login --device-auth), then codex login status. "
                               "API-key login is intentionally refused.")
        help_result = subprocess.run([self.executable, "exec", "--help"],
                                     capture_output=True, text=True, env=self.env, timeout=30)
        if help_result.returncode != 0 or "--ignore-user-config" not in help_result.stdout:
            raise RuntimeError("Update Codex CLI: npm install -g @openai/codex. "
                               "This script requires --ignore-user-config and --ephemeral.")

    def request(self, instruction, records, translation=False):
        expected_ids = [record["id"] for record in records]
        # A failed translation batch is split into single papers by the shared
        # pipeline. Remove synthetic formula context from singleton requests so
        # the model cannot keep copying it into the translated prose on retries.
        prompt = paper_prompt(instruction, records, translation,
                              fragments_only=translation and len(records) == 1)
        schema = result_schema(translation, segmented=translation)
        if translation and len(records) == 1:
            # Split/retried papers have a known output shape. Enforce counts at
            # generation time as well as checking them after the response.
            item = schema['properties']['results']['items']['properties']
            item['id']['enum'] = expected_ids
            for field in ('title', 'abstract', 'reason'):
                parts = split_translation_text(records[0][field])[0]
                count = len(parts)
                item[f'zh_{field}_parts'].update(minItems=count, maxItems=count)
                # When every source slot contains prose, enforce nonempty output
                # at generation time. Mixed empty/punctuation slots must remain
                # permitted and are checked individually during reconstruction.
                if all(part.strip() and any(c.isalnum() for c in part) for part in parts):
                    item[f'zh_{field}_parts']['items']['minLength'] = 1
            schema['properties']['results'].update(minItems=1, maxItems=1)
        error = ""
        for attempt in range(self.retries):
            try:
                with tempfile.TemporaryDirectory(prefix="arxiv-codex-") as work_dir:
                    schema_path = Path(work_dir) / "schema.json"
                    output_path = Path(work_dir) / "response.json"
                    write_json(schema_path, schema)
                    command = [
                        self.executable, "exec", "--ignore-user-config", "--ephemeral",
                        "--skip-git-repo-check", "--sandbox", "read-only",
                        "--model", self.model, "--color", "never", "--cd", work_dir,
                        "-c", 'forced_login_method="chatgpt"',
                        "-c", 'model_provider="openai"',
                        "-c", f'model_reasoning_effort="{self.effort}"',
                        "-c", 'web_search="disabled"',
                        "-c", "features.shell_tool=false",
                        "-c", "features.apps=false", "-c", "features.plugins=false",
                        "-c", "features.multi_agent=false",
                        "--output-schema", str(schema_path),
                        "--output-last-message", str(output_path), "-",
                    ]
                    process = subprocess.run(command, input=prompt, capture_output=True,
                                             text=True, env=self.env, timeout=self.timeout)
                    if process.returncode != 0:
                        if "not supported when using codex with a chatgpt account" in process.stderr.lower():
                            raise ModelAccessError(
                                f"{self.model} is not supported for the current ChatGPT login. "
                                "Update Codex CLI first (npm install -g @openai/codex), then "
                                "confirm model availability for this account/workspace. "
                                "No alternative model or API billing was used; completed batches remain cached.")
                        raise RuntimeError(f"Codex exited {process.returncode}: "
                                           f"{process.stderr[-2000:]}")
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                    received = (decode_translation_parts(payload, records) if translation
                                else validate_results(payload, expected_ids))
                    return received
            except ModelAccessError:
                raise
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                error = str(exc)
                if translation and isinstance(exc, ValueError):
                    content_error = exc if isinstance(exc, TranslationContentError) else TranslationContentError(error)
                    if len(records) > 1 or attempt + 1 == self.retries:
                        if content_error is exc:
                            raise
                        raise content_error from exc
                    # A singleton gets a bounded retry with actionable validation feedback.
                    prompt += "\nPrevious response was invalid: " + error + ". Return all text fragments in their original order."
                if attempt + 1 < self.retries:
                    delay = min(2 ** (attempt + 1), 30)
                    print(f"Codex batch failed ({attempt + 1}/{self.retries}); retry in {delay}s: {error}",
                          file=sys.stderr)
                    time.sleep(delay)
        raise RuntimeError(f"Codex batch failed after {self.retries} attempts: {error}. "
                           "Completed batches are cached; rerun to resume. "
                           f"Check subscription limits and {self.model} account availability.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Daily arXiv filter using GPT-6 Luna and Codex subscription")
    parser.add_argument("--date", help="YYYY-MM-DD; defaults to yesterday UTC")
    parser.add_argument("--threshold", type=int, choices=range(1, 6),
                        help="arXiv requires 5; conference default: 4")
    parser.add_argument("--model", choices=("gpt-6-luna", "gpt-5.6-luna"), default=MODEL,
                        help="Explicit Luna model selection; no automatic fallback")
    parser.add_argument("--save_dir", help="Output directory; defaults to daily_arxiv/runs/MODEL_runs")
    parser.add_argument("--workers", type=int, default=1, help="Independent concurrent Codex requests; batch sizes stay unchanged")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--translation_batch_size", type=int, default=10)
    parser.add_argument("--reasoning_effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for each Codex batch")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_results", type=int, default=2000)
    parser.add_argument("--limit", type=int, help="Process only the first N fetched/input papers (smoke test)")
    parser.add_argument("--input_json", type=Path, help="Reuse raw paper list or existing all_papers JSON, without arXiv access")
    parser.add_argument("--paper_info_dir", default=str(DEFAULT_PAPER_INFO_DIR), help="Shared raw paper snapshot directory")
    parser.add_argument("--cache_dir", type=Path, help="Defaults to SAVE_DIR/.codex_cache")
    parser.add_argument("--codex_bin", default="codex")
    args = parser.parse_args(argv)
    if args.save_dir is None:
        args.save_dir = str(RUNS_DIR / f"{args.model}_runs")
    for option in ("workers", "batch_size", "translation_batch_size", "timeout", "max_retries", "max_results", "limit"):
        value = getattr(args, option)
        if value is not None and value < 1:
            parser.error(f"--{option} must be positive")
    date = args.date
    if date is not None:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            parser.error("--date must be YYYY-MM-DD")
    papers, date, source_metadata = load_filter_input(args)
    conference = source_metadata.get('source') == 'openreview'
    args.threshold = (args.threshold if args.threshold is not None else 4) if conference else 5
    input_digest = paper_info_digest(papers)
    if reuse_completed_input(args.save_dir, date, papers, minimum_threshold=None if conference else 5):
        return
    cache_dir = args.cache_dir or Path(args.save_dir) / ".codex_cache"
    if papers:
        from daily_arxiv.reporting.artifacts import prior_paper_results, completed_date_threshold
        if conference:
            args.threshold = completed_date_threshold(args.save_dir, date, args.threshold)
        prior = prior_paper_results(args.save_dir, papers)
        client = None
        if needs_inference(papers, prior, args.threshold):
            client = CodexClient(args.codex_bin, args.reasoning_effort, args.timeout, args.max_retries,
                                 model=args.model)
            client.check_login()
        papers, translated = run_filter(papers, client, cache_dir, args.threshold,
                                        args.batch_size, args.translation_batch_size, args.workers, prior=prior)
    else:
        translated = []
    save_reports(papers, translated, date, args.threshold, args.save_dir,
                 input_digest=input_digest, source_metadata=source_metadata)
    print(f"Complete: {len(papers)} papers evaluated, {len(translated)} hits translated using {args.model}.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

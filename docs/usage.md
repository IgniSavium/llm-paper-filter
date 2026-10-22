# Advanced usage

Run commands from the repository root with the Python environment activated.

## Fetching and date queries

The default categories are `cs.AI`, `cs.LG`, `cs.CL`, and `stat.ML`. To add a
category to incremental discovery:

```bash
python daily_arxiv/run_daily_arxiv.py \
  --categories cs.AI cs.LG cs.CL stat.ML cs.CV --workers 1
```

Without a date argument, the runner synchronizes through today UTC, starting
with a seven-day overlap. A successful harvest updates
`daily_arxiv/paper_info/oai_sync_state.json`. Later runs recover outages from that
boundary and resume unfinished model batches. Every category and pagination
stream must finish before discovery advances.

```bash
python daily_arxiv/run_daily_arxiv.py --sync_from 2026-10-01 --workers 1
python daily_arxiv/run_daily_arxiv.py --through_date 2026-10-09 --workers 1
```

Explicit `--date` or `--from_date` queries always fetch fresh inputs. Use the
range form to process every requested date. Empty results retain JSON records
but produce no HTML; the website shows an unlinked archive entry. Paper dates refer to OAI metadata changes, not version submission times.
A delayed publication is retrieved when its public metadata becomes available.
Provider delays with old datestamps are covered only within the chosen overlap.
No fixed execution time guarantees that every provider update is already indexed.

```bash
python daily_arxiv/fetch_arxiv_papers.py \
  --from_date 2026-10-08 --through_date 2026-10-09
```

This fetch-only command saves inputs without invoking a model. All pages are
read; `--max_results` is a fail-closed bound rather than permission to truncate
results. Increase it if a requested date exceeds the default limit.

## Scoring and translation

Each paper receives four independent scores in `eval.scores`:

| Score | Interpretation |
| --- | --- |
| 1 | Unrelated |
| 2 | Incidental connection through tools, terminology, or application context |
| 3 | Substantive but secondary or limited contribution |
| 4 | A substantial main contribution advances this direction |
| 5 | The central question and main contribution directly address a core topic |

Evidence in one direction cannot raise another direction's score. The derived
`relevance_score` is the maximum, used for ordering and candidate admission.
The default admission threshold is 4; change it with `--threshold`.

GPT screening uses three papers per batch; translation uses ten. All GPT entry
points, including the daily scheduler, conference queue and translation repair,
default to `--workers 1`. Serial requests prioritize reliability during unattended
runs. You can explicitly try `--workers 2` after confirming stable access; return
to one worker if rate-limit errors recur. Changing workers affects concurrency,
not batch sizes, inference-cache identity or completed reports. Other useful
options include `--batch_size`, `--translation_batch_size`, `--timeout`,
`--max_retries`, and `--reasoning_effort`.

Both GPT and local-model translation send prose fragments to the model and reinserts recognized LaTeX
formulas in Python, preserving exact formula bytes and order in titles,
abstracts, and reasons. Invalid translation batches retain valid sibling results
and automatically split the remaining work down to bounded single-paper retries.
Network and model-access errors do not trigger batch splitting.

Use an existing input without another arXiv query:

```bash
python daily_arxiv/daily_arxiv_filter_codex.py \
  --input_json daily_arxiv/paper_info/papers_2026-10-09.json \
  --date 2026-10-09 --workers 1
```

Caches are identified by paper content, model, reasoning effort, prompt, and
schema. Completed dates are preserved even after changing the model, rubric,
reasoning effort, or threshold; these changes apply to unfinished/new work.
Unchanged title/abstract content also reuses saved decisions and translations
when bibliographic metadata changes. A changed title/abstract is new model work.
Late additions to an existing date retain that date's original admission threshold.
To deliberately run a new comparison, use a separate `--runs_dir` outside the
existing run root (or an independent `--save_dir` for a backend entry point).
Rerun an interrupted command to reuse validated completed work. `--dry_run` on the
shared runner prints the plan without fetching or inference. GPT model selection is explicit; the code
does not silently switch to another model or API billing.

## HTML filters and reading

Each direction has its own match mode: ignore, include, or NOT/exclude. Select
exactly 4, exactly 5, or either score. Included directions combine with AND or
OR; matching excluded conditions are always removed.

The default is `MechInterp=5 OR CompGen=5`. Custom expressions support `=`, `>=`,
`<=`, `>`, `<`, `!=`, AND, OR, NOT, and parentheses:

```text
(MechInterp>=4 OR CompGen=5) AND NOT Systems=5
```

A bare direction means a score of at least 4. Reset restores the default filter.
The loaded HTML pool contains only admitted candidates; all scored inputs are
stored separately, including lower scores.

Bookmarks, their reading order and read status are saved locally in the browser. **备份收藏**
downloads a versioned JSON file covering all reports' saved bookmarks and their
order and read status, including records outside the current report/filter/page. Keep this file
on disk. After clearing browser data, click **恢复收藏** and choose the file;
restoration merges with existing bookmarks, deduplicates records and preserves
reading order. Existing bookmarks keep their current order; new records are
appended in backup order. A paper stays read if either the current bookmark or
the restored record says it is read; older backups without read status restore
new bookmarks as unread. Invalid files leave existing data intact. Existing
browser bookmark keys are migrated automatically when opening an updated reader.
The Markdown **导出目录** remains a separate reading-list export, not a restorable
backup. **仅看已收藏** shows all bookmarks in the current report, ignoring direction
filters, including 4-point papers hidden by the normal default filter.
Click an outline title to open the arXiv abstract or the conference paper's
OpenReview page in a new tab. The checkbox immediately before the title marks
the paper as read; uncheck it to mark it unread. Each outline entry groups five
actions: jump to its report card (loading the correct page when needed), copy
metadata, move up, move down, and remove the bookmark. In the
bookmark view, the icon beside each card's star returns to its outline entry.

Conference readers exported for Pages filter the whole corpus before pagination,
then fetch just the required page chunks. **展开本页摘要** affects the current page;
abstract formulas render when opened. Shared resources are cached across reports.
The web export needs HTTP(S); use the original single-file HTML for offline reading.
Abstracts can show English, Chinese, or both; sentence highlighting is approximate
when translations split or merge sentences. Single-file reports embed formula
assets for offline reading.

## Generated files

Generated inputs, model outputs and caches stay outside the source branch:

```text
daily_arxiv/paper_info/
├── papers_YYYY-MM-DD.json         # Fresh explicit date snapshots
├── oai_sync_state.json           # Incremental discovery boundary
└── updates/                      # Immutable incremental input batches

daily_arxiv/runs/
├── .scheduler/
│   ├── state.json                # Schedule, completion boundary, and next run
│   ├── scheduler.log             # Rotating lifecycle log
│   └── job_*.log                 # Fetching and inference logs per attempt
└── gpt-6-luna_runs/
    ├── .reports/                  # Completion manifests and per-date locks
    ├── all_papers/                # Every evaluated paper
    ├── hits/                      # Admitted candidates
    ├── hits_zh/                   # Translated candidates
    ├── html_reports/              # One reading report per metadata date
    ├── .codex_cache/              # Reusable inference results
    └── updates/                   # Incremental model batch artifacts
```

Reports are rendered and validated in a staging directory before replacement.
The completion manifest is committed last and includes every artifact's checksum;
an interrupted commit cannot pass publication checks. Zero-hit/empty dates are
completed results without an HTML file. Legacy reports are checked against their
JSON data and rebuilt during website export without invoking a model.

If the same paper changes on several queried dates, only its latest available
version is retained. A paper subsequently updated outside a historical query
range may disappear from that range; old snapshots remain useful for reproducing
previous inputs, but provider queries do not reconstruct old announcement states.

## Scheduling

```bash
python daily_arxiv/scheduler.py start \
  --at 01:00 --timezone Asia/Shanghai --day_lag 2 --lookback_days 7
python daily_arxiv/scheduler.py status
python daily_arxiv/scheduler.py stop
```

`--day_lag 2` targets the day before yesterday in the selected local timezone,
while arXiv metadata dates themselves remain UTC dates. `--lookback_days` controls
how many recent metadata dates are requeried. Longer outages extend the query
back to the last completed date; the initial date bounds the retrospective window.

The first start schedules the next daily run and uses existing complete reports
to seed completion. Work is serialized with a process lock. Interrupted jobs
resume from the saved boundary. The tmux session continues after SSH disconnects,
but must be restarted after a machine or container reboot. The same user must
retain a working Codex login. Diagnose failures in the scheduler and job logs.
Failed filtering/fetching jobs retry after 1 hour, 1 hour, 2 hours, and then at
most every 4 hours until recovery; the next daily trigger takes priority if it
is sooner. Failures never advance the completed date. OAI errors include the
HTTP status or network exception type and the category/page position.
After exhausted transport retries, multi-category OAI queries fall back to
separate categories, then recursively smaller date windows. Every split must
complete before results are saved; malformed XML, missing records, and invalid
pagination remain errors. This reduces dependence on a single slow multi-day
page, but cannot resolve an unavailable single-day endpoint.

## Local models and legacy scripts

The shared runner supports all three local models:

```bash
python daily_arxiv/run_daily_arxiv.py --models Llama-3.3-70B-Instruct \
  --llama_model_path /path/to/Llama-3.3-70B-Instruct
python daily_arxiv/run_daily_arxiv.py --models Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --llama31_model_path /path/to/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
python daily_arxiv/run_daily_arxiv.py --models Qwen2.5-72B-Instruct \
  --qwen_model_path /path/to/Qwen2.5-72B-Instruct
```

The Llama 3.3 entry point is `daily_arxiv/daily_arxiv_filter.py`. The Llama 3.1
AWQ INT4 and Qwen 2.5 entry points are in `daily_arxiv/other_local_models/` and
forward to the same pipeline; the AWQ wrapper selects `awq_marlin`.
Adjust `--tp_size` and `--gpu_util` for the available GPUs. GPT runs do not
require GPU dependencies. Local runs use `.local_cache/`; GPT runs use
`.codex_cache/`. Shared validation and report saving keep their output formats aligned.
The local backend requires vLLM 0.19+ and uses the same JSON schema for
structured decoding as the GPT backend. Local requests are serialized for GPU safety, with automatic splitting
when a batch exceeds the configured context budget. Local CLI options include
`--max_model_len`, `--evaluation_tokens`, `--translation_tokens`, `--dtype`,
and `--quantization`.
`daily_arxiv/fetching/announcements.py` and `--initial_backfill` retain legacy RSS/submission-date
utilities. Use OAI date queries for the current metadata-date reports.

`daily_arxiv/daily_arxiv_filter.sh` is a portable compatibility wrapper that
forwards its arguments to `run_daily_arxiv.py`. Configure proxies through the
standard `HTTP_PROXY` / `HTTPS_PROXY` environment variables when needed.

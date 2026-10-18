# OpenReview conference papers

Fetch public **accepted main-track** papers from ICML, NeurIPS and ICLR, then
reuse the daily arXiv scoring prompt, per-direction threshold, translation,
inference caches and bilingual HTML reader. No extra dependency is needed.

Also supports **public active submissions** with `--paper_status submitted`.
The default remains `--paper_status accepted`.

## Screen submissions before acceptance

```bash
# Fetch ICLR 2027 public active main-track submissions, without LLM inference.
python openreview/fetch_openreview_papers.py --paper_status submitted --conferences ICLR --years 2027

# Fetch and score/translate with the existing pipeline.
python openreview/run_openreview.py --paper_status submitted --conferences ICLR --years 2027 --workers 4

# Refresh the public list; unchanged inputs reuse inference caches.
python openreview/run_openreview.py --paper_status submitted --conferences ICLR --years 2027 --refresh --workers 4

# Process saved submission snapshots with queue retry/resume.
python openreview/process_snapshots.py --paper_status submitted --workers 4

# Preview commands without loading credentials, fetching or inference.
python openreview/run_openreview.py --paper_status submitted --conferences ICLR --years 2027 --dry_run
```

Submission snapshots live in `openreview/paper_info/ICLR_2027_submitted/`;
results in `openreview/runs/gpt-6-luna_runs/ICLR_2027_submitted/` and queue state
in `openreview/runs/.queue/gpt-6-luna/submitted/`. Accepted inputs, results and
queue state retain their existing paths. Both workers select only their own mode.
Without `--years`, submission mode checks the current and next calendar years;
specifying the conference and year explicitly is recommended.

“Submitted” means publicly visible, currently active main-track submissions,
including anonymous papers. It excludes accepted/rejected decisions, withdrawals,
desk rejections and private notes. API v2 uses the conference's `/-/Submission`
invitation; historical API v1 uses `/-/Blind_Submission`. Both the venue ID and an
explicit active-submission label are checked. Anonymous authors are retained as
provided by OpenReview; restricted author fields are redacted, even if login can
read them. Missing or restricted public titles/abstracts stop the fetch before
saving. No reviews or author identities are inferred.

Visibility depends on each venue/year:

| Venue | Public submissions before decisions |
| --- | --- |
| ICLR 2027 | The [venue](https://openreview.net/group?id=ICLR.cc/2027/Conference) enables public submissions; [author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) describe open reviewing and anonymous submissions. |
| ICML 2026 | [Author instructions](https://icml.cc/Conferences/2026/AuthorInstructions) explicitly keep submissions private during review and publish only accepted papers. |
| NeurIPS 2026 | The [main-track handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook) describes publication after decisions, including optional publication for rejected papers; an early complete public corpus cannot be assumed. |

This mode records `coverage=public_active_submissions_only`: successful pagination
means all records in the selected visible API query were retrieved, not that every
submission to the conference is public. A successful empty query is skipped without
an empty snapshot; 401/403 remains an access error. Login can resolve some API
challenges but does not make private submissions eligible. No accepted-paper
fallback is substituted for unavailable submissions.

Snapshots record their fetch time. Use `--refresh` to update statuses and remove
papers that have since withdrawn or received decisions. A failed refresh preserves
the previous snapshot; without refresh the saved snapshot remains reusable offline.
The HTML reader labels these as “在审投稿” and report metadata counts them as
`submitted_papers`, rather than accepted papers. Scoring still uses title/abstract,
with the same research-interest prompt as the accepted-paper mode.

## Local account credentials

The scripts automatically read `openreview/credentials.json` (excluded from Git).
Store the same login email and password used on the OpenReview website:

```json
{
  "username": "your-login-email@example.com",
  "password": "your-openreview-password"
}
```

Create it from `credentials.example.json` if missing, restrict its permissions
with `chmod 600 openreview/credentials.json`, then edit it locally. Do not send
credentials in chat. Use `--credentials_file /path/to/file.json` for a different
location. The default file is loaded only if present; `--dry_run` does not read
credentials or log in. An incomplete file fails with instructions to fill it in.

Login follows the [official client's `/login` and Bearer-token flow](https://github.com/openreview/openreview-py/blob/master/openreview/api/client.py).
Credentials are sent only to official API hosts. Passwords, login responses and
tokens are never printed or stored in snapshots/reports. Login is lazy, so reading
existing paper snapshots does not require network access. If the account requires
MFA, obtain a session token using the official authenticated client and store
`{"token": "YOUR_SESSION_TOKEN"}` instead; this script does not implement MFA.
Authentication does not guarantee that the server's challenge will disappear.

## Fetch and filter

To process only existing snapshots in newest-first conference order, with automatic
retry and cache-based resume:

```bash
python openreview/process_snapshots.py --workers 4
python openreview/process_snapshots.py --dry_run
```

The ordering is `ICLR n+1 → NeurIPS n → ICML n`, descending over years. Thus an
existing ICML 2026 snapshot precedes ICLR 2026, followed by NeurIPS 2025 and ICML
2025. This worker never fetches or reads OpenReview credentials. It writes queue
state and per-conference logs under `openreview/runs/.queue/gpt-6-luna/`, retries
failed jobs after 5 minutes (backoff up to 30 minutes), and skips completed jobs
on restart when their inputs, scoring prompt and relevant model options match.

```bash
# Run from the repository root. Default: all three venues, 2023–2026 in 2026.
python openreview/fetch_openreview_papers.py
python openreview/run_openreview.py --workers 4

# Choose venues/years; older years are allowed if hosted by OpenReview.
python openreview/run_openreview.py --conferences ICML ICLR --years 2024 2025 2026 --workers 4

# Small inference run (still fetches the complete accepted-paper corpus).
python openreview/run_openreview.py --conferences ICLR --years 2026 --limit 10

# Refresh provider metadata; unchanged inference inputs reuse caches.
python openreview/run_openreview.py --years 2026 --refresh --workers 4

# Inspect commands without network or inference.
python openreview/run_openreview.py --dry_run
```

Defaults include the three previous calendar years **and the current year**.
The current year's accepted list may be incomplete; use `--refresh` as new
papers become public. An empty current-year venue is skipped without saving
an empty snapshot; empty older venues are errors. Some older conferences are
not hosted on OpenReview, and no proceedings-site fallback is implemented.

Raw inputs are saved under
`openreview/paper_info/ICLR_2026/papers_2026-01-01.json` and model outputs under
`openreview/runs/gpt-6-luna_runs/ICLR_2026/`. Each venue/year has separate
`all_papers/`, `hits/`, `hits_zh/`, `html_reports/` and inference caches.
The `YYYY-01-01` filename/`target_date` is a compatibility label for the shared
pipeline, **not a publication date**; metadata records `date_basis=conference_year`
and the reader displays the actual conference/year. Inputs and results stay local.

Acceptance requires an explicit accepted `content.venue` label in addition to
[OpenReview's exact `content.venueid` query](https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc):
`ICLR.cc/YEAR/Conference`, `ICML.cc/YEAR/Conference`, or
`NeurIPS.cc/YEAR/Conference`. “Published” here means publicly accessible notes
marked accepted by the main conference, rather than proof of final proceedings
publication. Poster, Oral, Spotlight, Regular and recognized historical accepted labels are
allowed. Submitted, rejected, withdrawn, missing or unknown venue labels are
excluded even if they share the conference venue ID, as can happen for ICLR 2023. API v2 content wrappers and API v1 flat
fields are normalized to the same `title`, `abstract`, `url`, `authors` format.
API v1 is tried for older venues; incomplete pagination, duplicate notes,
missing abstracts and changing totals stop the fetch without replacing a valid
snapshot. Saved snapshots are checked by SHA-256 and reused unless refreshed.

`--fetch_timeout`, `--fetch_retries` and `--page_size` control retrieval;
429/5xx/network failures retry. A 401/403 (including OpenReview's
`ChallengeRequiredError`) stops with an access error instead of treating the
venue as empty. Retry after restoring API access; existing snapshots remain
usable without a network request. LLM options match daily arXiv, including
`--models`, `--threshold`, `--workers`, `--reasoning_effort` and `--codex_bin`.
Local vLLM backends can also consume these snapshots using their existing
`--input_json`, `--date YEAR-01-01` and separate `--save_dir` arguments.

Offline verification:

Translation keeps LaTeX formulas locally, including command-led formulas with
inner whitespace or a following number (such as `$ \mathbf{U} $` and `$\sim$11M`).
Singleton requests require nonempty output slots when all source slots contain
prose; omission errors identify the exact source fragment for corrective retries. Valid results are cached immediately. If one paper still fails
translation content validation after singleton retries, other batches continue;
after the pass the queue retries the remaining uncached papers. Failed papers
are never silently omitted or accepted with invalid formulas.

```bash
python -m unittest discover -s openreview/tests -v
```

## Audit and repair saved translations

Scan existing results without fetching papers or invoking the LLM:

```bash
python openreview/repair_translations.py --dry_run
```

Repair invalid translations, reuse evaluations, and rebuild bilingual HTML:

```bash
python openreview/repair_translations.py --workers 4 --repair_dir openreview/runs/.translation-repairs/my-repair
# Resume the same repair after an interruption using the same --repair_dir.
```

The repair directory keeps original JSON/HTML, completion markers and replaced
cache entries under `backup/`, new inference caches under `cache/`, and a
cumulative `targets.json`. `--include_audit path/to/translation_failures.json`
also reprocesses corpus/ID pairs from an earlier audit, including semantic errors
which cannot be detected mechanically. The model and reasoning effort are retained
from the original completion markers. Evaluation scores are never recomputed.

Translation, validation and HTML rendering share formula boundaries, including
space-padded formulas, symbols touching digits and adjacent inline formulas.
Validation preserves formula order and rejects substantial verbatim English prose.
These checks do not prove semantic equivalence; ambiguous formula referents and
short untranslated phrases still require review. All translated rows are validated
before report replacement. New reports commit a checksum manifest only after all
JSON and HTML artifacts are ready. Interrupted commits or corrupted artifacts
fail completion checks. A completed corpus is preserved when model defaults,
prompts or reasoning effort change; translation-rule version changes alone do
not schedule historical inference. Use the explicit repair command for repairs.
Zero-hit corpora are complete without an HTML reader.

## Direction statistics

Generate a self-contained HTML analysis from existing full evaluations:

```bash
pip install -r requirements-statistics.txt
python openreview/analyze_direction_counts.py
# Open openreview/runs/direction_statistics/index.html
```

The report counts `MechInterp == 5`, `CompGen == 5`, their union and intersection,
using every scored paper in each conference/year/status as the denominator.
Accepted main-track corpora and ICLR 2027 public active submissions are labeled
separately. Counts, shares and within-conference accepted year-over-year changes
are included; submissions do not form an accepted-paper time-series continuation.
The HTML embeds its charts and data for offline use and follows the daily reader's
light/dark themes. It shows two vertically aligned charts: direction-specific counts/shares above
and total accepted/submitted papers below. The submission toggle controls both. The lower chart also switches to official
acceptance rates, using the sourced records in `acceptance_statistics.json` rather
than dividing local accepted snapshots by unrelated submission totals. Missing
verified rates and undecided ICLR 2027 submissions remain blank. The rate view
provides source links and labels each point with its percentage and official
accepted/submitted counts. Missing submitted counts are estimated as accepted
count divided by the official rate and marked with ≈. Official accepted counts
are preferred; the local accepted snapshot count is used when absent. Original
reported values remain separate from display estimates in the exported JSON. Original official fact sheets are archived locally under
`runs/.statistics-sources/2026-10-16/`. Separate data files and SVG figures accompany it. Generation
requires Matplotlib but performs no network access or LLM inference.

## GitHub Pages

The shared Pages exporter appends conference statistics and a year-by-year report
archive after the existing daily arXiv content. It copies the generated direction
statistics page and its downloadable assets from `runs/direction_statistics/`,
then builds conference readers from `runs/gpt-6-luna_runs/*/hits_zh/` using the
shared bilingual template. Only papers scoring 4 or 5 in **MechInterp or CompGen**
are exported; conference controls, badges and ordering use those two directions.
The default reader filter remains `MechInterp=5 OR CompGen=5`. Statistics retain
their original score-5 definition. Accepted and submitted corpora are labeled
separately, and source inference/translation artifacts are not modified.

Web readers keep a compact corpus-wide filter index and fetch card chunks for
50 papers per page. Filters and the bookmark view apply to the entire conference,
not just the visible page. Formula rendering waits until an abstract is opened;
KaTeX and reader assets are shared across conference reports. Local inference
outputs remain single-file offline readers. Both formats offer **备份收藏** and
**恢复收藏**: download all saved bookmarks/order as JSON, then merge them back after
clearing browser data or moving to another browser. The existing Markdown export
is for reading lists; it is not a restorable backup.

From the repository root:

```bash
# Preview locally without publishing (no network or LLM inference).
python -m daily_arxiv.reporting.site --update-openreview --output /tmp/arxiv-site
python -m http.server 8000 --directory /tmp/arxiv-site

# Publish the complete daily + conference site to gh-pages.
python -m daily_arxiv.reporting.site --update-openreview
```

Regenerate direction statistics first after updating conference evaluations.
Conference fetching, filtering, statistics generation and Pages refresh are manual.
The daily scheduler never runs these steps. Without `--update-openreview`, the
exporter preserves the already published conference section and files unchanged;
a fresh destination contains only the daily archive. Use `--update-openreview`
when adding conference data or changing its reader/layout.

Source code and tests are versioned. Paper snapshots, credentials, model results,
inference caches, queue state, repair backups and generated reports remain local
under the ignored paths in `.gitignore`. Keep `paper_info/`, `runs/*_runs/`,
repair backups and official source documents when cleaning temporary files.
Completed queue input copies, obsolete chart exports and Python bytecode may be
removed; the queue recreates frozen inputs when a job needs to run again.

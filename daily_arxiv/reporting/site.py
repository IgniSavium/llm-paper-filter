"""Export completed daily reports and publish an isolated GitHub Pages branch."""
import argparse
from datetime import date, datetime, timedelta
import fcntl
from html import escape
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from zoneinfo import ZoneInfo

from daily_arxiv.paths import PROJECT_DIR, RUNS_DIR
from daily_arxiv.filtering.prompts import CATEGORIES, REVIEW_INSTRUCTIONS
from daily_arxiv.reporting.artifacts import load_completed_report
from daily_arxiv.reporting.html import generate_mobile_html_report

FIRST_DATE = '2026-10-08'
DAILY_PUBLICATION_THRESHOLD = 5


def release_cutoff(now=None, timezone='Asia/Shanghai', day_lag=2):
    now = now or datetime.now(ZoneInfo(timezone))
    return (now.astimezone(ZoneInfo(timezone)).date() - timedelta(days=day_lag)).isoformat()


def export_site(model_dir, destination, first_date=FIRST_DATE, through_date=None, allow_pending=False,
                update_openreview=False):
    through_date = min(through_date or release_cutoff(), release_cutoff())
    date.fromisoformat(first_date)
    date.fromisoformat(through_date)
    destination = Path(destination)
    reports = destination / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    entries = []
    for translated_path in sorted((Path(model_dir) / 'hits_zh').glob('hits_zh_*.json'), reverse=True):
        match = re.fullmatch(r'hits_zh_(\d{4}-\d{2}-\d{2})\.json', translated_path.name)
        if not match or not first_date <= match[1] <= through_date:
            continue
        day = match[1]
        # Require final translated output as well as HTML; only run this exporter
        # after the runner has finished (the scheduler supplies its completed date).
        complete = load_completed_report(model_dir, day)
        payload = complete['hits_zh']
        published = [paper for paper in payload['hits_zh']
                     if any(paper.get('eval', {}).get('scores', {}).get(category) == DAILY_PUBLICATION_THRESHOLD
                            for category in CATEGORIES)]
        if not published:
            all_papers = complete['all_papers']
            hits = complete['hits']
            if (all_papers['metadata']['target_date'] != day or hits['metadata']['target_date'] != day
                    or (not payload['hits_zh'] and hits['hits'])):
                raise ValueError(f'Incomplete empty report: {day}')
            status = 'no_updates' if not all_papers['all_papers_evaluated'] else 'no_matches'
            entries.append({'date': day, 'url': None, 'papers': 0, 'status': status,
                            'score_5_counts': {category: 0 for category in CATEGORIES}})
            continue
        # Render from the validated generation. This also upgrades legacy readers
        # locally, without rescoring or trusting their old, unversioned HTML.
        metadata = dict(payload['metadata'], threshold_used=DAILY_PUBLICATION_THRESHOLD,
                        total_hits=len(published),
                        new_papers=sum(paper.get('status') == 'v1 (New)' for paper in published))
        metadata['version_updates'] = len(published) - metadata['new_papers']
        source = Path(generate_mobile_html_report(published, day, destination, metadata=metadata))
        content = source.read_text(encoding='utf-8')
        nav = '<nav style="padding:12px 24px"><a href="../index.html">← 日期归档</a></nav>'
        content = re.sub(r'(<body\b[^>]*>)', lambda m: m[1] + nav, content, count=1, flags=re.I)
        (reports / source.name).write_text(content, encoding='utf-8')
        source.unlink()
        counts = {category: sum(p.get('eval', {}).get('scores', {}).get(category) == 5
                                for p in published) for category in CATEGORIES}
        entries.append({'date': day, 'url': 'reports/' + source.name,
                        'papers': len(published), 'score_5_counts': counts})
    if (not entries or entries[-1]['date'] != first_date) and not (allow_pending and not entries):
        raise ValueError(f'Starting report {first_date} must exist before publication')
    keep = {Path(row['url']).name for row in entries if row['url']}
    for stale in reports.glob('report_*.html'):
        if stale.name not in keep:
            stale.unlink()
    def score_counts(row):
        return '<div class="score-counts">' + ''.join(
            f'<span class="score-count">{category} · 5 分<b>{row["score_5_counts"][category]} 篇</b></span>'
            for category in CATEGORIES) + '</div>'

    months = {}
    for row in entries:
        months.setdefault(row['date'][:7], []).append(row)
    groups = []
    def archive_row(row):
        if not row['url']:
            label = '未找到有效更新' if row['status'] == 'no_updates' else '无符合筛选条件的论文'
            return (f'<li class="archive-empty"><time datetime="{row["date"]}">{row["date"]}</time>'
                    f'<span>{label}</span></li>')
        return (f'<li><div class="archive-heading"><a href="{row["url"]}"><time datetime="{row["date"]}">{row["date"]}</time> ↗</a>'
                f'<span class="paper-count">{row["papers"]} 篇候选论文</span></div>{score_counts(row)}</li>')
    for index, (month, days) in enumerate(months.items()):
        year, number = month.split('-')
        rows = '\n'.join(archive_row(row) for row in days)
        expanded = ' open' if index == 0 else ''
        groups.append(f'<details class="archive-month" data-month="{month}"{expanded}>'
                      f'<summary><span class="month-chevron" aria-hidden="true">›</span>'
                      f'<span>{year} 年 {int(number)} 月</span><span class="month-meta">{sum(bool(row["url"]) for row in days)} 份日报</span></summary>'
                      f'<ul class="archive">{rows}</ul></details>')
    latest = next((row for row in entries if row['url']), None)
    latest_card = (f'<a class="latest" href="{latest["url"]}"><strong>阅读最新日报 · {latest["date"]} ↗</strong>'
                   f'<span class="description">{latest["papers"]} 篇候选论文 · 可在日报内继续筛选</span>{score_counts(latest)}</a>'
                   if latest else ('<div class="latest"><strong>暂无有效日报</strong><span class="description">已检查的日期见下方归档。</span></div>' if entries
                                   else f'<div class="latest"><strong>首份日报 · {first_date}</strong><span class="description">正在生成评分和双语摘要，完成后自动上线。</span></div>'))
    # Use the actual prompt sections, so the displayed rubric follows code changes.
    interests, common = REVIEW_INSTRUCTIONS.split('READING PREFERENCE', 1)
    intro, directions = interests.split('RESEARCH INTERESTS', 1)
    sections = re.split(r'(?m)^\d+\. ', directions)[1:]
    if len(sections) != len(CATEGORIES):
        raise ValueError('Expected one prompt section per research category')
    prompt_cards = []
    for category, section in zip(CATEGORIES, sections):
        heading, text = section.strip().split('\n', 1)
        subtitle = heading.split(' — ', 1)[1]
        prompt_cards.append(f'<details><summary>{escape(category)}<small>{escape(subtitle)}</small></summary>'
                            f'<pre class="prompt" lang="en">{escape(text.strip())}</pre></details>')
    page = (Path(__file__).parent / 'templates/index.html').read_text(encoding='utf-8')
    replacements = {'LATEST_CARD': latest_card, 'ARCHIVE_GROUPS': ''.join(groups),
                    'CATEGORY_PROMPTS': ''.join(prompt_cards),
                    'COMMON_PROMPT': escape(intro.strip() + '\n\nREADING PREFERENCE' + common)}
    for key, value in replacements.items():
        page = page.replace('{{' + key + '}}', value)
    if update_openreview:
        from openreview.site import export_openreview
        conference_section = export_openreview(PROJECT_DIR / 'openreview', destination)
    else:
        # Conference snapshots are updated manually. Daily publication preserves
        # the existing section and assets, without loading or rendering corpora.
        previous = destination / 'index.html'
        existing = previous.read_text(encoding='utf-8') if previous.exists() else ''
        match = re.search(r'<section id="openreview"[ >].*?</section>', existing, re.S)
        conference_section = match[0] if match else ''
    if conference_section:
        page = page.replace('</main>', conference_section + '\n</main>', 1)
    (destination / 'index.html').write_text(page, encoding='utf-8')
    url = escape(latest['url'] if latest else 'index.html', quote=True)
    (destination / 'latest.html').write_text(f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={url}"><title>最新日报</title></head><body><a href="{url}">阅读最新日报</a></body></html>', encoding='utf-8')
    (destination / '.nojekyll').touch()
    temporary_reports = destination / 'html_reports'
    if temporary_reports.is_dir() and not any(temporary_reports.iterdir()):
        temporary_reports.rmdir()
    (destination / 'archive.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return entries


def publish_site(through_date, model_dir=None, allow_pending=False, update_openreview=False):
    model_dir = Path(model_dir or RUNS_DIR / 'gpt-6-luna_runs')
    root = RUNS_DIR / '.site'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=PROJECT_DIR, text=True).strip()
        # Fresh shallow clone avoids stale checkout state after failed pushes.
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            checkout = Path(temporary) / 'pages'
            def git(*args, cwd=checkout, check=True):
                return subprocess.run(['git', *args], cwd=cwd, check=check, text=True,
                                      capture_output=True, timeout=120,
                                      env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
            exists = git('ls-remote', '--heads', remote, 'gh-pages', cwd=PROJECT_DIR).stdout.strip()
            if exists:
                git('clone', '--depth', '1', '--branch', 'gh-pages', remote, str(checkout), cwd=PROJECT_DIR)
            else:
                checkout.mkdir()
                git('init', '-b', 'gh-pages')
                git('remote', 'add', 'origin', remote)
            git('config', 'user.name', 'arXiv Daily Publisher')
            git('config', 'user.email', 'IgniSavium@users.noreply.github.com')
            entries = export_site(model_dir, checkout, through_date=through_date, allow_pending=allow_pending,
                                  update_openreview=update_openreview)
            latest_date = next((row['date'] for row in entries if row['url']), None)
            # The isolated checkout contains only website artifacts; this also
            # handles a site with no linked reports and stages removed pages.
            git('add', '-A')
            if git('diff', '--cached', '--quiet', check=False).returncode == 0:
                return {'status': 'unchanged', 'latest_date': latest_date}
            git('commit', '-m', f'Publish daily arXiv through {latest_date or "pending first report"}')
            git('push', 'origin', 'gh-pages')
            return {'status': 'pushed', 'latest_date': latest_date}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through-date', default=release_cutoff())
    parser.add_argument('--output', type=Path, help='Export locally without pushing')
    parser.add_argument('--allow-pending', action='store_true', help='Bootstrap an empty site while the first report is being generated')
    parser.add_argument('--update-openreview', action='store_true',
                        help='Manually rebuild conference readers/statistics assets; daily exports preserve the published version')
    args = parser.parse_args()
    if args.output:
        result = export_site(RUNS_DIR / 'gpt-6-luna_runs', args.output, through_date=args.through_date, allow_pending=args.allow_pending, update_openreview=args.update_openreview)
    else:
        result = publish_site(args.through_date, allow_pending=args.allow_pending, update_openreview=args.update_openreview)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

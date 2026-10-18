"""Add conference statistics and two-direction readers to the daily Pages export."""
from html import escape
import json
from pathlib import Path
import re
import shutil

from daily_arxiv.reporting.html import generate_mobile_html_report
from daily_arxiv.reporting.artifacts import load_completed_report

CATEGORIES = ('MechInterp', 'CompGen')
CONFERENCES = ('NeurIPS', 'ICML', 'ICLR')


def export_openreview(source, destination):
    source, destination = Path(source), Path(destination)
    runs = source / 'runs'
    statistics = runs / 'direction_statistics'
    corpora = sorted((runs / 'gpt-6-luna_runs').glob('*/hits_zh/hits_zh_*.json'))
    if not corpora and not (statistics / 'index.html').exists():
        return ''
    output = destination / 'openreview'
    output.mkdir(parents=True, exist_ok=True)
    rows, keep = [], set()
    for translated in corpora:
        complete = load_completed_report(translated.parent.parent, translated.stem.removeprefix('hits_zh_'))
        payload = complete['hits_zh']
        metadata = payload['metadata']
        conference, year = metadata['conference'], metadata['conference_year']
        status = metadata.get('paper_status', 'accepted')
        if (metadata.get('source') != 'openreview' or conference not in CONFERENCES
                or type(year) is not int or status not in ('accepted', 'submitted')):
            raise ValueError(f'Invalid conference report metadata: {translated}')
        day = metadata['target_date']
        if day != f'{year}-01-01' or translated.name != f'hits_zh_{day}.json':
            raise ValueError(f'Mismatched conference report date: {translated}')
        # Inference may be in progress: only export finalized readers.
        papers = [p for p in payload['hits_zh']
                  if any(p['eval']['scores'].get(c, 0) in (4, 5) for c in CATEGORIES)]
        name = f'{conference}_{year}' + ('_submitted' if status == 'submitted' else '')
        if name in keep:
            raise ValueError(f'Duplicate conference report: {name}')
        keep.add(name)
        # Reuse the bilingual renderer without changing local inference artifacts.
        report = Path(generate_mobile_html_report(papers, day, output / name, metadata,
                                                  page_size=50, asset_dir=output / 'assets'))
        content = report.read_text(encoding='utf-8')
        nav = '<nav style="padding:12px 24px"><a href="../../../index.html#conference-reports">← 顶会论文归档</a></nav>'
        content = re.sub(r'(<body\b[^>]*>)', lambda m: m[1] + nav, content, count=1, flags=re.I)
        report.write_text(content, encoding='utf-8')
        rows.append(dict(conference=conference, year=year, status=status, papers=len(papers),
                         url=report.relative_to(destination).as_posix(),
                         counts={c: {str(score): sum(p['eval']['scores'][c] == score for p in papers)
                                     for score in (4, 5)} for c in CATEGORIES}))
    for stale in output.iterdir():
        if stale.is_dir() and re.fullmatch(r'(ICLR|ICML|NeurIPS)_\d{4}(_submitted)?', stale.name) and stale.name not in keep:
            shutil.rmtree(stale)
    rows.sort(key=lambda r: (-r['year'], CONFERENCES.index(r['conference']), r['status']))
    (output / 'archive.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    groups = []
    for year in sorted({r['year'] for r in rows}, reverse=True):
        items = []
        for row in (r for r in rows if r['year'] == year):
            label = '在审投稿 · 尚未录用' if row['status'] == 'submitted' else '已录用'
            counts = ''.join(f'<span class="score-count">{c} · {score} 分<b>{row["counts"][c][str(score)]} 篇</b></span>'
                             for c in CATEGORIES for score in (4, 5))
            items.append(f'<li><div class="archive-heading"><a href="{escape(row["url"], quote=True)}">'
                         f'{row["conference"]} {year} · {label} ↗</a>'
                         f'<span class="paper-count">{row["papers"]} 篇候选论文</span></div>'
                         f'<div class="score-counts">{counts}</div></li>')
        groups.append(f'<details class="archive-month" open><summary>{year} 年 · 顶会论文</summary>'
                      f'<ul class="archive">{"".join(items)}</ul></details>')
    analysis = ''
    if (statistics / 'index.html').is_file():
        stats_output = output / 'direction_statistics'
        stats_output.mkdir(exist_ok=True)
        for asset in statistics.iterdir():
            if asset.is_file() and asset.suffix in ('.html', '.json', '.csv', '.svg'):
                shutil.copyfile(asset, stats_output / asset.name)
        embedded = (statistics / 'index.html').read_text(encoding='utf-8')
        embedded_style = """<style>
html,body{overflow:hidden}
.shell{max-width:none;margin:0;padding:0}
.masthead{display:none}
.report{margin-top:0;border-radius:10px;box-shadow:none}
.chart-wrap{overflow:visible}
.chart{min-width:0}
@media(max-width:760px){.shell{padding:0}.chart{min-width:0}.chart-wrap{overflow:visible}}
</style>"""
        embedded = embedded.replace('</head>', embedded_style + '</head>', 1)
        (stats_output / 'embedded.html').write_text(embedded, encoding='utf-8')
        analysis = '''<iframe id="conference-statistics" title="AI 顶会论文方向统计分析" src="openreview/direction_statistics/embedded.html" scrolling="no" style="display:block;width:100%;height:720px;border:0;overflow:hidden" loading="lazy"></iframe>
<script>
(function () {
 const frame = document.getElementById('conference-statistics');
 frame.addEventListener('load', function () {
  const doc = frame.contentDocument;
  if (!doc) return;
  function theme() {
   doc.documentElement.dataset.theme = document.documentElement.dataset.theme;
   frame.contentWindow.updateTheme();
  }
  theme();
  new MutationObserver(theme).observe(document.documentElement, {attributes:true, attributeFilter:['data-theme']});
  const content = doc.querySelector('main') || doc.body;
  function resize() { frame.style.height = Math.ceil(content.getBoundingClientRect().height) + 2 + 'px'; }
  new ResizeObserver(resize).observe(content);
  resize();
 });
})();
</script>'''
    return ('<section id="openreview" aria-labelledby="conference-title" style="margin-top:48px;padding-top:16px;border-top:1px solid var(--line)">'
            '<h2 id="conference-title">AI 顶会 · 研究方向统计</h2>' + analysis
            + '<h2 id="conference-reports">顶会论文归档</h2>'
            + ''.join(groups) + '</section>')

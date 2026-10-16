"""Build an offline, self-contained conference direction statistics report."""
import argparse
import csv
from datetime import datetime
from html import escape
import io
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from daily_arxiv.fetching.snapshots import paper_info_digest
from daily_arxiv.filtering.prompts import PROMPT_SHA256, validate_evaluation

METRICS = {'mech': 'MechInterp = 5', 'comp': 'CompGen = 5',
           'either': 'MechInterp = 5 OR CompGen = 5', 'both': 'MechInterp = 5 AND CompGen = 5'}
COLORS = {'ICLR': '#b17a46', 'ICML': '#527eac', 'NeurIPS': '#226b54'}


def collect(runs_dir):
    rows, selected = [], []
    for path in sorted(Path(runs_dir).glob('*/all_papers/all_papers_*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        m, papers = payload['metadata'], payload['all_papers_evaluated']
        conference, year, status = m['conference'], m['conference_year'], m.get('paper_status', 'accepted')
        if conference not in COLORS or status not in ('accepted', 'submitted'):
            raise ValueError(f'Unknown conference or status: {path}')
        if status == 'submitted' and (conference, year) != ('ICLR', 2027):
            continue
        if not papers or len({p['url'] for p in papers}) != len(papers):
            raise ValueError(f'Empty or duplicate paper corpus: {path}')
        if m['paper_info_sha256'] != paper_info_digest(papers):
            raise ValueError(f'Input checksum mismatch: {path}')
        if m['filter_prompt_sha256'] != PROMPT_SHA256:
            raise ValueError(f'Scoring prompt differs from current prompt: {path}')
        if m['total_papers_fetched'] != len(papers):
            raise ValueError(f'Paper count mismatch: {path}')
        counts = {k: 0 for k in METRICS}
        for paper in papers:
            validate_evaluation(paper['eval'])
            scores = paper['eval']['scores']
            mech, comp = scores['MechInterp'] == 5, scores['CompGen'] == 5
            flags = {'mech': mech, 'comp': comp, 'either': mech or comp, 'both': mech and comp}
            for key, matched in flags.items():
                counts[key] += int(matched)
                if matched:
                    selected.append(dict(conference=conference, year=year, status=status, metric=key,
                                         title=paper['title'], url=paper['url'],
                                         mech_score=scores['MechInterp'], comp_score=scores['CompGen']))
        assert counts['either'] == counts['mech'] + counts['comp'] - counts['both']
        rows.append(dict(conference=conference, year=year, status=status, total=len(papers),
                         counts=counts, rates={key: value / len(papers) * 100 for key, value in counts.items()},
                         fetched_at=m.get('fetched_at'), source=str(path.resolve()),
                         paper_info_sha256=m['paper_info_sha256'], filter_prompt_sha256=m['filter_prompt_sha256']))
    identities = [(r['conference'], r['year'], r['status']) for r in rows]
    if len(set(identities)) != len(identities) or not rows:
        raise ValueError('Duplicate corpus identities or no scored corpora')
    rows.sort(key=lambda r: (r['year'], list(COLORS).index(r['conference']), r['status']))
    previous = {}
    for row in rows:
        prior = previous.get(row['conference']) if row['status'] == 'accepted' else None
        # Only consecutive years of the same status form a year-over-year comparison.
        row['change'] = None
        if prior and row['year'] == prior['year'] + 1:
            row['change'] = {k: dict(count=row['counts'][k] - prior['counts'][k],
                                    percentage_points=row['rates'][k] - prior['rates'][k]) for k in METRICS}
        if row['status'] == 'accepted':
            previous[row['conference']] = row
    return rows, selected


def place_value_labels(ax, labels):
    """Place text in display coordinates, clear of every curve, point and label."""
    from matplotlib.transforms import Bbox
    from matplotlib.path import Path as PlotPath
    from matplotlib.text import Text
    labels = sorted(labels, key=lambda label: (label[0], -label[1]))
    fig = ax.figure
    fig.canvas.draw()
    # Freeze the layout before measuring display coordinates. Export must keep
    # the same axes bounds used by the collision checks.
    fig.set_layout_engine('none')
    renderer = fig.canvas.get_renderer()
    curves = [line.get_path().transformed(line.get_transform()) for line in ax.lines]
    points = [ax.transData.transform((x, y)) for x, y, _, _ in labels]
    point_boxes = [Bbox.from_bounds(x - 7, y - 7, 14, 14) for x, y in points]
    occupied, placements = [], []
    scale = fig.dpi / 72
    # Prefer nearby positions; expand the search only for dense clusters.
    offsets = sorted(((dx, dy) for dy in (18, -18, 34, -34, 50, -50, 66, -66, 82, -82, 100, -100)
                      for dx in (0, -22, 22, -44, 44, -66, 66, -88, 88, -110, 110)),
                     key=lambda p: (p[0] ** 2 + p[1] ** 2, p[1] < 0, abs(p[0])))
    for (x, y, text, color), anchor in zip(labels, points):
        # Measure the full multiline label with the same font and line spacing
        # used for annotation, rather than treating a newline as a glyph.
        probe = Text(0, 0, text, fontsize=10, linespacing=1.35, ha='center', va='center')
        probe.set_figure(fig)
        measured = probe.get_window_extent(renderer)
        width, height = measured.width, measured.height
        found = None
        for dx, dy in offsets:
            cx, cy = anchor[0] + dx * scale, anchor[1] + dy * scale
            box = Bbox.from_bounds(cx - width / 2 - 5, cy - height / 2 - 6,
                                   width + 10, height + 12)
            if not ax.bbox.contains(box.x0, box.y0) or not ax.bbox.contains(box.x1, box.y1):
                continue
            if any(box.overlaps(other) for other in occupied + point_boxes):
                continue
            if any(curve.intersects_bbox(box, filled=False) for curve in curves):
                continue
            found = (dx, dy, box)
            break
        if found is None:
            raise ValueError(f'No collision-free label position for {text} at {x}, {y}')
        dx, dy, box = found
        occupied.append(box)
        placements.append((x, y, text, color, dx, dy, anchor, box))
    # Lay out all text first; then draw only leaders that clear every other
    # label. Earlier leaders must not push later text far from its data point.
    for x, y, text, color, dx, dy, anchor, box in placements:
        leader_path = PlotPath([anchor, (anchor[0] + dx * scale, anchor[1] + dy * scale)])
        leader = (abs(dx) > 0 or abs(dy) > 24) and not any(
            leader_path.intersects_bbox(other, filled=False) for other in occupied if other is not box)
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points',
                    ha='center', va='center', color=color, fontsize=10, linespacing=1.35,
                    arrowprops=dict(arrowstyle='-', color=color, alpha=.35, linewidth=.65,
                                    shrinkA=5, shrinkB=7) if leader else None,
                    annotation_clip=False, zorder=5)
    return occupied


def make_chart(rows, metric, include_submissions, kind):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                               'axes.spines.top': False, 'axes.spines.right': False,
                               'axes.labelcolor': '#67756e', 'text.color': '#25352f',
                               'xtick.color': '#67756e', 'ytick.color': '#67756e', 'font.size': 11})
    accepted = [r for r in rows if r['status'] == 'accepted'
                and (metric != 'acceptance' or r.get('acceptance_rate') is not None)]
    visible_rows = [r for r in rows if r['status'] == 'accepted' or include_submissions]
    submissions = [r for r in visible_rows if r['status'] == 'submitted' and metric != 'acceptance']
    years = sorted({r['year'] for r in visible_rows})
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    # Identical plot bounds align years vertically across the two charts.
    fig.subplots_adjust(left=.095, right=.98, bottom=.12, top=.86)
    value_of = lambda row: (row['total'] if metric == 'total' else row['acceptance_rate']
                            if metric == 'acceptance' else row[kind][metric])
    fig.patch.set_facecolor('#ffffff')
    labels = []
    for conference, color in COLORS.items():
        series = [r for r in accepted if r['conference'] == conference]
        if not series:
            continue
        index = {r['year']: r for r in series}
        xs = list(range(min(index), max(index) + 1))
        ys = [value_of(index[y]) if y in index else float('nan') for y in xs]
        ax.plot(xs, ys, label=conference, color=color, marker='o', linewidth=2.4, markersize=6)
        for r in series:
            value = value_of(r)
            if metric == 'acceptance':
                accepted_count = r.get('acceptance_accepted')
                submitted_count = r.get('acceptance_submitted')
                numerator = f'{accepted_count:,}' if accepted_count is not None else '—'
                denominator = f'{submitted_count:,}' if submitted_count is not None else '—'
                if r.get('acceptance_submitted_estimated'):
                    denominator = '≈' + denominator
                rate_prefix = '≈' if r.get('acceptance_rate_estimated') else ''
                label = f'{rate_prefix}{value:.1f}%\n{numerator} / {denominator}'
            else:
                label = f'{value:,}' if kind == 'counts' else f'{value:.2f}%'
            labels.append((r['year'], value, label, color))
    for row in submissions:
        value = value_of(row)
        ax.scatter([row['year']], [value], marker='D', s=85, facecolors='white',
                   edgecolors=COLORS[row['conference']], linewidths=2, zorder=4,
                   label=f"{row['conference']} {row['year']} submission")
        labels.append((row['year'], value, f'{value:,}' if kind == 'counts' else f'{value:.2f}%', COLORS[row['conference']]))
    ax.set(xticks=years, ylabel='Acceptance rate (%)' if metric == 'acceptance' else
           'Papers' if kind == 'counts' else 'Share of all papers (%)')
    ax.set_xlim(min(years) - .25, max(years) + .3)
    ax.set_ylim(bottom=0)
    ax.margins(y=.24)
    if metric == 'acceptance':
        ax.set_ylim(0, max(r['acceptance_rate'] for r in accepted) * 1.24)
    ax.grid(axis='y', color='#dde4dc', linewidth=.7, alpha=.6)
    ax.set_axisbelow(True)
    ax.spines['left'].set_color('#dde4dc')
    ax.spines['bottom'].set_color('#dde4dc')
    if kind == 'counts':
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    else:
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=1))
    ax.legend(loc='upper left', fontsize=10, frameon=False, ncol=4, bbox_to_anchor=(0, 1.13))
    place_value_labels(ax, labels)
    stream = io.StringIO()
    fig.savefig(stream, format='svg', metadata={'Date': None})
    plt.close(fig)
    svg = stream.getvalue()
    return svg[svg.index('<svg'):]


def csv_text(rows):
    buffer = io.StringIO(newline='')
    fields = ['conference', 'year', 'status', 'total']
    for metric in METRICS:
        fields.extend([metric + '_count', metric + '_percent', metric + '_count_change', metric + '_pp_change'])
    fields += ['fetched_at', 'paper_info_sha256', 'filter_prompt_sha256']
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        item = {k: row[k] for k in ('conference', 'year', 'status', 'total', 'fetched_at', 'paper_info_sha256', 'filter_prompt_sha256')}
        for key in METRICS:
            item[key + '_count'], item[key + '_percent'] = row['counts'][key], row['rates'][key]
            change = row['change'][key] if row['change'] else None
            item[key + '_count_change'] = change['count'] if change else ''
            item[key + '_pp_change'] = change['percentage_points'] if change else ''
        writer.writerow(item)
    return buffer.getvalue()


def load_acceptance_statistics():
    """Keep official decision statistics separate from local snapshot counts."""
    from urllib.parse import urlparse
    statistics = json.loads(Path(__file__).with_name('acceptance_statistics.json').read_text())['statistics']
    seen = set()
    for item in statistics:
        identity = (item['conference'], item['year'])
        if identity in seen or item['conference'] not in COLORS:
            raise ValueError('Duplicate or unknown official conference statistics')
        seen.add(identity)
        rate = item['rate_percent']
        if not isinstance(rate, (int, float)) or not 0 <= rate <= 100:
            raise ValueError('Invalid acceptance rate')
        if item['rate_basis'] == 'official_counts':
            accepted, submitted = item['accepted'], item['submitted']
            if type(accepted) is not int or type(submitted) is not int or not 0 <= accepted <= submitted or not submitted:
                raise ValueError('Invalid official acceptance counts')
            if abs(rate - accepted / submitted * 100) > 1e-9:
                raise ValueError('Official rate does not match its own counts')
        elif item['rate_basis'] != 'official_reported_rate':
            raise ValueError('Unknown acceptance rate basis')
        source = urlparse(item['source_url'])
        if source.scheme != 'https' or source.hostname not in ('media.iclr.cc', 'media.icml.cc', 'media.neurips.cc'):
            raise ValueError('An official conference source is required')
    return statistics


def estimate_icml_2023_rate(statistics, rows):
    """Fill the requested historical gap, keeping estimates distinct from sources."""
    indexed = {(s['conference'], s['year']): s for s in statistics}
    identity = ('ICML', 2023)
    if identity in indexed or not any((r['conference'], r['year'], r['status']) == ('ICML', 2023, 'accepted') for r in rows):
        return statistics
    first, second = indexed[('ICML', 2024)], indexed[('ICML', 2025)]
    # One-year linear back-extrapolation: 27.5% - (27.0% - 27.5%) = 28.0%.
    rate = 2 * first['rate_percent'] - second['rate_percent']
    if not 0 < rate <= 100:
        raise ValueError('Invalid estimated historical acceptance rate')
    estimate = dict(conference='ICML', year=2023, accepted=None, submitted=None,
                    rate_percent=rate, rate_basis='estimated_from_adjacent_years', rate_estimated=True,
                    source_url=first['source_url'], supporting_source_urls=[first['source_url'], second['source_url']],
                    source_title='ICML 2023 estimate from 2024/2025 official rates',
                    estimation_note='2024 官方录用率 27.5%、2025 为 27.0%，按一年线性外推估计 2023 为 28.0%；录用数采用本地快照，投稿数由录用数除以估计录用率计算。')
    return sorted([*statistics, estimate], key=lambda s: (list(COLORS).index(s['conference']), s['year']))


def complete_acceptance_counts(statistics, rows):
    """Estimate missing submission counts without rewriting official facts."""
    local = {(r['conference'], r['year']): r['total'] for r in rows if r['status'] == 'accepted'}
    result = []
    for source in statistics:
        item = dict(source)
        identity = (item['conference'], item['year'])
        numerator = item['accepted'] if item['accepted'] is not None else local.get(identity)
        denominator = item['submitted']
        item['accepted_count_basis'] = 'official' if item['accepted'] is not None else 'local_snapshot'
        item['submitted_estimated'] = denominator is None
        if denominator is None:
            if numerator is None or item['rate_percent'] <= 0:
                raise ValueError(f'Cannot estimate submission count for {identity}')
            denominator = round(numerator / (item['rate_percent'] / 100))
        item['display_accepted'] = numerator
        item['display_submitted'] = denominator
        result.append(item)
    return result


def build_report(rows, selected, destination):
    destination.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    acceptance = complete_acceptance_counts(estimate_icml_2023_rate(load_acceptance_statistics(), rows), rows)
    official = {(item['conference'], item['year']): item for item in acceptance}
    plot_rows = [dict(row, acceptance_rate=official.get((row['conference'], row['year']), {}).get('rate_percent'),
                      acceptance_accepted=official.get((row['conference'], row['year']), {}).get('display_accepted'),
                      acceptance_submitted=official.get((row['conference'], row['year']), {}).get('display_submitted'),
                      acceptance_rate_estimated=official.get((row['conference'], row['year']), {}).get('rate_estimated', False),
                      acceptance_submitted_estimated=official.get((row['conference'], row['year']), {}).get('submitted_estimated', False))
                 for row in rows]
    missing_rates = [dict(conference=r['conference'], year=r['year']) for r in rows
                     if r['status'] == 'accepted' and (r['conference'], r['year']) not in official]
    payload = dict(acceptance_statistics=acceptance, missing_acceptance_rates=missing_rates, generated_at=generated, denominator='All scored papers in the same conference/year/status',
                   score=5, metrics=METRICS, rows=rows)
    (destination / 'statistics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    (destination / 'statistics.csv').write_text(csv_text(rows), encoding='utf-8-sig')
    charts, totals = [], []
    for key in ('mech', 'comp', 'either', 'total', 'acceptance'):
        for include in (False, True):
            for kind in (('counts',) if key == 'total' else ('rates',) if key == 'acceptance' else ('counts', 'rates')):
                svg = make_chart(plot_rows, key, include, kind)
                name = f"{key}_{kind}_{'with_submission' if include else 'accepted'}"
                (destination / (name + '.svg')).write_text(svg, encoding='utf-8')
                for color, token in {'#ffffff': '--surface', '#67756e': '--muted',
                                     '#25352f': '--text', '#dde4dc': '--border',
                                     '#b17a46': '--iclr', '#527eac': '--icml',
                                     '#226b54': '--neurips'}.items():
                    svg = svg.replace(color, f'var({token})')
                element = f'<div class="chart" data-metric="{key}" data-kind="{kind}" data-submission="{str(include).lower()}" hidden>{svg}</div>'
                (totals if key in ('total', 'acceptance') else charts).append(element)
    template = Path(__file__).with_name('direction_statistics.html').read_text(encoding='utf-8')
    html = template.replace('<!--CHARTS-->', '\n'.join(charts)).replace('<!--TOTAL_CHARTS-->', '\n'.join(totals))
    html = html.replace('__REPORT_DATA__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
    html = html.replace('__GENERATED__', escape(generated.replace('T', ' ')))
    (destination / 'index.html').write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs_dir', type=Path, default=Path(__file__).resolve().parent / 'runs/gpt-6-luna_runs')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'runs/direction_statistics')
    args = parser.parse_args()
    rows, selected = collect(args.runs_dir)
    build_report(rows, selected, args.output)
    print(args.output.resolve() / 'index.html')
    print(f'{len(rows)} corpora; {sum(r["total"] for r in rows):,} scored papers')


if __name__ == '__main__':
    main()

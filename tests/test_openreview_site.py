import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from daily_arxiv.reporting.site import export_site
from openreview.site import export_openreview
from daily_arxiv.filtering.pipeline import save_reports


class ConferenceSiteTests(unittest.TestCase):
    def corpus(self, root, name='ICLR_2027_submitted', scores=None):
        year = int(name.split('_')[1])
        status = 'submitted' if name.endswith('_submitted') else 'accepted'
        folder = root / 'runs/gpt-6-luna_runs' / name
        papers = [dict(title=f'Paper {i}', abstract='An abstract.', url=f'https://openreview.net/forum?id={i}',
                       eval=dict(scores=dict(zip(('MechInterp', 'CompGen', 'Reasoning', 'Systems'), values)), reason='Reason.'))
                  for i, values in enumerate(scores or [(4, 1, 5, 1), (1, 5, 1, 1), (1, 1, 5, 5)])]
        day = f'{year}-01-01'
        translated = [dict(p, zh_title='标题', zh_abstract='摘要。',
                           eval=dict(p['eval'], zh_reason='理由。')) for p in papers]
        save_reports(papers, translated, day, 4, folder,
                     source_metadata=dict(source='openreview', conference='ICLR', conference_year=year,
                                          paper_status=status))
        return folder

    def test_two_direction_pool_scores_navigation_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = self.corpus(root)
            original = folder / 'html_reports/report_2027-01-01.html'
            before = original.read_bytes()
            output = root / 'site'
            section = export_openreview(root, output)
            rows = json.loads((output / 'openreview/archive.json').read_text())
            self.assertEqual(rows[0]['papers'], 2)
            self.assertEqual(rows[0]['counts'], {'MechInterp': {'4': 1, '5': 0}, 'CompGen': {'4': 0, '5': 1}})
            html = (output / rows[0]['url']).read_text()
            self.assertEqual(html.count('<article '), 0)
            chunks = list((output / rows[0]['url']).parent.glob('data_*/*.json'))
            cards = ''.join(card for chunk in chunks for card in json.loads(chunk.read_text()))
            self.assertEqual(cards.count('<article '), 2)
            self.assertIn('data-score="4"', cards)
            self.assertNotIn('Reasoning', html)
            self.assertNotIn('Systems', html)
            self.assertIn('../../../index.html#conference-reports', html)
            self.assertIn('在审投稿 · 尚未录用', section)
            self.assertEqual(original.read_bytes(), before)
            self.corpus(root, 'ICLR_2026', [(1, 1, 5, 5)])
            export_openreview(root, output)
            empty = output / 'openreview/ICLR_2026/html_reports/report_2026-01-01.html'
            self.assertNotIn('<article ', empty.read_text())
            self.assertNotIn('Reasoning', empty.read_text())
            import shutil
            shutil.rmtree(folder)
            export_openreview(root, output)
            self.assertFalse((output / 'openreview/ICLR_2027_submitted').exists())

    def test_homepage_appends_statistics_without_changing_daily_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            source, output = project / 'openreview', project / 'site'
            self.corpus(source)
            statistics = source / 'runs/direction_statistics'
            statistics.mkdir()
            (statistics / 'index.html').write_text('<html><body>Charts</body></html>')
            (statistics / 'statistics.csv').write_text('counts')
            with patch('daily_arxiv.reporting.site.PROJECT_DIR', project):
                export_site(project / 'model', output, allow_pending=True, update_openreview=True)
                combined = (output / 'index.html').read_text()
                with patch('openreview.site.export_openreview', return_value=''):
                    export_site(project / 'model', project / 'daily-only', allow_pending=True)
            daily = (project / 'daily-only/index.html').read_text()
            self.assertEqual(combined.split('<section id="openreview"')[0], daily.split('</main>')[0])
            self.assertIn('src="openreview/direction_statistics/embedded.html"', combined)
            self.assertEqual((output / 'openreview/direction_statistics/statistics.csv').read_text(), 'counts')

    def test_daily_export_preserves_conference_section_and_assets_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'site'
            output.mkdir()
            section = '<section id="openreview" aria-label="会议"><h2>已发布的会议</h2></section>'
            (output / 'index.html').write_text('<main>' + section + '</main>')
            asset = output / 'openreview/report.html'
            asset.parent.mkdir()
            asset.write_text('existing report')
            with patch('openreview.site.export_openreview', side_effect=AssertionError('Must not rebuild')):
                export_site(root / 'model', output, allow_pending=True)
            self.assertIn(section, (output / 'index.html').read_text())
            self.assertEqual(asset.read_text(), 'existing report')


if __name__ == '__main__':
    unittest.main()

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from daily_arxiv.fetching.snapshots import load_paper_input
from daily_arxiv.filtering.pipeline import save_reports
from daily_arxiv.reporting.html import generate_mobile_html_report
from openreview.fetch_neurips_papers import (
    ABSTRACTS_URL, LIST_URL, fetch_snapshot, main_track_papers, parse_abstract,
)
from openreview.process_snapshots import snapshot_queue
from openreview.site import export_openreview


def event(identifier=100, forum='paper1', kind='poster'):
    return dict(id=identifier, paper_url=f'https://openreview.net/forum?id={forum}',
                sourceurl='https://openreview.net/group?id=NeurIPS.cc/2026/Conference',
                name='Mechanisms of model composition', authors=[dict(fullname='Author')],
                decision=f'Accept ({kind})', virtualsite_url=f'/virtual/2026/poster/{identifier}')


class WebsiteTests(unittest.TestCase):
    def test_deduplication_and_track_exclusion(self):
        workshop = event(103, 'workshop')
        workshop['sourceurl'] = 'https://openreview.net/group?id=NeurIPS.cc/2026/Workshop'
        dataset = event(104, 'dataset')
        dataset['sourceurl'] = 'https://openreview.net/group?id=NeurIPS.cc/2026/Evaluations_and_Datasets_Track'
        rows = [event(), event(101, kind='oral'), workshop, dataset]
        papers = main_track_papers(dict(results=rows, count=4, next=None), {'101': 'Full abstract $x$.'}, 2026)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]['venue'], 'NeurIPS 2026 oral')
        self.assertEqual(papers[0]['abstract'], 'Full abstract $x$.')
        self.assertNotIn('pdf_url', papers[0])

    def test_incomplete_list_and_conflicting_identity_are_rejected(self):
        for payload in [dict(results=[event()], count=2), dict(results=[event()], count=1, next='/next')]:
            with self.assertRaises(ValueError):
                main_track_papers(payload, {}, 2026)
        changed = dict(event(101), name='A different paper')
        with self.assertRaisesRegex(ValueError, 'Conflicting titles'):
            main_track_papers(dict(results=[event(), changed], count=2), {}, 2026)

    def test_html_abstract_preserves_math_entities_and_excludes_surroundings(self):
        html = '<h1>Title</h1><div id="abstractText"><div><p>Full $x_1$ &amp; $x_2$.<br/>Next paragraph.<br/>End.</p></div></div><p>Show more</p>'
        text = parse_abstract(html)
        self.assertIn('Full $x_1$ & $x_2$.', text)
        self.assertIn('Next paragraph.', text)
        self.assertIn('End.', text)
        self.assertNotIn('Show more', text)
        with self.assertRaises(ValueError):
            parse_abstract('<html>Login required</html>')

    def test_snapshot_is_compatible_and_failed_refresh_preserves_previous_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            listing = dict(results=[event()], count=1, next=None)
            client = Mock()
            def get(url):
                if url == LIST_URL.format(year=2026):
                    return Mock(json=lambda: copy.deepcopy(listing))
                if url == ABSTRACTS_URL.format(year=2026):
                    return Mock(json=lambda: {})
                return Mock(text='<a href="https://openreview.net/forum?id=paper1">Paper</a>'
                                 '<div id="abstractText">A complete abstract with $x$.</div>')
            client.get.side_effect = get
            path = fetch_snapshot(2026, root, workers=1, client=client)
            papers, day = load_paper_input(path)
            self.assertEqual(len(papers), 1)
            self.assertEqual(snapshot_queue(root)[0]['papers'], 1)
            metadata = json.loads(path.read_text())['metadata']
            self.assertEqual(metadata['discovery_method'], 'neurips_official_website')
            row = dict(papers[0], zh_title='论文', zh_abstract='完整摘要包含 $x$。',
                       eval=dict(scores=dict(MechInterp=4, CompGen=1, Reasoning=1, Systems=1),
                                 reason='Mechanisms.', zh_reason='机制。'))
            html = Path(generate_mobile_html_report([row], day, root / 'report', metadata)).read_text()
            self.assertIn('NeurIPS 官方网站公开元数据', html)
            self.assertIn('NeurIPS 官网 ↗', html)
            self.assertIn(papers[0]['official_url'], html)
            self.assertIn('Mechanisms of model composition', html)
            save_reports(papers=[dict(papers[0], eval={k: v for k, v in row['eval'].items() if k != 'zh_reason'})],
                         translated=[row], target_date=day, threshold=4,
                         save_dir=root / 'runs/gpt-6-luna_runs/NeurIPS_2026', source_metadata=metadata)
            section = export_openreview(root, root / 'site')
            self.assertIn('NeurIPS 2026', section)
            published = root / f'site/openreview/NeurIPS_2026/html_reports/report_{day}.html'
            self.assertIn('NeurIPS 官方网站公开元数据', published.read_text())
            previous = path.read_bytes()
            listing['results'].append(event(102, 'new-paper'))
            listing['count'] = 2
            client.get.side_effect = lambda url: get(url) if '/static/' in url else Mock(text='No abstract')
            with self.assertRaisesRegex(RuntimeError, 'no partial snapshot saved'):
                fetch_snapshot(2026, root, workers=1, client=client)
            self.assertEqual(path.read_bytes(), previous)


if __name__ == '__main__':
    unittest.main()

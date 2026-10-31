"""Newest-first ordering and retry/resume for the long-running worker."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from daily_arxiv.fetching.snapshots import save_snapshot
from daily_arxiv.filtering.pipeline import save_reports, write_json
from daily_arxiv.filtering.prompts import normalize_evaluation, CATEGORIES
from openreview.process_snapshots import main, snapshot_queue, reports_complete


def fixture(root, conference, year):
    venue = f'{conference}.cc/{year}/Conference'
    papers = [{'title': 'Features', 'abstract': 'Interpreting features.',
               'url': f'https://openreview.net/forum?id={conference}{year}', 'venueid': venue,
               'venue': f'{conference} {year} Poster'}]
    return save_snapshot(papers, f'{year}-01-01', Path(root) / f'{conference}_{year}', source='openreview',
                         metadata_extra={'conference': conference, 'conference_year': year, 'venueid': venue})


class QueueTests(unittest.TestCase):
    def test_usage_limit_can_outlast_five_hours_without_exhausting_queue_retries(self):
        with tempfile.TemporaryDirectory() as root:
            inputs, runs = Path(root) / 'inputs', Path(root) / 'runs'
            fixture(inputs, 'NeurIPS', 2026)
            cache = runs / 'gpt-6-luna_runs/NeurIPS_2026/.codex_cache/evaluation/completed.json'
            write_json(cache, {'completed_before_limit': True})
            before = cache.read_bytes()
            results = [Mock(returncode=1) for _ in range(22)] + [Mock(returncode=0)]
            argv = ['--paper_info_dir', str(inputs), '--runs_dir', str(runs),
                    '--conferences', 'NeurIPS', '--years', '2026',
                    '--retry_delay', '300', '--max_retry_delay', '900']
            # Exercise the controller; artifact validation is covered separately.
            def complete(output, day, identity, check_marker=True):
                return not check_marker
            with patch('openreview.process_snapshots.subprocess.run', side_effect=results) as child, \
                    patch('openreview.process_snapshots.reports_complete', side_effect=complete), \
                    patch('openreview.process_snapshots.time.sleep') as sleep:
                self.assertEqual(main(argv), 0)
            waits = [call.args[0] for call in sleep.call_args_list]
            self.assertGreaterEqual(sum(waits), 5 * 3600)
            self.assertEqual(max(waits), 900)
            self.assertEqual(child.call_count, 23)
            self.assertEqual(cache.read_bytes(), before)
            state = json.loads((runs / '.queue/gpt-6-luna/state.json').read_text())
            self.assertEqual(state['status'], 'complete')

    def test_single_conference_selection_leaves_other_corpora_out(self):
        with tempfile.TemporaryDirectory() as root:
            for conference, year in [('ICLR', 2026), ('NeurIPS', 2025), ('NeurIPS', 2026)]:
                fixture(root, conference, year)
            jobs = snapshot_queue(root, conferences=['NeurIPS'], years=[2026])
            self.assertEqual([(j['conference'], j['year']) for j in jobs], [('NeurIPS', 2026)])

    def test_cross_year_order_skips_missing_and_empty_venues(self):
        with tempfile.TemporaryDirectory() as root:
            for conference, year in [('ICLR', 2027), ('ICML', 2026), ('ICLR', 2026),
                                     ('NeurIPS', 2025), ('ICML', 2025), ('ICLR', 2025)]:
                fixture(root, conference, year)
            save_snapshot([], '2026-01-01', Path(root) / 'NeurIPS_2026', source='openreview',
                          metadata_extra={'conference': 'NeurIPS', 'conference_year': 2026})
            self.assertEqual([(j['conference'], j['year']) for j in snapshot_queue(root)],
                             [('ICLR', 2027), ('ICML', 2026), ('ICLR', 2026),
                              ('NeurIPS', 2025), ('ICML', 2025), ('ICLR', 2025)])

    def test_failed_job_retries_then_restart_skips_completed_job(self):
        with tempfile.TemporaryDirectory() as root:
            fixture(Path(root) / 'inputs', 'ICLR', 2026)
            runs = Path(root) / 'runs'
            attempts = []

            def inference(command, **kwargs):
                attempts.append(command)
                if len(attempts) == 1:
                    return Mock(returncode=1)
                output = Path(command[command.index('--save_dir') + 1])
                source = json.loads(Path(command[command.index('--input_json') + 1]).read_text())
                papers = source['papers']
                for paper in papers:
                    paper['eval'] = normalize_evaluation({'scores': {c: 4 for c in CATEGORIES}, 'reason': 'Relevant.'})
                translated = [dict(paper, zh_title='特征', zh_abstract='解释特征。',
                                   eval=dict(paper['eval'], zh_reason='相关。')) for paper in papers]
                save_reports(papers, translated, '2026-01-01', 4, output,
                             input_digest=source['metadata']['paper_info_sha256'],
                             source_metadata=source['metadata'])
                return Mock(returncode=0)

            argv = ['--paper_info_dir', str(Path(root) / 'inputs'), '--runs_dir', str(runs), '--retry_delay', '1']
            with patch('openreview.process_snapshots.subprocess.run', side_effect=inference), patch('openreview.process_snapshots.time.sleep') as sleep:
                self.assertEqual(main(argv), 0)
                sleep.assert_called_once_with(1)
                self.assertEqual(len(attempts), 2)
                self.assertEqual(main(argv), 0)
                self.assertEqual(len(attempts), 2)
            state = json.loads((runs / '.queue/gpt-6-luna/state.json').read_text())
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['completed'], ['ICLR_2026'])
            output = runs / 'gpt-6-luna_runs/ICLR_2026'
            marker = json.loads((output / '.queue_complete.json').read_text())
            self.assertTrue(reports_complete(output, '2026-01-01', marker))
            # Matching markers must not hide subsequent translation corruption.
            zh_path = output / 'hits_zh/hits_zh_2026-01-01.json'
            payload = json.loads(zh_path.read_text())
            payload['hits_zh'][0]['zh_abstract'] = ''
            write_json(zh_path, payload)
            self.assertFalse(reports_complete(output, '2026-01-01', marker))
            payload['hits_zh'][0]['zh_abstract'] = '解释特征。'
            write_json(zh_path, payload)
            self.assertFalse(reports_complete(output, '2026-01-01',
                                             dict(marker, translation_contract_sha256='obsolete')))
            (output / 'html_reports/report_2026-01-01.html').unlink()
            self.assertFalse(reports_complete(output, '2026-01-01', marker))


if __name__ == '__main__':
    unittest.main()

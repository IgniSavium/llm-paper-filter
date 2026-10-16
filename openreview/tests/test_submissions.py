"""Public submission boundaries, corpus isolation and shared report integration."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from daily_arxiv.filtering.pipeline import load_filter_input, run_filter, save_reports
from openreview.fetching import API_URLS, OpenReviewClient, get_snapshot, normalize_note
from openreview.process_snapshots import main as queue_main, snapshot_queue
from openreview.run_openreview import main
from test_openreview import FakeLLM, note


def submission(index=0, *, year=2027, v2=True):
    result = note(index, year=year, v2=v2)
    for field, text in [('venueid', f'ICLR.cc/{year}/Conference/Submission'),
                        ('venue', f'ICLR {year} Conference Submission' if v2 else f'Submitted to ICLR {year}')]:
        result['content'][field] = {'value': text} if v2 else text
    result['readers'] = ['everyone']
    invitation = f'ICLR.cc/{year}/Conference/-/' + ('Submission' if v2 else 'Blind_Submission')
    result['invitations' if v2 else 'invitation'] = [invitation] if v2 else invitation
    return result


class SubmissionTests(unittest.TestCase):
    def test_only_public_active_main_track_submissions_survive(self):
        active = submission()
        private = submission(1)
        private['readers'] = ['ICLR.cc/2027/Conference/Reviewers']
        decided = []
        for index, label in enumerate(['ICLR 2027 Poster', 'Rejected by ICLR 2027',
                                       'Withdrawn from ICLR 2027', 'Desk rejected by ICLR 2027'], 2):
            record = submission(index)
            record['content']['venue'] = {'value': label}
            decided.append(record)
        workshop = submission(6)
        workshop['content']['venueid'] = {'value': 'ICLR.cc/2027/Workshop/Submission'}
        deleted = submission(7)
        deleted['ddate'] = 1
        records = [active, private, *decided, workshop, deleted]
        client = OpenReviewClient(page_size=5)
        client.request = Mock(side_effect=[{'count': len(records), 'notes': records[:5]},
                                           {'count': len(records), 'notes': records[5:]}])
        papers, api = client.fetch('ICLR', 2027, paper_status='submitted')
        self.assertEqual([p['openreview_id'] for p in papers], ['paper0'])
        self.assertEqual(api, API_URLS[0])
        self.assertEqual(papers[0]['status'], 'ICLR 2027 Submitted')
        params = [call.args[1] for call in client.request.call_args_list]
        self.assertEqual(params[0]['invitation'], 'ICLR.cc/2027/Conference/-/Submission')
        self.assertNotIn('content.venueid', params[0])
        self.assertEqual(params[1]['offset'], 5)

    def test_old_blind_submissions_with_shared_venueid(self):
        record = submission(year=2023, v2=False)
        record['content']['venueid'] = 'ICLR.cc/2023/Conference'
        client = OpenReviewClient()
        client.request = Mock(return_value={'notes': [record], 'count': 1})
        papers, api = client.fetch('ICLR', 2023, paper_status='submitted')
        self.assertEqual(len(papers), 1)
        self.assertEqual(api, API_URLS[1])
        self.assertTrue(client.request.call_args.args[1]['invitation'].endswith('/Blind_Submission'))

    def test_invalid_response_and_nonpublic_fields_fail_closed(self):
        cases = []
        for field in ('title', 'abstract'):
            record = submission()
            record['content'][field]['readers'] = ['ICLR.cc/2027/Conference/Authors']
            cases.append(record)
        reply = submission()
        reply['replyto'] = 'another-paper'
        cases.append(reply)
        wrong_invitation = submission()
        wrong_invitation['invitations'] = ['ICLR.cc/2027/Workshop/-/Submission']
        cases.append(wrong_invitation)
        missing = submission()
        missing['content'].pop('abstract')
        cases.append(missing)
        for record in cases:
            client = OpenReviewClient()
            client.request = Mock(return_value={'notes': [record], 'count': 1})
            with self.subTest(record=record), self.assertRaises(ValueError):
                client.fetch('ICLR', 2027, paper_status='submitted')

    def test_nonpublic_authors_are_redacted_even_when_login_can_read_them(self):
        record = submission()
        record['content']['authors'] = {'value': ['Private Author'], 'readers': ['Authors']}
        paper = normalize_note(record, 'ICLR', 2027, paper_status='submitted')
        self.assertEqual(paper['authors'], [])
        self.assertNotIn('Private Author', json.dumps(paper))

    def test_no_visible_submissions_are_skipped_and_never_cached(self):
        with tempfile.TemporaryDirectory() as root:
            client = OpenReviewClient()
            client.request = Mock(return_value={'notes': [], 'count': 0})
            self.assertIsNone(get_snapshot('ICLR', 2027, root, client=client, paper_status='submitted'))
            self.assertFalse(list(Path(root).rglob('*.json')))
            self.assertEqual(client.request.call_count, 2)

    def test_submitted_snapshot_cache_refresh_and_filter_reports(self):
        with tempfile.TemporaryDirectory() as root:
            client = Mock()
            client.fetch.return_value = ([normalize_note(submission(), 'ICLR', 2027, paper_status='submitted')], API_URLS[0])
            source = get_snapshot('ICLR', 2027, root, client=client, paper_status='submitted')
            self.assertEqual(source.parent.name, 'ICLR_2027_submitted')
            self.assertEqual(get_snapshot('ICLR', 2027, root, client=client, paper_status='submitted'), source)
            self.assertEqual(client.fetch.call_count, 1)
            self.assertEqual(snapshot_queue(root), [])
            self.assertEqual(len(snapshot_queue(root, 'submitted')), 1)
            old = source.read_bytes()
            client.fetch.side_effect = RuntimeError('Denied access')
            with self.assertRaises(RuntimeError):
                get_snapshot('ICLR', 2027, root, client=client, refresh=True, paper_status='submitted')
            self.assertEqual(source.read_bytes(), old)
            papers, date, metadata = load_filter_input(argparse.Namespace(input_json=source, date='2027-01-01', limit=None))
            self.assertEqual(metadata['coverage'], 'public_active_submissions_only')
            llm = FakeLLM()
            evaluated, translated = run_filter(papers, llm, Path(root) / 'cache')
            calls = llm.calls
            run_filter(copy.deepcopy(papers), llm, Path(root) / 'cache')
            self.assertEqual(llm.calls, calls)
            save_reports(evaluated, translated, date, 4, root, source_metadata=metadata)
            saved = json.loads((Path(root) / 'all_papers/all_papers_2027-01-01.json').read_text())
            self.assertEqual(saved['metadata']['submitted_papers'], 1)
            self.assertNotIn('accepted_papers', saved['metadata'])
            report = (Path(root) / 'html_reports/report_2027-01-01.html').read_text()
            self.assertIn('在审投稿', report)
            self.assertNotIn('主会已录用论文', report)
            with patch('openreview.process_snapshots.subprocess.run') as inference:
                self.assertEqual(queue_main(['--paper_status', 'submitted', '--paper_info_dir', root, '--dry_run']), 0)
                inference.assert_not_called()

    def test_runner_isolates_outputs_and_dry_run_has_no_side_effects(self):
        with patch('openreview.run_openreview.get_snapshot') as fetch, \
                patch('openreview.fetch_openreview_papers.load_credentials') as credentials, \
                patch('openreview.run_openreview.subprocess.run') as inference, \
                patch('builtins.print') as printed:
            self.assertEqual(main(['--conferences', 'ICLR', '--years', '2027',
                                   '--paper_status', 'submitted', '--dry_run']), 0)
            fetch.assert_not_called()
            credentials.assert_not_called()
            inference.assert_not_called()
            command = printed.call_args.args[0]
            self.assertIn('paper_info/ICLR_2027_submitted/', command)
            self.assertIn('gpt-6-luna_runs/ICLR_2027_submitted', command)


if __name__ == '__main__':
    unittest.main()

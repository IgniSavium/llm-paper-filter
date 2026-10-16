"""Offline retrieval integrity and shared-filter integration checks."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from daily_arxiv.filtering.pipeline import load_filter_input, run_filter, save_reports, validate_results
from daily_arxiv.filtering.prompts import CATEGORIES
from openreview.fetching import API_URLS, OpenReviewClient, default_years, get_snapshot, load_credentials, normalize_note
from openreview.run_openreview import main


def note(index=0, *, v2=True, conference="ICLR", year=2025):
    content = {"title": f"Interpretability {index}", "abstract": "Explaining model features.",
               "authors": ["Author"], "venueid": f"{conference}.cc/{year}/Conference",
               "venue": f"{conference} {year} Poster"}
    if v2:
        content = {key: {"value": value} for key, value in content.items()}
    return {"id": f"paper{index}", "number": index + 1, "replyto": None, "content": content}


def response(payload=None, status=200, headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = payload
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError(response=result)
    return result


class FetchTests(unittest.TestCase):
    def test_shared_conference_id_does_not_make_submissions_accepted(self):
        accepted = note(0, v2=False, year=2023)
        accepted['content']['venue'] = 'ICLR 2023 notable top 25%'
        submitted = note(1, v2=False, year=2023)
        submitted['content']['venue'] = 'Submitted to ICLR 2023'
        missing = note(2, v2=False, year=2023)
        missing['content'].pop('venue')
        for invalid in (submitted, missing):
            with self.assertRaisesRegex(ValueError, 'accepted venue'):
                normalize_note(invalid, 'ICLR', 2023)
        client = OpenReviewClient()
        client.request = Mock(return_value={'count': 3, 'notes': [accepted, submitted, missing]})
        papers, _ = client.fetch('ICLR', 2023)
        self.assertEqual([p['openreview_id'] for p in papers], ['paper0'])

    def test_cached_snapshot_with_submission_is_rejected(self):
        from daily_arxiv.fetching.snapshots import save_snapshot
        paper = normalize_note(note(v2=False, year=2023), 'ICLR', 2023)
        paper['venue'] = 'Submitted to ICLR 2023'
        with tempfile.TemporaryDirectory() as directory:
            save_snapshot([paper], '2023-01-01', Path(directory) / 'ICLR_2023', source='openreview',
                          metadata_extra={'venueid': 'ICLR.cc/2023/Conference'})
            with self.assertRaisesRegex(ValueError, 'Invalid conference snapshot'):
                get_snapshot('ICLR', 2023, directory)

    def test_defaults_include_current_year(self):
        year = datetime.now(timezone.utc).year
        self.assertEqual(default_years(), list(range(year - 3, year + 1)))

    def test_v1_v2_normalization_and_acceptance_boundary(self):
        self.assertEqual(normalize_note(note(), "ICLR", 2025),
                         normalize_note(note(v2=False), "ICLR", 2025))
        for venue in ("ICLR.cc/2025/Conference/Rejected", "ICLR.cc/2025/Workshop",
                      "ICLR.cc/2025/Conference/Submission", "NeurIPS.cc/2025/Conference"):
            invalid = note()
            invalid["content"]["venueid"]["value"] = venue
            with self.subTest(venue=venue), self.assertRaises(ValueError):
                normalize_note(invalid, "ICLR", 2025)
        for field in ("title", "abstract"):
            invalid = note()
            invalid["content"].pop(field)
            with self.assertRaises(ValueError):
                normalize_note(invalid, "ICLR", 2025)

    def test_paginate_until_count_even_if_server_returns_short_page(self):
        client = OpenReviewClient(page_size=1000)
        client.request = Mock(side_effect=[{"count": 3, "notes": [note(0)]},
                                           {"notes": [note(1)]}, {"count": 3, "notes": [note(2)]}])
        papers, api = client.fetch("ICLR", 2025)
        self.assertEqual(len(papers), 3)
        self.assertEqual(api, API_URLS[0])
        params = [call.args[1] for call in client.request.call_args_list]
        self.assertNotIn('offset', params[0])
        self.assertEqual([p.get('offset', 0) for p in params], [0, 1, 2])
        self.assertTrue(all(p['content.venueid'] == 'ICLR.cc/2025/Conference' for p in params))

    def test_incomplete_duplicate_and_changing_pages_fail(self):
        cases = [
            [{"count": 2, "notes": [note()]}, {"count": 2, "notes": []}],
            [{"count": 2, "notes": [note()]}, {"count": 2, "notes": [note()]}],
            [{"count": 2, "notes": [note()]}, {"count": 3, "notes": [note(1)]}],
            [{"notes": [note()]}],
        ]
        for pages in cases:
            with self.subTest(pages=pages):
                client = OpenReviewClient()
                client.request = Mock(side_effect=pages)
                with self.assertRaises(RuntimeError):
                    client.fetch("ICLR", 2025)

    def test_old_iclr_uses_api1(self):
        client = OpenReviewClient()
        client.request = Mock(return_value={"count": 1, "notes": [note(v2=False, year=2023)]})
        papers, api = client.fetch("ICLR", 2023)
        self.assertEqual(api, API_URLS[1])
        self.assertEqual(papers[0]['title'], 'Interpretability 0')
        self.assertNotIn('count', client.request.call_args.args[1])

    def test_empty_v2_falls_back_to_v1(self):
        client = OpenReviewClient()
        client.request = Mock(side_effect=[{"count": 0, "notes": []},
                                           {"count": 1, "notes": [note(v2=False)]}])
        _, api = client.fetch("ICLR", 2025)
        self.assertEqual(api, API_URLS[1])

    def test_rate_limit_and_network_errors_retry_but_access_denial_does_not(self):
        session = Mock()
        session.get.side_effect = [response(status=429, headers={"Retry-After": "1"}),
                                   requests.Timeout(), response({"count": 0, "notes": []})]
        client = OpenReviewClient(session=session, retries=3)
        with patch('openreview.fetching.time.sleep') as sleep:
            self.assertEqual(client.request(API_URLS[0], {}), {"count": 0, "notes": []})
            self.assertEqual(sleep.call_count, 2)
        session.get.reset_mock()
        session.get.side_effect = [response(status=403)]
        with self.assertRaisesRegex(RuntimeError, 'denied access'):
            client.request(API_URLS[0], {})
        self.assertEqual(session.get.call_count, 1)

    def test_snapshots_reuse_validate_and_preserve_old_on_failed_refresh(self):
        with tempfile.TemporaryDirectory() as root:
            client = Mock()
            client.fetch.return_value = ([normalize_note(note(), 'ICLR', 2025)], API_URLS[0])
            path = get_snapshot('ICLR', 2025, root, client=client)
            old = path.read_bytes()
            self.assertEqual(get_snapshot('ICLR', 2025, root, client=client), path)
            self.assertEqual(client.fetch.call_count, 1)
            client.fetch.side_effect = RuntimeError('Access denied')
            with self.assertRaises(RuntimeError):
                get_snapshot('ICLR', 2025, root, refresh=True, client=client)
            self.assertEqual(path.read_bytes(), old)
            payload = json.loads(old)
            payload['papers'][0]['title'] = 'Tampered'
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                get_snapshot('ICLR', 2025, root, client=client)

    def test_empty_current_year_is_not_cached_but_old_empty_year_fails(self):
        with tempfile.TemporaryDirectory() as root:
            year = datetime.now(timezone.utc).year
            client = OpenReviewClient()
            client.request = Mock(return_value={"count": 0, "notes": []})
            self.assertIsNone(get_snapshot('ICLR', year, root, client=client))
            self.assertFalse(list(Path(root).rglob('*.json')))
            with self.assertRaisesRegex(RuntimeError, 'No public accepted papers'):
                get_snapshot('ICLR', year - 1, root, client=client)


class FakeLLM:
    model, effort = 'gpt-6-luna', 'medium'

    def __init__(self):
        self.calls = 0

    def request(self, instruction, records, translation=False):
        self.calls += 1
        results = []
        for record in records:
            if translation:
                results.append({'id': record['id'], 'zh_title': '可解释性',
                                'zh_abstract': '解释模型特征。', 'zh_reason': '相关研究。'})
            else:
                score = 5 if record['title'].endswith('0') else 1
                results.append({'id': record['id'], 'scores': {c: score for c in CATEGORIES},
                                'reason': 'Relevant research.'})
        return validate_results({'results': results}, [r['id'] for r in records], translation)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        credentials = patch('openreview.fetch_openreview_papers.load_credentials', return_value={})
        credentials.start()
        self.addCleanup(credentials.stop)

    def test_snapshot_shared_filter_caches_metadata_and_report(self):
        with tempfile.TemporaryDirectory() as root:
            client = Mock()
            client.fetch.return_value = ([normalize_note(note(i), 'ICLR', 2025) for i in range(2)], API_URLS[0])
            source = get_snapshot('ICLR', 2025, Path(root) / 'inputs', client=client)
            args = argparse.Namespace(input_json=source, date='2025-01-01', limit=None)
            papers, date, metadata = load_filter_input(args)
            llm = FakeLLM()
            evaluated, translated = run_filter(papers, llm, Path(root) / 'cache')
            self.assertEqual(len(evaluated), 2)
            self.assertEqual(len(translated), 1)
            calls = llm.calls
            self.assertEqual(run_filter(copy.deepcopy(papers), llm, Path(root) / 'cache'), (evaluated, translated))
            self.assertEqual(llm.calls, calls)
            save_reports(evaluated, translated, date, 4, root, source_metadata=metadata)
            report = (Path(root) / 'html_reports/report_2025-01-01.html').read_text()
            self.assertIn('ICLR 2025', report)
            self.assertIn('在 OpenReview 阅读', report)
            self.assertNotIn('Daily ArXiv', report)
            saved = json.loads((Path(root) / 'all_papers/all_papers_2025-01-01.json').read_text())
            self.assertEqual(saved['metadata']['venueid'], 'ICLR.cc/2025/Conference')
            self.assertEqual(saved['metadata']['accepted_papers'], 2)
            self.assertNotIn('version_updates', saved['metadata'])

    def test_runner_dry_run_has_no_fetch_or_inference(self):
        with patch('openreview.run_openreview.get_snapshot') as fetch, patch('openreview.run_openreview.subprocess.run') as inference:
            self.assertEqual(main(['--years', '2026', '--dry_run']), 0)
            fetch.assert_not_called()
            inference.assert_not_called()

    def test_runner_skips_unavailable_current_year_and_stops_on_inference_failure(self):
        with patch('openreview.run_openreview.get_snapshot', return_value=None), patch('openreview.run_openreview.subprocess.run') as inference:
            self.assertEqual(main(['--years', '2026']), 0)
            inference.assert_not_called()
        with patch('openreview.run_openreview.get_snapshot', return_value=Path('/tmp/input.json')), patch('openreview.run_openreview.subprocess.run', return_value=Mock(returncode=2)) as inference:
            self.assertEqual(main(['--years', '2025']), 2)
            self.assertEqual(inference.call_count, 1)


class CredentialTests(unittest.TestCase):
    def test_local_file_validation_and_optional_token(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'credentials.json'
            self.assertEqual(load_credentials(path), {})
            with self.assertRaises(ValueError):
                load_credentials(path, required=True)
            path.write_text('{"username":"", "password":""}')
            with self.assertRaisesRegex(ValueError, 'Fill in username and password'):
                load_credentials(path)
            path.write_text(json.dumps({'username': 'person@example.com', 'password': ' example-secret '}))
            self.assertEqual(load_credentials(path)['password'], ' example-secret ')
            path.write_text('{"token": "Bearer example-token"}')
            self.assertEqual(load_credentials(path), {'token': 'example-token'})

    def test_lazy_login_per_host_and_bearer_headers(self):
        session = Mock()
        session.post.side_effect = [response({'token': 'v2-token'}), response({'token': 'v1-token'})]
        session.get.return_value = response({'notes': [], 'count': 0})
        client = OpenReviewClient(session=session, credentials={'username': 'user', 'password': 'test-secret'})
        session.post.assert_not_called()
        client.request(API_URLS[0], {})
        client.request(API_URLS[0], {})
        client.request(API_URLS[1], {})
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(session.post.call_args_list[0].kwargs['json']['id'], 'user')
        self.assertEqual(session.get.call_args_list[0].kwargs['headers'], {'Authorization': 'Bearer v2-token'})
        self.assertEqual(session.get.call_args_list[-1].kwargs['headers'], {'Authorization': 'Bearer v1-token'})
        self.assertFalse(session.post.call_args.kwargs['allow_redirects'])
        with self.assertRaises(ValueError):
            client.auth_headers('https://untrusted.example')
        client.token_times[API_URLS[0]] -= 3600
        session.post.side_effect = [response({'token': 'renewed-token'})]
        self.assertEqual(client.auth_headers(API_URLS[0]), {'Authorization': 'Bearer renewed-token'})

    def test_login_errors_and_mfa_never_print_secret_response(self):
        for result in (response({'message': 'test-secret'}, status=403),
                       response({'mfaPending': True, 'mfaPendingToken': 'test-secret'}),
                       response({'message': 'test-secret'})):
            session = Mock()
            session.post.return_value = result
            client = OpenReviewClient(session=session, credentials={'username': 'user', 'password': 'test-secret'})
            with self.assertRaises(RuntimeError) as caught:
                client.request(API_URLS[0], {})
            self.assertNotIn('test-secret', str(caught.exception))
            session.get.assert_not_called()

    def test_dry_run_never_loads_credentials(self):
        with patch('openreview.fetch_openreview_papers.load_credentials') as credentials:
            self.assertEqual(main(['--conferences', 'ICLR', '--years', '2026', '--dry_run']), 0)
            credentials.assert_not_called()

    def test_transient_login_throttling_retries(self):
        session = Mock()
        session.post.side_effect = [response(status=429, headers={'Retry-After': '1'}),
                                   response({'token': 'example-token'})]
        client = OpenReviewClient(session=session, credentials={'username': 'user', 'password': 'test-secret'})
        with patch('openreview.fetching.time.sleep') as sleep:
            self.assertEqual(client.auth_headers(API_URLS[0]), {'Authorization': 'Bearer example-token'})
            sleep.assert_called_once_with(1)


if __name__ == '__main__':
    unittest.main()

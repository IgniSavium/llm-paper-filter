"""Regression checks for OAI query splitting and persistent scheduler recovery.

Run with: python -m unittest discover -s tests -p 'test_fetch_recovery.py'
"""
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from daily_arxiv.fetching import oai
from daily_arxiv.scheduling import scheduler


def instant(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def state():
    return {'timezone': 'Asia/Shanghai', 'at': '07:00',
            'last_completed_date': '2026-10-11', 'attempts': 1}


class FetchRecoveryTests(unittest.TestCase):
    def test_transport_failure_splits_categories_then_dates_and_merges(self):
        def query(start, until, categories, **kwargs):
            if len(categories) > 1 or start != until:
                raise oai.OaiTransportError('timeout')
            row = {'url': 'https://arxiv.org/abs/1v1', 'oai_datestamp': start}
            return oai.Harvest([row], '2026-10-15', 1, ['deleted'])
        with patch.object(oai, '_fetch_metadata_updates_once', side_effect=query):
            result = oai.fetch_metadata_updates('2026-10-12', '2026-10-13', ('cs.AI', 'cs.LG'))
        self.assertEqual(result.papers, [{'url': 'https://arxiv.org/abs/1v1', 'oai_datestamp': '2026-10-13'}])
        self.assertEqual(result.record_count, 4)
        self.assertEqual(result.deleted_identifiers, ['deleted'])

    def test_invalid_protocol_does_not_fall_back_and_single_day_failure_propagates(self):
        for failure in (RuntimeError('Malformed OAI XML'), oai.OaiTransportError('timeout')):
            with patch.object(oai, '_fetch_metadata_updates_once', side_effect=failure) as query:
                with self.assertRaises(RuntimeError):
                    oai.fetch_metadata_updates('2026-10-13', '2026-10-13', ('cs.AI',))
                self.assertEqual(query.call_count, 1)

    def test_repeated_failures_keep_retrying_with_a_four_hour_cap(self):
        value = state()
        for attempt, hours in ((1, 1), (2, 1), (3, 2), (4, 4), (20, 4)):
            value['attempts'] = attempt
            now = instant('2026-10-12T02:00:00')
            scheduler.record_outcome(value, now, now, '2026-10-11', 1)
            self.assertEqual(value['status'], 'retry_wait')
            self.assertEqual(datetime.fromisoformat(value['next_run_at']), now + timedelta(hours=hours))
        value['attempts'] = 20
        now = instant('2026-10-12T22:30:00')
        scheduler.record_outcome(value, now, now, '2026-10-11', 1)
        self.assertEqual(value['next_run_at'], '2026-10-12T23:00:00+00:00')


if __name__ == '__main__':
    unittest.main()

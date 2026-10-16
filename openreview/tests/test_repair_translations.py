"""Saved translations are repaired without changing evaluations or losing originals."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from daily_arxiv.fetching.snapshots import paper_info_digest
from daily_arxiv.filtering.pipeline import save_reports, write_json
from daily_arxiv.filtering.prompts import normalize_evaluation, CATEGORIES, PROMPT_SHA256
from openreview.repair_translations import main
from openreview.process_snapshots import reports_complete


class RepairTests(unittest.TestCase):
    def fixture(self, root):
        output = Path(root) / 'gpt-6-luna_runs/ICLR_2026'
        paper = {'title': 'Features', 'abstract': 'We identify $x$ in the model.',
                 'url': 'https://openreview.net/forum?id=example', 'openreview_id': 'example',
                 'eval': normalize_evaluation({'scores': {c: 4 for c in CATEGORIES}, 'reason': 'Relevant.'})}
        translated = dict(paper, zh_title='特征', zh_abstract='我们识别模型中的 $x$。',
                          eval=dict(paper['eval'], zh_reason='相关。'))
        digest = paper_info_digest([paper])
        save_reports([paper], [translated], '2026-01-01', 4, output, input_digest=digest,
                     source_metadata={'source': 'openreview', 'conference': 'ICLR',
                                      'conference_year': 2026})
        marker = {'model': 'gpt-6-luna', 'threshold': 4, 'reasoning_effort': 'medium',
                  'paper_info_sha256': digest, 'filter_prompt_sha256': PROMPT_SHA256}
        write_json(output / '.queue_complete.json', marker)
        path = output / 'hits_zh/hits_zh_2026-01-01.json'
        payload = json.loads(path.read_text())
        payload['hits_zh'][0]['zh_abstract'] = '错误公式 $y$。'
        write_json(path, payload)
        return output, path, translated

    def test_dry_run_never_loads_inference_or_mutates_results(self):
        with tempfile.TemporaryDirectory() as root:
            output, path, _ = self.fixture(root)
            before = path.read_bytes()
            with patch('openreview.repair_translations.CodexClient') as client:
                self.assertEqual(main(['--runs_dir', root, '--dry_run']), 0)
                client.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_repair_backs_up_original_and_preserves_all_english_and_scores(self):
        with tempfile.TemporaryDirectory() as root:
            output, path, translated = self.fixture(root)
            before = path.read_bytes()
            original_evaluated = json.loads((output / 'all_papers/all_papers_2026-01-01.json').read_text())['all_papers_evaluated']
            repair = Path(root) / 'repair'
            with patch('openreview.repair_translations.CodexClient') as client, \
                    patch('openreview.repair_translations.process_batches', return_value=[copy.deepcopy(translated)]) as inference:
                self.assertEqual(main(['--runs_dir', root, '--repair_dir', str(repair)]), 0)
                client.return_value.check_login.assert_called_once()
                self.assertTrue(inference.call_args.kwargs['translation'])
            self.assertEqual(json.loads(path.read_text())['hits_zh'], [translated])
            self.assertEqual((repair / 'backup/ICLR_2026/hits_zh/hits_zh_2026-01-01.json').read_bytes(), before)
            self.assertEqual(json.loads((output / 'all_papers/all_papers_2026-01-01.json').read_text())['all_papers_evaluated'], original_evaluated)
            marker = json.loads((output / '.queue_complete.json').read_text())
            self.assertTrue(reports_complete(output, '2026-01-01', marker))

    def test_inference_failure_leaves_all_reports_intact(self):
        with tempfile.TemporaryDirectory() as root:
            output, _, _ = self.fixture(root)
            before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
            with patch('openreview.repair_translations.CodexClient'), \
                    patch('openreview.repair_translations.process_batches', side_effect=RuntimeError('retry later')):
                with self.assertRaisesRegex(RuntimeError, 'retry later'):
                    main(['--runs_dir', root, '--repair_dir', str(Path(root) / 'repair')])
            self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))


if __name__ == '__main__':
    unittest.main()

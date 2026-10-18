"""Report transactions, empty results and preservation of historical decisions."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import Mock, patch

from daily_arxiv.cli import runner
from daily_arxiv.fetching.oai import Harvest
from daily_arxiv.fetching.snapshots import paper_info_digest, save_snapshot
from daily_arxiv.fetching.updates import discover_updates, pending_batches
from daily_arxiv.filtering import codex
from daily_arxiv.filtering.pipeline import save_reports, write_json
from daily_arxiv.reporting.artifacts import load_completed_report, manifest_path
from daily_arxiv.reporting.site import export_site, publish_site
from daily_arxiv.filtering.prompts import normalize_evaluation


DAY = '2026-10-08'


def paper(title='Original', score=5):
    return dict(title=title, abstract='A study of model features.', authors=['Author'],
                url='https://arxiv.org/abs/2610.00001v1', date=DAY + ' 00:00:00 UTC',
                oai_datestamp=DAY, status='v1 (New)',
                eval=dict(scores=dict(MechInterp=score, CompGen=1, Reasoning=1, Systems=1), reason='A contribution.'))


def translated(row):
    return dict(row, zh_title='中文标题', zh_abstract='研究模型特征。',
                eval=dict(row['eval'], zh_reason='一项贡献。'))


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / 'runs/old_runs'

    def save(self, row=None):
        row = row or paper()
        save_reports([row], [translated(row)] if max(row['eval']['scores'].values()) >= 4 else [],
                     DAY, 4, self.model, source_metadata={'date_basis': 'metadata_datestamp'})

    def test_renderer_failure_preserves_every_previous_artifact(self):
        self.save()
        before = {path: path.read_bytes() for path in self.model.rglob('*') if path.is_file()}
        with patch('daily_arxiv.filtering.pipeline.generate_mobile_html_report', side_effect=RuntimeError('render failed')):
            with self.assertRaisesRegex(RuntimeError, 'render failed'):
                self.save(paper('New title'))
        self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))
        load_completed_report(self.model, DAY)
        self.assertFalse(list(self.model.glob('.report-stage-*')))

    def test_publisher_rejects_modified_or_partially_committed_generation(self):
        self.save()
        html = self.model / f'html_reports/report_{DAY}.html'
        html.write_text('<html><body>old content</body></html>')
        with self.assertRaisesRegex(ValueError, 'Incomplete or modified'):
            export_site(self.model, self.root / 'site', first_date=DAY, through_date=DAY)
        write_json(manifest_path(self.model, DAY), {'schema_version': 1, 'target_date': DAY, 'state': 'committing'})
        with self.assertRaisesRegex(ValueError, 'completion manifest'):
            load_completed_report(self.model, DAY)

    def test_interruption_during_file_replacement_blocks_publication_until_recovery(self):
        self.save()
        replace = Path.replace
        def interrupted(source, target):
            if Path(target) == self.model / f'hits/hits_{DAY}.json':
                raise OSError('simulated interrupted commit')
            return replace(source, target)
        with patch.object(Path, 'replace', interrupted):
            with self.assertRaisesRegex(OSError, 'interrupted commit'):
                self.save(paper('Changed title'))
        with self.assertRaisesRegex(ValueError, 'completion manifest'):
            export_site(self.model, self.root / 'site', first_date=DAY, through_date=DAY)
        self.save(paper('Changed title'))
        self.assertEqual(load_completed_report(self.model, DAY)['all_papers']['all_papers_evaluated'][0]['title'], 'Changed title')

    def test_legacy_json_is_cross_checked_and_html_is_rebuilt_without_inference(self):
        self.save()
        manifest_path(self.model, DAY).unlink()
        (self.model / f'html_reports/report_{DAY}.html').write_text('<html><body>stale</body></html>')
        with patch.object(codex, 'CodexClient') as client:
            export_site(self.model, self.root / 'site', first_date=DAY, through_date=DAY)
            client.assert_not_called()
        published = (self.root / f'site/reports/report_{DAY}.html').read_text()
        self.assertIn('Original', published)
        self.assertNotIn('<body>stale', published)
        path = self.model / f'hits_zh/hits_zh_{DAY}.json'
        payload = json.loads(path.read_text())
        payload['hits_zh'][0]['title'] = 'Wrong generation'
        write_json(path, payload)
        with self.assertRaisesRegex(ValueError, 'Translations do not match'):
            load_completed_report(self.model, DAY)

    def test_zero_hit_batch_is_complete_without_html(self):
        row = paper(score=1)
        self.save(row)
        inputs = self.root / 'inputs'
        save_snapshot([row], DAY, inputs)
        self.assertFalse((self.model / f'html_reports/report_{DAY}.html').exists())
        self.assertEqual(list(pending_batches(inputs, self.model, 5, DAY)), [])
        rows = export_site(self.model, self.root / 'site', first_date=DAY, through_date=DAY)
        self.assertEqual(rows[0]['status'], 'no_matches')

    def test_empty_site_can_publish_to_a_local_remote_and_repeat_idempotently(self):
        self.save(paper(score=1))
        remote = self.root / 'remote.git'
        project = self.root / 'project'
        subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
        subprocess.run(['git', 'init', str(project)], check=True, capture_output=True)
        subprocess.run(['git', 'remote', 'add', 'origin', str(remote)], cwd=project, check=True, capture_output=True)
        with patch('daily_arxiv.reporting.site.PROJECT_DIR', project), \
                patch('daily_arxiv.reporting.site.RUNS_DIR', self.root / 'publisher'):
            self.assertEqual(publish_site(DAY, self.model)['status'], 'pushed')
            self.assertEqual(publish_site(DAY, self.model)['status'], 'unchanged')

    def test_empty_single_date_refresh_removes_stale_html_and_updates_archive(self):
        self.model = self.root / 'runs/gpt-6-luna_runs'
        self.save()
        def execute(command, **kwargs):
            codex.main(command[2:])
            return Mock(returncode=0)
        with patch.object(runner, 'get_papers', return_value=([], DAY)), \
                patch.object(runner.subprocess, 'run', side_effect=execute), patch.object(codex, 'CodexClient') as client:
            self.assertEqual(runner.main(['--date', DAY, '--runs_dir', str(self.model.parent),
                                          '--paper_info_dir', str(self.root / 'inputs')]), 0)
            client.assert_not_called()
        self.assertEqual(load_completed_report(self.model, DAY)['all_papers']['all_papers_evaluated'], [])
        self.assertFalse((self.model / f'html_reports/report_{DAY}.html').exists())
        rows = export_site(self.model, self.root / 'site', first_date=DAY, through_date=DAY)
        self.assertEqual(rows[0]['status'], 'no_updates')

    def test_changed_model_prompt_and_threshold_do_not_rescore_completed_date(self):
        self.save()
        old = load_completed_report(self.model, DAY)
        path = save_snapshot([paper()], DAY, self.root / 'inputs')
        new = self.model.parent / 'new_runs'
        with patch.object(codex, 'CodexClient') as client, \
                patch('daily_arxiv.filtering.pipeline.PROMPT_SHA256', 'new-prompt'):
            codex.main(['--input_json', str(path), '--date', DAY, '--threshold', '5',
                        '--reasoning_effort', 'high', '--save_dir', str(new), '--codex_bin', '/missing'])
            client.assert_not_called()
        self.assertEqual(load_completed_report(new, DAY), old)

    def test_same_version_metadata_changes_are_discovered_and_keep_existing_scores(self):
        self.save()
        inputs = self.root / 'inputs'
        save_snapshot([paper()], DAY, inputs)
        changed = copy.deepcopy(paper())
        changed.update(journal_ref='Journal 2026', oai_datestamp='2026-10-09')
        with patch('daily_arxiv.fetching.updates.fetch_metadata_updates',
                   return_value=Harvest([changed], '2026-10-10', 1, [])):
            discover_updates(inputs, now=datetime(2026, 10, 10, tzinfo=timezone.utc), through_date='2026-10-09')
        source = next((inputs / 'updates').glob('*/papers_*.json'))
        output = self.model / 'updates' / source.parent.name
        with patch.object(codex, 'CodexClient') as client:
            codex.main(['--input_json', str(source), '--save_dir', str(output), '--codex_bin', '/missing'])
            client.assert_not_called()
        result = load_completed_report(output, '2026-10-09')['hits_zh']['hits_zh'][0]
        self.assertEqual(result['journal_ref'], 'Journal 2026')
        self.assertEqual(result['eval'], translated(paper())['eval'])

    def test_only_new_content_is_scored_when_an_existing_date_gets_more_papers(self):
        self.save()
        added = paper('New paper', score=4)
        added['url'] = 'https://arxiv.org/abs/2610.00002v1'
        source = save_snapshot([paper(), added], DAY, self.root / 'inputs')
        calls = []
        def request(instruction, records, translation=False):
            calls.append((records, translation))
            if translation:
                return {r['id']: dict(zh_title='新论文', zh_abstract='新摘要。', zh_reason='新贡献。') for r in records}
            return {r['id']: normalize_evaluation(dict(scores=added['eval']['scores'], reason='New contribution.')) for r in records}
        with patch.object(codex, 'CodexClient') as factory:
            factory.return_value.model = 'new-model'
            factory.return_value.effort = 'high'
            factory.return_value.request.side_effect = request
            codex.main(['--input_json', str(source), '--save_dir', str(self.model), '--threshold', '5'])
        self.assertEqual([(len(records), stage) for records, stage in calls], [(1, False), (1, True)])
        self.assertEqual(calls[0][0][0]['title'], 'New paper')
        evaluated = load_completed_report(self.model, DAY)['all_papers']['all_papers_evaluated']
        self.assertEqual(evaluated[0]['eval'], paper()['eval'])
        self.assertEqual(load_completed_report(self.model, DAY)['all_papers']['metadata']['threshold_used'], 4)

    def test_conference_statistics_keep_scores_from_a_historical_prompt(self):
        from openreview.analyze_direction_counts import collect
        output = self.root / 'conference_runs/ICLR_2026'
        save_reports([paper()], [translated(paper())], '2026-01-01', 4, output,
                     source_metadata=dict(source='openreview', conference='ICLR', conference_year=2026,
                                          filter_prompt_sha256='historical-prompt'))
        rows, _ = collect(output.parent)
        self.assertEqual(rows[0]['counts']['mech'], 1)
        self.assertEqual(rows[0]['filter_prompt_sha256'], 'historical-prompt')


if __name__ == '__main__':
    unittest.main()

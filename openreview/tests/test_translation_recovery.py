"""Regressions for formula-marker copying and omitted formula-adjacent separators."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from daily_arxiv.filtering.codex import CodexClient
from daily_arxiv.reporting.html import math_markup, split_sentences
from daily_arxiv.filtering.pipeline import (
    TRANSLATION_PROMPT, TranslationContentError, assemble_translation, paper_prompt,
    validate_translation_math, split_translation_text, process_batches,
    validate_translation_content, save_reports,
)


class TranslationRecoveryTests(unittest.TestCase):
    def test_inline_math_spaces_symbols_digits_and_adjacent_formulas(self):
        source = r'Atoms $ R^2 $; acceptance $>$90%; speedup $3.88$$\times$ on SVD, $2.43$$\times$ on CogVideoX.'
        parts, formulas = split_translation_text(source)
        self.assertEqual(formulas, [r'$ R^2 $', '$>$', '$3.88$', r'$\times$', '$2.43$', r'$\times$'])
        self.assertEqual(parts, ['Atoms ', '; acceptance ', '90%; speedup ', '', ' on SVD, ', '', ' on CogVideoX.'])
        result = assemble_translation(source, ['原子 ', '；接受率 ', '90%；加速 ', '', '（SVD），', '', '（CogVideoX）。'])
        validate_translation_content(source, result)
        markup = math_markup(source)
        self.assertEqual(markup.count('class="math-source"'), len(formulas))
        self.assertIn('data-tex="&gt;"', markup)
        self.assertIn('data-tex="3.88"', markup)
        self.assertEqual(''.join(split_sentences(source, 'en')), source)

    def test_escaped_dollar_inside_formula_and_malformed_closing_command(self):
        source = r'Web$\\text{\\$}$hell and large-$ \gamma\$ limit.'
        parts, formulas = split_translation_text(source)
        self.assertEqual(parts, ['Web', 'hell and large-', ' limit.'])
        self.assertEqual(formulas, [r'$\\text{\\$}$', r'$ \gamma$'])
        result = assemble_translation(source, ['Web', 'hell，以及大 ', ' 极限。'])
        validate_translation_content(source, result)

    def test_formula_reordering_is_rejected_even_with_identical_multiset(self):
        with self.assertRaisesRegex(ValueError, 'reordered'):
            validate_translation_math('SVD $3.88$; CogVideoX $2.43$.', 'SVD $2.43$；CogVideoX $3.88$。')

    def test_english_leakage_and_report_validation_happen_before_writes(self):
        source = 'We introduce a new method that improves the quality of predictions in a variety of settings.'
        with self.assertRaisesRegex(ValueError, 'English prose'):
            validate_translation_content(source, '我们提出方法。' + source)
        validate_translation_content('We use Large Language Models (LLMs).', '我们使用 Large Language Models (LLMs)。')
        with self.assertRaisesRegex(ValueError, 'protocol marker'):
            validate_translation_content('A study.', '这是一项研究。zh_reason_parts cannot additional invalid?')
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / 'hits/hits_2026-01-01.json'
            destination.parent.mkdir()
            destination.write_text('previous report')
            bad = {'title': 'Test', 'abstract': source, 'eval': {'reason': 'Useful.', 'zh_reason': '有用。'},
                   'zh_title': '测试', 'zh_abstract': source}
            with self.assertRaisesRegex(ValueError, 'English prose'):
                save_reports([], [bad], '2026-01-01', 4, root)
            self.assertEqual(destination.read_text(), 'previous report')

    def test_command_math_touching_digits_does_not_pair_with_later_dollars(self):
        source = r'At $\sim$11M, width changes 0.47$\to$0.34 and $C\approx0.5\,d_{\mathrm{clean}}$.'
        parts, formulas = split_translation_text(source)
        self.assertEqual(formulas, [r'$\sim$', r'$\to$', r'$C\approx0.5\,d_{\mathrm{clean}}$'])
        self.assertEqual(parts, ['At ', '11M, width changes 0.47', '0.34 and ', '.'])
        translated = assemble_translation(source, ['在 ', '11M 下，宽度从 0.47', '0.34，且 ', '。'])
        validate_translation_math(source, translated)

    def test_leading_and_trailing_spaces_in_command_math_remain_exact(self):
        source = r'We separate $ \mathbf{U_L} $ and $ \mathbf{U_R} $, improving by $ \mathbf{40\%} $.'
        parts, formulas = split_translation_text(source)
        self.assertEqual(formulas, [r'$ \mathbf{U_L} $', r'$ \mathbf{U_R} $', r'$ \mathbf{40\%} $'])
        self.assertEqual(parts, ['We separate ', ' and ', ', improving by ', '.'])
        translated = assemble_translation(source, ['我们分离 ', ' 与 ', '，提高 ', '。'])
        validate_translation_math(source, translated)
        with self.assertRaises(ValueError):
            validate_translation_math(source, translated.replace('40', '50'))

    def test_short_prose_slot_is_required_and_omission_feedback_is_specific(self):
        records = [{'id': '15146', 'title': 'Decoding',
                    'abstract': 'The goal of $L$-step decoding is to accelerate generation.', 'reason': 'Useful.'}]
        with self.assertRaisesRegex(TranslationContentError, "index 0: 'The goal of '"):
            assemble_translation(records[0]['abstract'], ['', '步解码的目标是加速生成。'])
        with patch('daily_arxiv.filtering.codex.shutil.which', return_value='/fake/codex'):
            client = CodexClient(retries=1)

        def execute(command, **kwargs):
            schema = json.loads(Path(command[command.index('--output-schema') + 1]).read_text())
            field = schema['properties']['results']['items']['properties']['zh_abstract_parts']
            self.assertEqual(field['items']['minLength'], 1)
            Path(command[command.index('--output-last-message') + 1]).write_text(json.dumps({'results': [{
                'id': '15146', 'zh_title_parts': ['解码'],
                'zh_abstract_parts': ['目标是使 ', '步解码加速生成。'], 'zh_reason_parts': ['有用。']}]}))
            return subprocess.CompletedProcess(command, 0, '', '')

        with patch('daily_arxiv.filtering.codex.subprocess.run', side_effect=execute):
            translated = client.request(TRANSLATION_PROMPT, records, True)['15146']
        validate_translation_math(records[0]['abstract'], translated['zh_abstract'])

    def test_trailing_space_in_formula_is_protected_from_translation(self):
        source = r'We use $\textbf{Multi-Agent Task Graph (MATG)} $and enforce $x$.'
        parts, formulas = split_translation_text(source)
        self.assertEqual(parts, ['We use ', 'and enforce ', '.'])
        self.assertEqual(formulas, [r'$\textbf{Multi-Agent Task Graph (MATG)} $', '$x$'])
        translated = assemble_translation(source, ['我们使用 ', '并约束 ', '。'])
        self.assertIn(formulas[0], translated)
        validate_translation_math(source, translated)
        with self.assertRaises(ValueError):
            validate_translation_math(source, translated.replace('} $', '}$'))
        # Keep the existing currency boundary and escaped-dollar behavior.
        validate_translation_math('a $1 budget and a $1,000 budget', '预算 1 美元与 1,000 美元')

    def test_failed_singleton_does_not_starve_later_translation_batches(self):
        class Client:
            model, effort = 'test', 'medium'

            def __init__(self):
                self.calls = []

            def request(self, instruction, records, translation):
                self.calls.extend(r['id'] for r in records)
                if records[0]['id'] == '0':
                    raise TranslationContentError('bad formula in paper 0')
                return {r['id']: {'zh_title': '标题', 'zh_abstract': '摘要', 'zh_reason': '原因'} for r in records}

        papers = [{'title': str(i), 'abstract': 'Abstract', 'eval': {'reason': 'Reason'}} for i in range(3)]
        client = Client()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(TranslationContentError, 'all other batches were processed'):
                process_batches(papers, client, root, batch_size=1, translation=True, workers=1)
            self.assertEqual(client.calls, ['0', '1', '2'])
            self.assertEqual(len(list((Path(root) / 'translation').glob('*.json'))), 2)
            client.calls.clear()
            with self.assertRaises(TranslationContentError):
                process_batches(papers, client, root, batch_size=1, translation=True, workers=1)
            self.assertEqual(client.calls, ['0'])

    def test_singleton_prompt_contains_only_prose_fragments(self):
        records = [{'id': '20', 'title': 'Tokenization',
                    'abstract': 'We outperform baselines at the $100$ million parameter scale.',
                    'reason': 'Useful.'}]
        prompt = paper_prompt(TRANSLATION_PROMPT, records, True, fragments_only=True)
        self.assertNotIn('⟪FORMULA_', prompt)
        self.assertNotIn('$100$', prompt)
        submitted = json.loads(prompt.split('Paper data (JSON):\n', 1)[1])
        self.assertEqual(submitted[0]['abstract_parts'], ['We outperform baselines at the ', ' million parameter scale.'])
        self.assertNotIn('abstract', submitted[0])
        with patch('daily_arxiv.filtering.codex.shutil.which', return_value='/fake/codex'):
            client = CodexClient(retries=1)

        def execute(command, **kwargs):
            self.assertNotIn('⟪FORMULA_', kwargs['input'])
            Path(command[command.index('--output-last-message') + 1]).write_text(json.dumps({'results': [{
                'id': '20', 'zh_title_parts': ['分词'],
                'zh_abstract_parts': ['我们在 ', ' 百万参数规模上超过基线。'], 'zh_reason_parts': ['有用。']}]}))
            return subprocess.CompletedProcess(command, 0, '', '')

        with patch('daily_arxiv.filtering.codex.subprocess.run', side_effect=execute):
            result = client.request(TRANSLATION_PROMPT, records, True)['20']
        self.assertIn('$100$', result['zh_abstract'])
        validate_translation_math(records[0]['abstract'], result['zh_abstract'])

    def test_empty_punctuation_fragments_restore_source_without_omitting_prose(self):
        source = 'Speedup $1.16\\times$-$2.16\\times$; savings $14.3$%-$25.3$%.'
        translated = assemble_translation(source, ['加速 ', '', '；节能 ', '', '%。'])
        self.assertIn('$1.16\\times$-$2.16\\times$', translated)
        self.assertIn('$14.3$%-$25.3$', translated)
        validate_translation_math(source, translated)
        with self.assertRaisesRegex(TranslationContentError, 'omitted'):
            assemble_translation('Hello $x$ world.', ['', ' 世界。'])
        with self.assertRaisesRegex(TranslationContentError, 'context marker'):
            assemble_translation('Hello $x$ world.', ['你好⟪FORMULA_0⟫', ' 世界。'])


if __name__ == '__main__':
    unittest.main()

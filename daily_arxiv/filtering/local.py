"""Local vLLM transport for the same research filtering pipeline as GPT."""
import argparse
import json
from pathlib import Path
from daily_arxiv.paths import RUNS_DIR
import sys
import tempfile

from daily_arxiv.filtering.prompts import SYSTEM_PROMPT, evaluation_score
from daily_arxiv.filtering.pipeline import (
    TRANSLATION_PROMPT, DATA_BOUNDARY, TranslationContentError,
    result_schema, validate_results, decode_translation_parts, paper_prompt,
    process_batches, load_filter_input, run_filter, save_reports,
)
from daily_arxiv.reporting.html import generate_mobile_html_report
from daily_arxiv.fetching.snapshots import DEFAULT_PAPER_INFO_DIR, paper_info_digest, resolve_date


def parse_json_response(text):
    """Allow an outer Markdown fence; do not repair or reinterpret invalid JSON."""
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        if '\n' not in text:
            raise ValueError('Invalid JSON fence')
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return json.loads(text)


class LocalClient:
    """Keep GPU inference separate from shared scoring, validation, and caching."""
    def __init__(self, llm, tokenizer, model, sampling_params=None, max_model_len=16384,
                 evaluation_tokens=2048, translation_tokens=8192, retries=3, structured_output_params=None):
        if sampling_params is None:
            from vllm import SamplingParams
            sampling_params = SamplingParams
        self.llm = llm
        self.tokenizer = tokenizer
        self.model = model
        self.sampling_params = sampling_params
        self.max_model_len = max_model_len
        self.evaluation_tokens = evaluation_tokens
        self.translation_tokens = translation_tokens
        self.retries = retries
        self.structured_output_params = structured_output_params
        self.effort = (f'temperature=0;context={max_model_len};'
                       f'evaluation_tokens={evaluation_tokens};translation_tokens={translation_tokens};'
                       f'structured_json={structured_output_params is not None}')

    def request(self, instruction, records, translation=False):
        prompt = paper_prompt(instruction, records, translation)
        # The same instructions and data boundary are used by both transports.
        schema = result_schema(translation, segmented=translation)
        messages = [{'role': 'system', 'content': instruction + '\n\n' + DATA_BOUNDARY
                     + '\nResponse JSON schema:\n' + json.dumps(schema)},
                    {'role': 'user', 'content': prompt.split('Paper data (JSON):\n', 1)[1]}]
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        budget = self.translation_tokens if translation else self.evaluation_tokens
        # Split oversized requests rather than silently truncating abstracts.
        if len(tokens) + budget > self.max_model_len and len(records) > 1:
            middle = len(records) // 2
            first = self.request(instruction, records[:middle], translation)
            try:
                second = self.request(instruction, records[middle:], translation)
            except TranslationContentError as exc:
                raise TranslationContentError(str(exc), {**first, **exc.partial_results}) from exc
            return {**first, **second}
        budget = min(budget, self.max_model_len - len(tokens))
        if budget < 64:
            raise RuntimeError(f'Paper {records[0]["id"]} exceeds the local context budget; increase --max_model_len')
        error = None
        for attempt in range(self.retries):
            params = {'temperature': 0.0, 'max_tokens': budget}
            if self.structured_output_params is not None:
                params['structured_outputs'] = self.structured_output_params(json=schema)
            outputs = self.llm.generate(
                [{'prompt_token_ids': tokens}],
                sampling_params=self.sampling_params(**params), use_tqdm=False)
            try:
                if len(outputs) != 1 or not outputs[0].outputs:
                    raise ValueError('Local model returned an incomplete response')
                response = outputs[0].outputs[0]
                if getattr(response, 'finish_reason', None) == 'length':
                    raise ValueError('Local response reached its token limit')
                payload = parse_json_response(response.text)
                return (decode_translation_parts(payload, records) if translation else
                        validate_results(payload, [record['id'] for record in records]))
            except ValueError as exc:
                error = exc
                if translation:
                    error = exc if isinstance(exc, TranslationContentError) else TranslationContentError(str(exc))
                    if len(records) > 1 or attempt + 1 == self.retries:
                        raise error
                if attempt + 1 < self.retries:
                    print(f'Local response invalid ({attempt + 1}/{self.retries}): {exc}', flush=True)
                    # Preserve the system/user role sequence required by Gemma
                    # and other strict chat templates while adding retry feedback.
                    messages[0]['content'] += (f'\nPrevious response was invalid: {exc}. '
                                               'Return the complete JSON with all required fields and original IDs.')
                    tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                    budget = min(budget, self.max_model_len - len(tokens))
                    if budget < 64:
                        raise RuntimeError('Local validation retry exceeds context budget') from exc
        raise RuntimeError(f'Local evaluation failed after {self.retries} attempts: {error}') from error


def create_client(args):
    # Import GPU dependencies only when there is actual model work to do.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp_size,
              dtype=args.dtype, quantization=args.quantization, gpu_memory_utilization=args.gpu_util,
              enable_prefix_caching=True, max_num_seqs=args.max_num_seqs,
              max_model_len=args.max_model_len)
    client = LocalClient(llm, tokenizer, str(Path(args.model_path).resolve()), SamplingParams,
                         args.max_model_len, args.evaluation_tokens, args.translation_tokens,
                         args.max_retries, structured_output_params=StructuredOutputsParams)
    client.effort += f';dtype={args.dtype};quantization={args.quantization}'
    return client


def filter_papers(papers, llm, tokenizer, threshold=4):
    """Compatibility helper; the CLI uses persistent per-model caches."""
    client = LocalClient(llm, tokenizer, getattr(tokenizer, 'name_or_path', 'local'))
    with tempfile.TemporaryDirectory(prefix='arxiv-local-evaluation-') as cache:
        evaluated = process_batches(papers, client, cache)
    return [p for p in evaluated if evaluation_score(p['eval']) >= threshold], evaluated


def translate_hits(hits, llm, tokenizer):
    client = LocalClient(llm, tokenizer, getattr(tokenizer, 'name_or_path', 'local'))
    with tempfile.TemporaryDirectory(prefix='arxiv-local-translation-') as cache:
        return process_batches(hits, client, cache, batch_size=10, translation=True)


def main(argv=None, default_model_path='./weights/Llama-3.3-70B-Instruct', default_quantization=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', help='Metadata date YYYY-MM-DD; default: yesterday UTC')
    parser.add_argument('--threshold', type=int, choices=range(1, 6), default=4)
    parser.add_argument('--model_path', default=default_model_path)
    parser.add_argument('--tp_size', type=int, default=4)
    parser.add_argument('--gpu_util', type=float, default=0.90)
    parser.add_argument('--max_num_seqs', type=int, default=128)
    parser.add_argument('--max_model_len', type=int, default=16384)
    parser.add_argument('--dtype', default='auto', choices=('auto', 'float16', 'bfloat16', 'float32'))
    parser.add_argument('--quantization', default=default_quantization, help='vLLM quantization override; otherwise use model configuration')
    parser.add_argument('--evaluation_tokens', type=int, default=2048)
    parser.add_argument('--translation_tokens', type=int, default=8192)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--translation_batch_size', type=int, default=10)
    parser.add_argument('--max_retries', type=int, default=3)
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--cache_dir', type=Path)
    parser.add_argument('--paper_info_dir', type=Path, default=DEFAULT_PAPER_INFO_DIR)
    parser.add_argument('--input_json', type=Path)
    parser.add_argument('--max_results', type=int, default=2000)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    for option in ('tp_size', 'max_num_seqs', 'max_model_len', 'evaluation_tokens',
                   'translation_tokens', 'batch_size', 'translation_batch_size',
                   'max_retries', 'max_results', 'limit'):
        value = getattr(args, option)
        if value is not None and value < 1:
            parser.error(f'--{option} must be positive')
    if not 0 < args.gpu_util <= 1:
        parser.error('--gpu_util must be in (0, 1]')
    if args.date is not None:
        resolve_date(args.date)
    model_name = Path(args.model_path).name
    args.save_dir = args.save_dir or RUNS_DIR / f'{model_name}_runs'
    papers, date, metadata = load_filter_input(args)
    digest = paper_info_digest(papers)
    if papers:
        client = create_client(args)
        papers, translated = run_filter(papers, client, args.cache_dir or args.save_dir / '.local_cache',
                                        args.threshold, args.batch_size, args.translation_batch_size,
                                        workers=1)  # Serialize access to the local GPU engine.
    else:
        translated = []
    save_reports(papers, translated, date, args.threshold, args.save_dir,
                 input_digest=digest, source_metadata=metadata)
    print(f'Complete: {len(papers)} papers evaluated, {len(translated)} translated using {model_name}.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)

"""Validate complete report generations without invalidating historical scores."""
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re

from daily_arxiv.fetching.snapshots import paper_info_digest
from daily_arxiv.filtering.prompts import evaluation_score, validate_evaluation


def report_paths(day):
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day):
        raise ValueError('Invalid report date')
    return {folder: f'{folder}/{folder}_{day}.json'
            for folder in ('all_papers', 'hits', 'hits_zh')}


def manifest_path(root, day):
    report_paths(day)
    return Path(root) / '.reports' / f'{day}.json'


def file_digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


@contextmanager
def report_lock(root, day, *, exclusive=False):
    directory = Path(root) / '.reports'
    directory.mkdir(parents=True, exist_ok=True)
    report_paths(day)
    with (directory / f'{day}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def without_translation(paper):
    result = copy.deepcopy({k: v for k, v in paper.items() if not k.startswith('zh_')})
    result['eval'].pop('zh_reason', None)
    return result


def validate_payloads(reports, day):
    """Cross-check actual rows, not just their claimed input checksums."""
    from daily_arxiv.filtering.pipeline import validate_translated_paper
    metadata = reports['all_papers']['metadata']
    fields = ('target_date', 'paper_info_sha256', 'threshold_used',
              'filter_prompt_sha256', 'translation_contract_sha256', 'scoring_mode')
    if metadata.get('target_date') != day:
        raise ValueError('Mismatched report date')
    for report in reports.values():
        if any(report['metadata'].get(key) != metadata.get(key) for key in fields):
            raise ValueError('Report metadata belongs to different generations')
    papers = reports['all_papers']['all_papers_evaluated']
    digest = paper_info_digest(papers)
    if metadata.get('paper_info_sha256') not in (None, digest):
        raise ValueError('Report input checksum mismatch')
    for paper in papers:
        if metadata.get('scoring_mode') == 'per_direction':
            validate_evaluation(paper['eval'])
    expected = sorted((p for p in papers if evaluation_score(p['eval']) >= metadata['threshold_used']),
                      key=lambda p: evaluation_score(p['eval']), reverse=True)
    hits, translated = reports['hits']['hits'], reports['hits_zh']['hits_zh']
    if [without_translation(p) for p in hits] != [without_translation(p) for p in expected]:
        raise ValueError('Hits do not match the evaluated papers')
    if [without_translation(p) for p in translated] != [without_translation(p) for p in hits]:
        raise ValueError('Translations do not match the selected papers')
    for paper in translated:
        validate_translated_paper(paper)


def load_completed_report(root, day):
    """Accept validated legacy JSON, but never trust its unversioned HTML.

    Model/prompt/translation settings are provenance, not expiry dates. A valid
    historical result remains complete after defaults change.
    """
    root = Path(root)
    paths = report_paths(day)
    with report_lock(root, day):
        marker = manifest_path(root, day)
        manifest = json.loads(marker.read_text()) if marker.exists() else None
        if manifest is not None:
            expected = set(paths.values())
            html = f'html_reports/report_{day}.html'
            if manifest.get('has_html'):
                expected.add(html)
            if (manifest.get('schema_version') != 1 or manifest.get('target_date') != day
                    or set(manifest.get('files', {})) != expected):
                raise ValueError('Invalid report completion manifest')
            for relative, digest in manifest['files'].items():
                if file_digest(root / relative) != digest:
                    raise ValueError(f'Incomplete or modified report generation: {day}')
        reports = {key: json.loads((root / relative).read_text(encoding='utf-8'))
                   for key, relative in paths.items()}
        validate_payloads(reports, day)
        if manifest is not None and bool(reports['hits_zh']['hits_zh']) != manifest['has_html']:
            raise ValueError('Report HTML state does not match its translations')
        return reports


def report_candidates(save_dir):
    target = Path(save_dir)
    candidates = [target]
    model_root = next((p for p in (target, *target.parents) if p.name.endswith('_runs')), None)
    if model_root is not None:
        relative = target.relative_to(model_root)
        candidates += [p / relative for p in sorted(model_root.parent.glob('*_runs')) if p != model_root]
        if relative.parts and relative.parts[0] == 'updates':
            candidates += [model_root, *sorted(model_root.parent.glob('*_runs'))]
    return list(dict.fromkeys(candidates))


def completed_date_threshold(save_dir, day, default):
    """Late additions to an existing date keep that date's admission threshold."""
    for candidate in report_candidates(save_dir):
        if not (candidate / report_paths(day)['all_papers']).is_file():
            continue
        try:
            return load_completed_report(candidate, day)['all_papers']['metadata']['threshold_used']
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return default


def find_completed_report(save_dir, day, input_digest):
    """Reuse an unchanged corpus even when the user selects a different model."""
    for candidate in report_candidates(save_dir):
        if not (candidate / report_paths(day)['all_papers']).is_file():
            continue
        try:
            reports = load_completed_report(candidate, day)
            if paper_info_digest(reports['all_papers']['all_papers_evaluated']) == input_digest:
                return candidate, reports
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def content_key(paper):
    return hashlib.sha256(json.dumps([paper['title'], paper['abstract']], ensure_ascii=False).encode()).hexdigest()


def prior_paper_results(save_dir, papers):
    """Keep existing decisions when only metadata or model defaults change."""
    target = Path(save_dir)
    model_root = next((p for p in (target, *target.parents) if p.name.endswith('_runs')), None)
    roots = [target]
    if model_root is not None:
        # Conference corpora stay isolated; daily update batches share daily results.
        relative = target.relative_to(model_root)
        if relative.parts and relative.parts[0] == 'updates':
            relative = Path('.')
        roots = [p / relative for p in sorted(model_root.parent.glob('*_runs'))]
    wanted = {content_key(p) for p in papers}
    found = {}
    for root in roots:
        paths = sorted(root.glob('all_papers/all_papers_*.json'), reverse=True)
        paths += sorted(root.glob('updates/*/all_papers/all_papers_*.json'), reverse=True)
        for path in paths:
            try:
                day = path.stem.removeprefix('all_papers_')
                reports = load_completed_report(path.parent.parent, day)
                translated = {content_key(p): p for p in reports['hits_zh']['hits_zh']}
                for paper in reports['all_papers']['all_papers_evaluated']:
                    key = content_key(paper)
                    if key in wanted and key not in found:
                        found[key] = translated.get(key, paper)
                if wanted <= found.keys():
                    return found
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return found

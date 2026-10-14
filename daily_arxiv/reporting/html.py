"""Self-contained bilingual HTML reader shared by every model backend."""
from html import escape
import base64
import hashlib
from daily_arxiv.paths import KATEX_DIR, TEMPLATE_PATH
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from urllib.parse import urlparse

from daily_arxiv.filtering.prompts import CATEGORIES, PROMPT_SHA256, category_labels, evaluation_score, DEFAULT_FILTER_EXPRESSION


def bookmark_key(paper):
    """Keep an arXiv bookmark stable across protocols and paper revisions."""
    url = paper.get("url", "")
    parsed = urlparse(url)
    if parsed.hostname in ("arxiv.org", "www.arxiv.org", "export.arxiv.org") and parsed.path.startswith("/abs/"):
        return "arxiv:" + re.sub(r"v\d+$", "", parsed.path.removeprefix("/abs/"))
    if parsed.scheme in ("http", "https"):
        return "url:" + parsed.netloc.lower() + parsed.path + ("?" + parsed.query if parsed.query else "")
    return "paper:" + hashlib.sha256((paper.get("title", "") + "\n" + paper.get("abstract", "")).encode()).hexdigest()

MATH_RE = re.compile(r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<![\\$])\$(?!\$|\s*\d[\d,.]*(?:[A-Za-z]{1,3})?\s+[^\W\d_]{2,}(?=[\s,.;:，。；：]))(?:\\.|[^\\$])*?(?<!\\)\$(?![\d$])|\\\(.*?\\\)|\\\[.*?\\\]", re.S)


def math_markup(text):
    parts = []
    start = 0
    for match in MATH_RE.finditer(text):
        parts.append(escape(text[start:match.start()]))
        raw = match.group()
        display = raw.startswith(("$$", r"\["))
        width = 2 if raw.startswith(("$$", r"\(", r"\[")) else 1
        tex = raw[width:-width]
        parts.append(f'<span class="math-source" data-tex="{escape(tex, quote=True)}" data-display="{str(display).lower()}">{escape(raw)}</span>')
        start = match.end()
    parts.append(escape(text[start:]))
    return "".join(parts)


def split_sentences(text, language):
    """Keep every source character, while protecting math and common abbreviations."""
    if not text:
        return []
    protected = []
    def protect(match):
        marker = f"\ue000{len(protected)}\ue001"
        protected.append(match.group())
        return marker
    masked = MATH_RE.sub(protect, text)
    if language == "en":
        masked = re.sub(r"\b(?:e\.g\.|i\.e\.|et al\.|Fig\.|Eq\.|Dr\.|Prof\.|vs\.|etc\.|[A-Z]\.)", protect, masked)
        boundary = r"[.!?]+[\"”’')\]]*(?:\s+|$)"
    else:
        boundary = r"[。！？]+[\"”’）\]]*\s*"
    sentences = []
    start = 0
    for match in re.finditer(boundary, masked):
        sentences.append(masked[start:match.end()])
        start = match.end()
    if start < len(masked):
        sentences.append(masked[start:])
    def restore(sentence):
        return re.sub(r"\ue000(\d+)\ue001", lambda m: protected[int(m.group(1))], sentence)
    return [restore(s) for s in sentences if s.strip()]


def align_sentences(english, chinese):
    """Approximate contiguous alignment, allowing translation splits and merges."""
    en = split_sentences(english, "en")
    zh = split_sentences(chinese, "zh")
    if not en or not zh:
        return en, zh, list(range(len(en))), list(range(len(zh)))
    if len(en) == len(zh):
        return en, zh, list(range(len(en))), list(range(len(zh)))
    def lengths(sentences):
        result = [0]
        for sentence in sentences:
            result.append(result[-1] + max(1, len(MATH_RE.sub("x", sentence).strip())))
        return result
    e, z = lengths(en), lengths(zh)
    n, m = len(en), len(zh)
    scores = {(0, 0): 0.0}
    previous = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if (i, j) not in scores:
                continue
            for a, b in ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)):
                if i + a > n or j + b > m:
                    continue
                el = (e[i + a] - e[i]) / e[-1]
                zl = (z[j + b] - z[j]) / z[-1]
                cost = abs(math.log(el / zl)) + 0.25 * (a + b - 2)
                value = scores[i, j] + cost
                if value < scores.get((i + a, j + b), float("inf")):
                    scores[i + a, j + b] = value
                    previous[i + a, j + b] = (i, j)
    if (n, m) not in previous:
        # Unusual sentence-count ratios: normalized positions still preserve all text.
        size = min(n, m)
        return en, zh, [min(size - 1, i * size // n) for i in range(n)], [min(size - 1, j * size // m) for j in range(m)]
    groups = []
    point = (n, m)
    while point != (0, 0):
        prior = previous[point]
        groups.append((prior, point))
        point = prior
    en_ids, zh_ids = [0] * n, [0] * m
    for group, ((i, j), (end_i, end_j)) in enumerate(reversed(groups)):
        en_ids[i:end_i] = [group] * (end_i - i)
        zh_ids[j:end_j] = [group] * (end_j - j)
    return en, zh, en_ids, zh_ids


def sentence_markup(sentences, ids):
    return "".join(f'<span class="sentence" data-pair="{pair}" tabindex="0">{math_markup(sentence)}</span>' for sentence, pair in zip(sentences, ids))


@lru_cache(maxsize=1)
def math_assets():
    """Inline the pinned engine and fonts so a report remains a single offline file."""
    root = KATEX_DIR
    css = (root / "katex.min.css").read_text(encoding="utf-8")
    def embed(match):
        src = match.group()
        name = re.search(r"url\(([^)]+\.woff2)\)", src).group(1).strip('"\'')
        encoded = base64.b64encode((root / name).read_bytes()).decode("ascii")
        return f'src:url(data:font/woff2;base64,{encoded}) format("woff2")'
    css = re.sub(r"src:[^;{}]+", embed, css)
    license_text = (root / "LICENSE").read_text(encoding="utf-8").replace("*/", "* /")
    js = "/* KaTeX 0.19.0 — MIT license\n" + license_text + "\n*/\n"
    js += (root / "katex.min.js").read_text(encoding="utf-8").replace("</script", r"<\/script")
    return css, js


def generate_mobile_html_report(translated_hits, target_date_str, save_dir, metadata=None):
    threshold = (metadata or {}).get('threshold_used', 4)
    papers = sorted([p for p in translated_hits if evaluation_score(p.get('eval', {})) >= threshold],
                    key=lambda p: evaluation_score(p['eval']), reverse=True)
    export_cache_path = Path(save_dir) / '.paper_metadata.json'
    export_cache = json.loads(export_cache_path.read_text()) if export_cache_path.exists() else {}
    cards = []
    for index, paper in enumerate(papers, 1):
        evaluation = paper["eval"]
        labels = category_labels(evaluation, threshold)
        badges = "".join(f'<span class="tag" data-tag="{escape(c, quote=True)}">{escape(c)}</span>' for c in labels)
        scores = evaluation.get("scores")
        score = evaluation_score(evaluation)
        if scores:
            badges = "".join(f'<span class="tag{" tag-low" if scores[c] < threshold else ""}" data-tag="{c}">{c} {scores[c]}/5</span>' for c in CATEGORIES)
        else:
            badges += '<span class="status">历史总分；方向未评分</span>'
        score_caption = "/5 最高方向分" if scores else "/5 历史总分"
        status = paper.get("status", "Update")
        url = paper.get("url", "")
        if not url.startswith(("https://", "http://")):
            url = "#"
        extra = export_cache.get(paper.get('url', ''), {})
        export_metadata = {
            'title': paper.get('title', ''), 'authors': paper.get('authors', []),
            'published': paper.get('published'), 'updated': paper.get('updated'),
            'categories': paper.get('arxiv_categories', []), 'url': url,
            'comment': paper.get('comment', extra.get('comment', '未记录')),
            'journal_ref': paper.get('journal_ref', extra.get('journal_ref', '未记录')),
        }
        en, zh, en_ids, zh_ids = align_sentences(paper.get("abstract", ""), paper.get("zh_abstract") or "暂无中文摘要")
        cards.append(f'''<article id="paper-{index}" class="paper-card" data-paper-metadata="{escape(json.dumps(export_metadata, ensure_ascii=False), quote=True)}" data-title="{escape(paper.get('title', ''), quote=True)}" data-bookmark-key="{escape(bookmark_key(paper), quote=True)}" data-categories="{escape(json.dumps(labels), quote=True)}" data-score="{score}" data-category-scores="{escape(json.dumps(scores), quote=True)}" data-new="{str(status == 'v1 (New)').lower()}">
  <div class="card-top"><div class="tags"><span class="result-position" aria-label="当前筛选中的位置">{index}/{len(papers)}</span>{badges}<span class="status">{escape(status)}</span></div><div class="card-actions"><button class="bookmark-button" type="button" aria-pressed="false" aria-label="标记这篇论文" title="标记这篇论文"><span class="bookmark-icon" aria-hidden="true">☆</span></button><span class="score score-{score}">{score}<small>{score_caption}</small></span></div></div>
  <h2>{math_markup(paper.get('title', ''))}</h2>
  <p class="zh-title" lang="zh-CN">{math_markup(paper.get('zh_title', ''))}</p>
  <p class="authors">{escape(', '.join(paper.get('authors', [])))}</p>
  <section class="reason" aria-label="Recommendation Reason">
    <h3>Recommendation Reason <span>推荐理由</span></h3>
    <p lang="en">{math_markup(evaluation.get('reason', 'No recommendation reason available.'))}</p>
    <p lang="zh-CN">{math_markup(evaluation.get('zh_reason') or '暂无中文推荐理由')}</p>
  </section>
  <details class="abstract" data-mode="both">
    <summary>阅读摘要 <span>Abstract</span></summary>
    <div class="abstract-toolbar"><select class="paper-mode" aria-label="这篇论文的摘要语言"><option value="both">中英对照</option><option value="en">仅英文</option><option value="zh">仅中文</option></select></div>
    <div class="abstract-columns"><section class="abstract-en" lang="en"><h3>ENGLISH</h3><p>{sentence_markup(en, en_ids)}</p></section><section class="abstract-zh" lang="zh-CN"><h3>中文</h3><p>{sentence_markup(zh, zh_ids)}</p></section></div>
  </details>
  <footer><span class="result-position" aria-label="当前筛选中的位置">{index}/{len(papers)}</span><a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">在 arXiv 阅读 ↗</a></footer>
</article>''')
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    direction_options = "".join(
        f'<div class="direction-row"><label class="direction-name" for="{c}-mode">{c}</label>'
        f'<select id="{c}-mode" data-direction-mode="{c}" aria-label="{c} 匹配方式">'
        f'<option value="ignore"{" selected" if c not in ("MechInterp", "CompGen") else ""}>不限</option>'
        f'<option value="include"{" selected" if c in ("MechInterp", "CompGen") else ""}>匹配</option>'
        '<option value="exclude">NOT · 排除</option></select>'
        f'<select id="{c}-score" data-direction-score="{c}" aria-label="{c} 分数">'
        '<option value="4">4 分</option><option value="5" selected>5 分</option>'
        '<option value="4,5">4 和 5 分</option></select></div>'
        for c in CATEGORIES)
    content = template.replace("{{DATE}}", escape(target_date_str)).replace("{{DIRECTION_OPTIONS}}", direction_options)
    content = content.replace("{{DEFAULT_EXPRESSION}}", escape(DEFAULT_FILTER_EXPRESSION, quote=True))
    content = content.replace("{{DEFAULT_EXPRESSION_JSON}}", json.dumps(DEFAULT_FILTER_EXPRESSION))
    legacy = any("categories" not in p.get("eval", {}) and "category" in p.get("eval", {}) for p in papers)
    note = ("历史两类分类结果：Reasoning 和 Systems 尚未单独评估。更新分类需要重新筛选原始论文。"
            if legacy else "各方向独立选择 4 分、5 分或 4 和 5 分，支持 AND / OR / NOT；自定义表达式支持括号，裸方向名表示 4 或 5 分")
    if any(not p["eval"].get("scores") for p in papers):
        note = "历史总分结果：没有独立方向分数，方向筛选需重新评分。"
    if metadata and metadata.get("filter_prompt_sha256") not in (None, PROMPT_SHA256):
        note = "这份报告使用此前版本的研究兴趣与评分规则；更新规则需要重新筛选原始论文。"
    elif not papers and not (metadata and metadata.get("filter_categories") == list(CATEGORIES)):
        note = "分类版本未记录；无法从空结果确认 Reasoning 和 Systems 是否已单独评估。"
    if metadata and metadata.get("source") == "recently_public":
        dates = metadata["submission_date_range"]
        kind = metadata.get('batch_kind')
        if kind == 'daily_announcements':
            note += f" · arXiv 当日公告：{metadata['announcement_date']}（官方公告日期）；首次发布与版本更新"
        elif kind == 'catch_up_announcements':
            note += f" · 补抓公告：{metadata['announcement_date']}（官方公告日期）；本次收录：{metadata['collected_on']}（北京时间）"
        elif kind == 'latest_updates':
            note += f" · 最新论文版本；收录日期：{metadata['collected_on']}（北京时间）"
        elif kind == 'catch_up':
            note += f" · 补抓批次：{target_date_str}（北京时间）；公告日期未确认，不计入当日公告"
        else:
            note += f" · 本次抓取日期：{target_date_str}（北京时间）"
        note += f"；版本提交时间：{dates[0]} 至 {dates[1]}"
        if metadata.get('metadata_datestamp_range'):
            changed = metadata['metadata_datestamp_range']
            note += f"；公开元数据修改日期：{changed[0]} 至 {changed[1]}（UTC）"
    if metadata and metadata.get("source") == "initial_backfill":
        note += f" · 首次回补日期：{metadata['backfilled_on']}（北京时间）；版本提交日期：{target_date_str}"
    if metadata and metadata.get('date_basis') == 'metadata_datestamp':
        note += f" · OAI 元数据修改日期：{target_date_str}（服务端记录日期）；每篇论文仅保留最新版本，版本提交时间单独记录"
    content = content.replace("{{CLASSIFICATION_NOTE}}", escape(note))
    css, js = math_assets()
    content = content.replace("{{MATH_CSS}}", css).replace("{{MATH_JS}}", js)
    content = content.replace("{{TOTAL}}", str(len(papers))).replace("{{CARDS}}", "\n".join(cards))
    directory = Path(save_dir) / "html_reports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"report_{target_date_str}.html"
    path.write_text(content, encoding="utf-8")
    return str(path)

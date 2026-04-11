import arxiv
import os
import json
import re
import argparse
from datetime import datetime, timezone, timedelta
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ==========================================
# [Optional Configuration] Uncomment if the server needs a proxy to pull arXiv
proxy = "http://127.0.0.1:1081"
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
# ==========================================

# ==================== Prompts ====================
SYSTEM_PROMPT = """You are a senior AI researcher and an expert literature reviewer.
Your task is to read the title and abstract of recently published arXiv papers and determine if they belong to either of these two specific subfields:

1. Mechanistic Interpretability & Representation Geometry (MechInterp): Look for studies deciphering the internal workings of language models, how they represent concepts, and their modular structures. Key topics include:
   - Feature & Concept Extraction: Sparse Autoencoders (SAEs), dictionary learning, superposition, monosemanticity, and feature attribution.
   - Circuit Analysis: Causal tracing, subnetwork discovery, understanding specific components (e.g., attention heads, MLPs), and reverse-engineering learned behaviors.
   - Representation Geometry: The geometric structure of latent spaces, linear representations, concept subspaces and concept feature relationships.
   - Knowledge Mechanisms: How facts are stored and recalled, knowledge localization, and parameter-level knowledge editing.
   - Other topics aimed at deciphering or interpreting internal model behaviors.

2. Compositional Generalization & Reasoning (CompGen): Look for studies examining how models systematically generalize to novel combinations of known elements, execute multi-step complex reasoning, or utilize modular structures. Key topics include:
   - Systematic Generalization: Combinatorial generalization, length extrapolation, Out-of-Distribution (OOD) robustness via structural composition, and algebraic/rule-based generalization.
   - Complex Reasoning Mechanisms: Chain-of-Thought (CoT) and its variants, continuous/latent space reasoning, multi-hop logical deduction, algorithmic reasoning, and mathematical/symbolic problem-solving.
   - Modular Architectures & Tuning: Mixture of Experts (MoE), dynamic routing mechanisms, neuro-symbolic integration, plug-and-play modularity, and concept-guided learning.
   - Compositional Data & Evaluation: Compositional instruction tuning, synthetic data generation for reasoning, and benchmarks evaluating compositional capabilities.
   - Other topics explicitly aimed at improving or evaluating the systematic reasoning and modular composition of AI models.

You must evaluate the paper and return ONLY a valid JSON object. 
CRITICAL: Do not include markdown formatting, code blocks (```json), or any conversational text. 
The JSON must strictly follow this schema:
{
  "relevance_score": integer, // Use this strict rubric: 1 = Unrelated (does not touch the subfields); 2 = Mentioned Only (keywords appear in background/future work, but not the core focus); 3 = Peripheral/Application (applies concepts as tools but lacks mechanism depth); 4 = Strong Match (core contribution improves/analyzes mechanisms in the subfields); 5 = Core Contribution (fundamentally solves or heavily focuses on base problems in MechInterp/CompGen).
  "category": string, // "MechInterp", "CompGen", "Both", or "None"
  "reason": string // A concise, one-sentence justification for your decision. Better to summarize the core contribution of the paper and how it relates to the subfield(s) than to simply restate the title/abstract.
}
"""

USER_TEMPLATE = """Title: {title}
Abstract: {abstract}
"""

TRANSLATION_SYSTEM_PROMPT = """You are an expert academic translator. Translate the given Title, Abstract, and Reason from English to Chinese.
Return ONLY a valid JSON object. Do not include markdown formatting or conversational text.
The JSON must strictly follow this schema:
{
  "zh_title": "string",
  "zh_abstract": "string",
  "zh_reason": "string"
}
"""

TRANSLATION_USER_TEMPLATE = """Title: {title}
Abstract: {abstract}
Reason: {reason}
"""

# ==================== Module 1: Fetching ====================

def fetch_papers_by_date(target_date_str=None, buffer_limit=1000, max_retries=3, retry_delay=5) -> tuple:
    """Optimized fetch using arXiv's native date querying API with retry mechanism."""
    import time
    
    if target_date_str is None:
        # Default to UTC "yesterday"
        target_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    else:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        
    print(f"\n[1/4] Target locked: Fetching all updates for UTC date {target_date}")
    
    # Use arXiv native Lucene syntax to filter by date on the server side
    date_query_str = target_date.strftime("%Y%m%d")
    # Build Query with date range
    query = f"(cat:cs.AI OR cat:cs.LG OR cat:cs.CL) AND submittedDate:[{date_query_str}0000 TO {date_query_str}2359]"
    
    search = arxiv.Search(
        query=query,
        max_results=buffer_limit, 
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    client = arxiv.Client()
    
    for attempt in range(max_retries):
        papers = []
        try:
            for paper in client.results(search):
                paper_date = paper.published.date()
                
                # Double Check for defensive programming (arXiv API occasionally returns slight time offsets)
                if paper_date == target_date:
                    papers.append({
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "abstract": paper.summary.replace('\n', ' '),
                        "url": paper.entry_id,
                        "date": paper.published.strftime("%Y-%m-%d %H:%M:%S UTC")
                    })
                elif paper_date < target_date:
                    # Keep truncation logic as a safety measure, though usually not triggered
                    print(f"      -> Encountered earlier date ({paper_date}), stopping fetch.")
                    break

            print(f"Fetch complete: {len(papers)} AI-related updates found for {target_date}.")
            return papers, str(target_date)
            
        except Exception as e:
            print(f"Fetch failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Fetch failed completely.")
                return [], str(target_date)

# ==================== Module 2: LLM Parsing ====================

def parse_json_response(response_text: str) -> dict:
    """Robust JSON parser handling potential Markdown formatting in response body"""
    default_ret = {
        "relevance_score": 1,
        "category": "None",
        "reason": "Failed to parse LLM output."
    }
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return default_ret

# ==================== Module 3: vLLM Filtering ====================

def filter_papers(papers: list, llm: LLM, tokenizer, threshold: int = 4):
    """Batch construct prompts and invoke vLLM for filtering"""
    print(f"\n[3/4] Building inference tasks for {len(papers)} papers, delegating to LLM judge...")
    
    prompts = []
    for paper in papers:
        user_content = USER_TEMPLATE.format(title=paper['title'], abstract=paper['abstract'])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        prompts.append({"prompt_token_ids": token_ids})

    # Set temperature to 0 for deterministic classification, max_tokens small for JSON output
    sampling_params = SamplingParams(temperature=0.0, max_tokens=150)
    
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    interesting_papers = []
    
    for paper, output in zip(papers, outputs):
        generated_text = output.outputs[0].text.strip()
        eval_result = parse_json_response(generated_text)
        paper['eval'] = eval_result
        
        # Meets relevance and threshold criteria
        if eval_result.get('relevance_score', 0) >= threshold:
            interesting_papers.append(paper)
            
    return interesting_papers, papers

# ==================== Module 4: Translation ====================

def translate_hits(hits: list, llm: LLM, tokenizer):
    """Batch translate hits to Chinese"""
    if not hits:
        return []
        
    print(f"\n[4/5] Translating {len(hits)} hits to Chinese...")
    prompts = []
    import copy
    
    for paper in hits:
        user_content = TRANSLATION_USER_TEMPLATE.format(
            title=paper['title'], 
            abstract=paper['abstract'],
            reason=paper['eval'].get('reason', '')
        )
        messages = [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        prompts.append({"prompt_token_ids": token_ids})

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1000)
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    
    translated_hits = []
    
    for paper, output in zip(hits, outputs):
        generated_text = output.outputs[0].text.strip()
        trans_result = parse_json_response(generated_text)
        
        t_paper = copy.deepcopy(paper)
        t_paper['zh_title'] = trans_result.get('zh_title', paper['title'])
        t_paper['zh_abstract'] = trans_result.get('zh_abstract', paper['abstract'])
        t_paper['eval']['zh_reason'] = trans_result.get('zh_reason', paper['eval'].get('reason', ''))
        translated_hits.append(t_paper)
        
    return translated_hits

# ==================== Module 5: Mobile HTML Report ====================

def generate_mobile_html_report(translated_hits, target_date_str, save_dir):
    """Generate a mobile-friendly HTML report for the filtered papers."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ArXiv 推荐: {target_date_str}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; padding: 15px; max-width: 600px; margin: auto; background-color: #f4f4f9; color: #333; }}
            h2 {{ text-align: center; color: #2c3e50; font-size: 1.4em; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
            .paper-card {{ background: #fff; padding: 16px; margin-bottom: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            .badges {{ margin-bottom: 12px; }}
            .badge-cat {{ display: inline-block; padding: 4px 8px; background: #3498db; color: #fff; border-radius: 6px; font-size: 0.8em; font-weight: bold; }}
            .badge-score {{ display: inline-block; padding: 4px 8px; background: #e74c3c; color: #fff; border-radius: 6px; font-size: 0.8em; font-weight: bold; margin-left: 6px; }}
            .zh-title {{ font-size: 1.15em; color: #2c3e50; font-weight: bold; margin-bottom: 6px; }}
            .en-title {{ font-size: 0.9em; color: #7f8c8d; margin-bottom: 10px; line-height: 1.4; }}
            .authors {{ font-size: 0.85em; color: #95a5a6; margin-bottom: 12px; font-style: italic; word-wrap: break-word; }}
            .reason {{ background: #fff8e1; padding: 12px; border-left: 4px solid #f1c40f; border-radius: 4px; font-size: 0.95em; margin-bottom: 12px; color: #5d4037; }}
            
            /* Abstract collapsible panel style */
            .abstract-group {{ margin-top: 10px; display: flex; flex-direction: column; gap: 8px; }}
            .abstract-toggle {{ cursor: pointer; font-size: 0.9em; font-weight: bold; padding: 5px 0; outline: none; }}
            .toggle-zh {{ color: #2980b9; }}
            .toggle-en {{ color: #8e44ad; }} /* Purple color for English abstract button */
            .abstract-content {{ font-size: 0.9em; color: #555; background: #f9f9f9; padding: 10px; border-radius: 6px; margin-top: 5px; }}
            .en-text {{ font-family: "Georgia", serif; line-height: 1.5; }} /* Different serif font for English content, more suitable for academic reading */
            
            .link-btn {{ display: block; text-align: center; margin-top: 15px; padding: 10px; background: #2ecc71; color: #fff; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 0.95em; }}
            .link-btn:active {{ background: #27ae60; }}
        </style>
    </head>
    <body>
        <h2>📚 ArXiv 每日高价值论文<br><span style="font-size: 0.7em; color: #7f8c8d;">{target_date_str}</span></h2>
    """

    if not translated_hits:
        html_content += '<div class="paper-card" style="text-align: center; color: #7f8c8d;">今日无符合阈值的高价值论文。</div>'
    else:
        for paper in translated_hits:
            eval_data = paper.get('eval', {})
            cat = eval_data.get('category', 'None')
            score = eval_data.get('relevance_score', 0)
            title = paper.get('title', '')
            zh_title = paper.get('zh_title', title)
            authors = ", ".join(paper.get('authors', []))
            reason = eval_data.get('zh_reason', eval_data.get('reason', '无推荐理由'))
            
            # Extract Chinese and English abstracts
            zh_abstract = paper.get('zh_abstract', '无中文翻译摘要')
            en_abstract = paper.get('abstract', 'No English abstract available.')

            html_content += f"""
            <div class="paper-card">
                <div class="badges">
                    <span class="badge-cat">{cat}</span>
                    <span class="badge-score">Score: {score}/5</span>
                </div>
                <div class="zh-title">{zh_title}</div>
                <div class="en-title">{title}</div>
                <div class="authors">{authors}</div>
                <div class="reason"><strong>💡 推荐理由：</strong><br>{reason}</div>
                
                <div class="abstract-group">
                    <details>
                        <summary class="abstract-toggle toggle-zh">展开中文摘要</summary>
                        <div class="abstract-content">{zh_abstract}</div>
                    </details>
                    
                    <details>
                        <summary class="abstract-toggle toggle-en">Read English Abstract</summary>
                        <div class="abstract-content en-text">{en_abstract}</div>
                    </details>
                </div>
                
                <a href="{paper.get('url', '#')}" class="link-btn" target="_blank">前往 arXiv 原文</a>
            </div>
            """

    html_content += """
    </body>
    </html>
    """

    html_dir = os.path.join(save_dir, "html_reports")
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(html_dir, f"report_{target_date_str}.html")
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return html_path

# ==================== Main Runner ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArXiv Daily Paper Filter using vLLM")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD. Defaults to yesterday (UTC).")
    parser.add_argument("--threshold", type=int, default=4, help="Minimum relevance score threshold (1-5)")
    parser.add_argument("--model_path", type=str, default="./weights/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor Parallelism size")
    parser.add_argument("--gpu_util", type=float, default=0.85, help="vLLM GPU memory utilization ratio")
    parser.add_argument("--max_num_seqs", type=int, default=256, help="Maximum number of sequences to process in parallel")
    parser.add_argument("--save_dir", type=str, default="./llm-paper-filter/daily_arxiv", help="Base directory to save results")

    args = parser.parse_args()

    # Step 1: Precise fetch
    papers, target_date_str = fetch_papers_by_date(target_date_str=args.date)
    
    if not papers:
        print("No updates found for today or network error. Exiting.")
        exit()

    # Step 2: Initialize vLLM (Delayed here to avoid occupying VRAM if no papers found)
    print(f"\n[2/4] Initializing vLLM and model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        quantization="awq_marlin",
        # quantization="awq",
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=True,
        max_num_seqs=256,
        max_model_len=4096
    )

    # Step 3: Run filtering pipeline
    hits, all_papers_evaluated = filter_papers(papers, llm, tokenizer, threshold=args.threshold)

    # Step 4: Translate hits
    translated_hits = translate_hits(hits, llm, tokenizer)

    # Step 5: Output report and save results
    print(f"\n[5/5] Processing complete! Preparing to save results...")
    print("="*80)
    print(f"From {len(all_papers_evaluated)} papers, the model unearthed {len(hits)} high-value papers!")
    print("="*80 + "\n")
    
    # Sort by score descending
    hits.sort(key=lambda x: x['eval'].get('relevance_score', 0), reverse=True)
    translated_hits.sort(key=lambda x: x['eval'].get('relevance_score', 0), reverse=True)
    
    for i, paper in enumerate(translated_hits, 1):
        eval_data = paper['eval']
        print(f"[{i}] {eval_data.get('category')} | Score: {eval_data.get('relevance_score')}/5")
        print(f"Title: {paper['title']}")
        print(f"Title(zh): {paper.get('zh_title', '')}")
        print(f"URL: {paper['url']}")
        print(f"Authors: {', '.join(paper.get('authors', []))}")
        print(f"Reason: {eval_data.get('zh_reason', eval_data.get('reason', ''))}")
        print("-" * 80)
        
# ================= Classification & Save Logic =================
    # Create sub-folders
    hits_dir = os.path.join(args.save_dir, "hits")
    hits_zh_dir = os.path.join(args.save_dir, "hits_zh")
    all_papers_dir = os.path.join(args.save_dir, "all_papers")
    
    os.makedirs(hits_dir, exist_ok=True)
    os.makedirs(hits_zh_dir, exist_ok=True)
    os.makedirs(all_papers_dir, exist_ok=True)
    
    # Construct data payloads for saving
    hits_payload = {
        "metadata": {
            "target_date": target_date_str,
            "total_hits": len(hits),
            "threshold_used": args.threshold
        },
        "hits": hits
    }

    hits_zh_payload = {
        "metadata": {
            "target_date": target_date_str,
            "total_hits": len(translated_hits),
            "threshold_used": args.threshold
        },
        "hits_zh": translated_hits
    }
    
    all_papers_payload = {
        "metadata": {
            "target_date": target_date_str,
            "total_papers_fetched": len(all_papers_evaluated),
            "threshold_used": args.threshold
        },
        "all_papers_evaluated": all_papers_evaluated
    }

    # Save file paths
    hits_save_path = os.path.join(hits_dir, f"hits_{target_date_str}.json")
    hits_zh_save_path = os.path.join(hits_zh_dir, f"hits_zh_{target_date_str}.json")
    all_papers_save_path = os.path.join(all_papers_dir, f"all_papers_{target_date_str}.json")
    
    # Write JSON
    with open(hits_save_path, "w", encoding="utf-8") as f:
        json.dump(hits_payload, f, ensure_ascii=False, indent=4)

    with open(hits_zh_save_path, "w", encoding="utf-8") as f:
        json.dump(hits_zh_payload, f, ensure_ascii=False, indent=4)
        
    with open(all_papers_save_path, "w", encoding="utf-8") as f:
        json.dump(all_papers_payload, f, ensure_ascii=False, indent=4)

    # === Generate HTML report for mobile reading ===
    html_report_path = generate_mobile_html_report(translated_hits, target_date_str, args.save_dir)

    print(f"\nHigh-value hits saved to: {hits_save_path}")
    print(f"Translated hits saved to: {hits_zh_save_path}")
    print(f"Full evaluation log saved to: {all_papers_save_path}")
    print(f"Mobile HTML report saved to: {html_report_path}") # Log HTML path
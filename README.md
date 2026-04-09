# 📚LLM-Paper-Finder

An automated, LLM-driven pipeline designed to sift through the daily avalanche of AI research papers. 

Currently, this repository features `daily_arxiv_filter.py`, a high-throughput tool that fetches daily updates from **arXiv**, uses local LLMs via `vLLM` to score and categorize papers based on specific research interests, and translates the highlights' title and abstract into Chinese. 

---

## 📂 Directory Structure

```text
.
└── daily_arxiv
    ├── all_papers/            # Stores the full evaluation logs of all fetched papers
    ├── hits/                  # Stores the filtered "high-value" papers (Score >= threshold)
    ├── hits_zh/               # Stores the translated Chinese versions of the hits
    └── daily_arxiv_filter.py  # The core execution script
```

------

## 🛠️ Prerequisites

**Dependencies:**

```
conda create -n paper-finder python=3.11
pip install arxiv vllm transformers
```

------

## 🗒️ Usage

### daily_arxiv / daily_arxiv_filter.py

```bash
python daily_arxiv/daily_arxiv_filter.py \
  --date 2026-04-08 \ # if not given, fetch arxiv papers submitted yesterday (UTC Time)
  --threshold 4 \
  --model_path /path/to/your/local/model \
  --tp_size 2 \
  --gpu_util 0.85 \
  --save_dir /path/to/save/directory
```

#### Arguments 

| **Argument**     | **Default**                            | **Description**                                              |
| ---------------- | -------------------------------------- | ------------------------------------------------------------ |
| `--date`         | `None` (Yesterday UTC)                 | Target date in `YYYY-MM-DD` format.                          |
| `--threshold`    | `4`                                    | Minimum relevance score (1-5) to classify a paper as a "hit". |
| `--model_path`   | `Meta-Llama-3.1-70B-Instruct-AWQ-INT4` | Path to the local HuggingFace model weights.                 |
| `--tp_size`      | `2`                                    | Tensor Parallelism size (Number of GPUs to split the model across). |
| `--gpu_util`     | `0.85`                                 | GPU memory utilization ratio for vLLM.                       |
| `--max_num_seqs` | `256`                                  | Maximum number of sequences to process in parallel           |
| `--save_dir`     | `daily_arxiv`                          | Base directory to save the output JSON files.                |

The script defaults to a 70B Llama-3.1 INT4 AWQ model using 2 GPUs of 48GB via Tensor Parallelism.

#### Prompt Demo

```
You are a senior AI researcher and an expert literature reviewer.
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
```

---

## 🚀TODO

⬜ Change LLM model version for best performance.

⬜ Build GUI for better reading.

⬜ Support major AI conferences (e.g., ICLR, NeurIPS, ICML, ACL).
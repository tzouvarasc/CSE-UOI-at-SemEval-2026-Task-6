# CSE-UOI @ SemEval-2026 Task 6 — Code Deliver

## Mini Introduction
This repository implements our SemEval-2026 Task 6 system for political interview clarity classification (`Clear Reply`, `Ambivalent`, `Clear Non-Reply`).  
The core approach is a two-stage pipeline:  
1. a heterogeneous dual-LLM ensemble with self-consistency and weighted voting (Stage 1),  
2. a post-hoc Deliberative Complexity Gating (DCG) layer that uses cross-model behavioral signals (Stage 2).  
In the official SemEval-2026 evaluation, this setup reached Macro-F1 `0.85` and ranked `3rd`.

---

## Folder Layout
- `code/`: runnable scripts (Stage 1, Stage 2, debate ablations)
- `files/`: input/output data files (CSV, TXT labels, detailed JSONs)
- `analysis/`: offline analysis script and generated analysis JSON

---

## Scripts (what they do + how to run)

### 1) `code/stage1.py`
Stage-1 inference (Grok + Gemini, self-consistency, weighted vote).  
Can run on HuggingFace split (`--split`) or on evaluation CSV (`--eval_csv`).  

Key outputs (by `--output_prefix`):
- `<prefix>_detailed.json` (if `--detailed_json`)
- `<prefix>_full.json` (if not detailed)
- `<prefix>_clarity.pkl`, `<prefix>_evasion.pkl`

Run example (evaluation CSV, detailed JSON):
```bash
python code/stage1.py \
  --grok_api_key "$XAI_API_KEY" \
  --gemini_api_key "$GEMINI_API_KEY" \
  --eval_csv files/clarity_task_evaluation_dataset.csv \
  --output_prefix files/stage1_eval_set \
  --k_samples 5 \
  --thinking_level high \
  --detailed_json
```

---

### 2) `code/dcg_stage2.py`
Stage-2 DCG post-processing on Stage-1 detailed JSON.  
No API calls. Uses Stage-1 metadata only.

Key outputs:
- Stage-2 detailed JSON (`--output`)

Run example:
```bash
python code/dcg_stage2.py \
  --input files/stage1_eval_set_detailed.json \
  --output files/stage2_eval_set_detailed.json \
  --task1_labels_txt files/task1_eval_labels.txt \
  --task2_labels_txt files/task2_eval_labels.txt \
  --percentile 25 \
  --grok-threshold 1.0
```

---

### 3) `code/debate_ablation/debate.py`
APEX debate ablation (supports:
- `--mode ultimate`: static weighted voting baseline
- `--mode debate`: iterative cross-model debate on disagreements)

Key outputs (by `--output_prefix`):
- `<prefix>_full.json`
- `<prefix>_clarity.pkl`, `<prefix>_evasion.pkl`
- optional `<prefix>_reasoning.txt` (if `--save_reasoning`)

Run example (debate mode on test split):
```bash
python code/debate_ablation/debate.py \
  --grok_api_key "$XAI_API_KEY" \
  --gemini_api_key "$GEMINI_API_KEY" \
  --split test \
  --mode debate \
  --k_samples 5 \
  --max_debate_rounds 2 \
  --debate_k 3 \
  --output_prefix code/debate_ablation/test_set
```

---

### 4) `code/Hu_Debate_ablation/hu_debate.py`
Reproduction of Hu-style homogeneous 7-agent Grok debate baseline.

Key outputs (by `--output_prefix`):
- `<prefix>_full.json`

Run example:
```bash
python code/Hu_Debate_ablation/hu_debate.py \
  --grok_api_key "$XAI_API_KEY" \
  --split test \
  --n_agents 7 \
  --max_rounds 3 \
  --output_prefix code/Hu_Debate_ablation/hu_debate
```

---

### 5) `analysis/analysis.py`
Offline comprehensive analysis (Sections A-H), no API calls.  
Consumes Stage-2 detailed JSONs and optional debate/Hu artifacts.

Main output:
- `analysis/results_ALL.json` (or `--out-json`)

Run with defaults:
```bash
python analysis/analysis.py
```

Run :
```bash
python analysis/analysis.py \
  --eval-json files/stage2_eval_set_detailed.json \
  --test-json files/stage2_test_set_detailed.json \
  --task1-labels files/task1_eval_labels.txt \
  --task2-labels files/task2_eval_labels.txt \
  --debate-json code/debate_ablation/test_set_detailed.json \
  --hu-json code/Hu_Debate_ablation/hu_debate_full.json \
  --hu-metrics-json code/Hu_Debate_ablation/hu_debate_metrics.json \
  --out-json analysis/results_ALL.json
```

---

## Minimal Requirements
- Python 3.10+
- `openai` (Grok API client)
- `google-genai` (Gemini API client)
- `pandas`
- `datasets`
- `scikit-learn` (required by Stage 1 / debate metrics; optional in Stage 2)

Install:
```bash
pip install openai google-genai pandas datasets scikit-learn
```

---

## API Keys
You can pass keys via CLI or environment variables:
- `XAI_API_KEY`
- `GEMINI_API_KEY`

Example:
```bash
export XAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

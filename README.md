# FIN-bench-eval

Evaluation pipeline for [FIN-bench-v2](https://github.com/TurkuNLP/FIN-bench-v2) —
a Finnish language model benchmark covering 12 tasks across factual recall,
commonsense reasoning, reading comprehension, sentiment/semantic tasks,
and ethical alignment.

This repository accompanies the paper:

> **FIN-bench-v2 Evaluation: Comparing Frontier API Models and Local Inference
> on Finnish Benchmarks**  
> Janne Tompuri, University of Helsinki, 2026

Scored outputs for all 12 model configurations are available on
[Hugging Face Datasets](https://huggingface.co/datasets/jtompuri/finbench-eval-outputs)
(see [Data](#data)).

---

## Key results

| Model | Normalised score |
|---|---|
| Gemini 3 Flash | **0.919** |
| Gemini 3.1 Pro | 0.918 |
| GPT-5.4 (CoT) | 0.907 |
| GPT-5.4 | 0.896 |
| Claude Sonnet 4.6 | 0.880 |
| Gemma 4 26B *(best local)* | 0.822 |
| Llama 3.1 8B | 0.496 |

Scores are normalised: $(s - b)/(1 - b)$, where $b$ is the random-chance
baseline. Mean over 11 tasks (SQuAD excluded from aggregate).

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/jtompuri/finbench-eval.git
cd finbench-eval
pip install -r requirements.txt

# 2. Add API keys
cp .env.example .env
# Edit .env and fill in your keys

# 3. Run a small test (5 items, ARC Challenge FI, GPT-5.4)
source .env
python scripts/run_frontier_jsonl.py \
    --provider openai \
    --model-id gpt-5.4 \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/test_openai.jsonl \
    --subset arc_challenge_fi \
    --n 5 \
    --verbose

# 4. Score the output
python scripts/score_eval.py \
    --input outputs/test_openai.jsonl \
    --output results/raw/test_openai.json
```

For local models (Apple Silicon / MLX):
```bash
python scripts/run_eval_jsonl.py \
    --model mlx-community/gemma-4-e4b-it-4bit \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/test_gemma4e4b.jsonl \
    --n 5 \
    --verbose
```

---

## Data

**Evaluation dataset:** Download from Hugging Face and place in `data/`:
```python
from datasets import load_dataset
# Individual tasks available under TurkuNLP/finbenchv2-*
```
The fixed 1,146-item evaluation subset (`data/finbench_combined_v1.jsonl`)
is included in this repository.

**Scored outputs:** All model responses and scores are published at
[jtompuri/finbench-eval-outputs](https://huggingface.co/datasets/jtompuri/finbench-eval-outputs)
on Hugging Face Datasets. This enables error analysis, new metrics
(e.g. LLM-as-a-judge), and reproduction without re-running API calls.

---

## Repository structure

```
scripts/
    eval_config.py          # Baselines and normalisation formula
    score_eval.py           # Scoring: accuracy, F1, Wilson CI
    normalize_answer.py     # Answer extraction (MCF letter/word, generative)
    aggregate_results.py    # Aggregate per-task scores → tidy CSV
    run_frontier_jsonl.py   # Run frontier API models
    run_eval_jsonl.py       # Run local MLX models
    frontier_adapters.py    # API adapters (OpenAI, Anthropic, Google)
    plot_figures.py         # Reproduce all paper figures
    plot_style.py           # Shared matplotlib style
    compute_bertscore_squad.py  # BERTScore validation for SQuAD FI
    analysis/
        final_summary.py    # Per-model summary table
        compare_runs.py     # Pairwise run comparison
        mcnemar_test.py     # McNemar significance tests (BH-corrected)
        analysis_normalized.py

report/
    draft_v1.tex            # Full paper source (ACL format)
    references.bib

data/
    finbench_combined_v1.jsonl   # Fixed 1,146-item evaluation subset
```

---

## Methodology notes

- **Generative scoring** throughout (no log-probabilities required),
  enabling evaluation of closed frontier APIs.
- Results are **not directly comparable** to official FIN-bench-v2
  log-probability scores.
- Random-chance baselines are task-specific; see `scripts/eval_config.py`.
- SQuAD FI uses token-level F1 and is excluded from the normalised aggregate.
  BERTScore validation confirms token-F1 underestimates frontier model
  performance by 28–45 pp due to Finnish morphological variation.

---

## Citation

```bibtex
@article{tompuri2026finbench,
  title   = {{FIN-bench-v2} Evaluation: Comparing Frontier {API} Models
             and Local Inference on Finnish Benchmarks},
  author  = {Tompuri, Janne},
  year    = {2026},
}
```

---

## License

Code: [MIT](LICENSE)  
Evaluation data: see individual
[TurkuNLP/FIN-bench-v2](https://github.com/TurkuNLP/FIN-bench-v2) dataset licences.

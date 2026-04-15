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
# 1. Clone
git clone https://github.com/jtompuri/finbench-eval.git
cd finbench-eval

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API keys
cp .env.example .env
# Edit .env and fill in your keys
```

> **Note:** `mlx-lm` (required for Apple Silicon local inference) installs only
> on macOS with an M-series chip. On other platforms, comment out the `mlx-lm`
> line in `requirements.txt` before running `pip install`.

### Frontier models (API)

```bash
# Run 5 items on ARC Challenge FI with GPT-5.4
source .env
python scripts/run_frontier_jsonl.py \
    --provider openai \
    --model-id gpt-5.4 \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/test_openai.jsonl \
    --subset arc_challenge_fi \
    --n 5 \
    --verbose
```

Supported `--provider` values:

| Provider | Description |
|---|---|
| `openai` | OpenAI Chat Completions (e.g. GPT-5.4) |
| `openai-thinking` | OpenAI Responses API with reasoning (CoT) |
| `anthropic` | Anthropic Messages API (e.g. Claude Sonnet 4.6) |
| `anthropic-thinking` | Anthropic extended thinking (CoT) |
| `google` | Google Gemini via AI Studio |
| `openrouter` | OpenRouter — unified access to third-party models |

Batch submission (50 % cost reduction, ~24 h turnaround) is supported for
`openai` and `anthropic`:

```bash
# Submit batch
python scripts/run_frontier_jsonl.py \
    --provider anthropic --model-id claude-sonnet-4-6 \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/combined_anthropic.jsonl \
    --batch

# Check status
python scripts/run_frontier_jsonl.py \
    --provider anthropic \
    --output outputs/combined_anthropic.jsonl \
    --batch-status

# Fetch results when complete
python scripts/run_frontier_jsonl.py \
    --provider anthropic --model-id claude-sonnet-4-6 \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/combined_anthropic.jsonl \
    --batch-fetch
```

For `google` and `openrouter`, use `--concurrency` instead of `--batch`:

```bash
python scripts/run_frontier_jsonl.py \
    --provider google --model-id gemini-3-flash \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/combined_google.jsonl \
    --concurrency 5
```

### Local models — Apple Silicon (MLX)

`run_eval_jsonl.py` uses [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms)
and requires an Apple M-series chip. Models are downloaded automatically from
Hugging Face on first use — no separate download step needed.

> **Platform:** macOS with Apple Silicon (M1 or later) only. On other
> platforms, remove or comment out the `mlx-lm` line in `requirements.txt`
> before running `pip install -r requirements.txt`.

Install mlx-lm:

```bash
pip install mlx-lm
```

Run evaluation:

```bash
# Smoke test — 5 items with live accuracy output
python scripts/run_eval_jsonl.py \
    --model mlx-community/gemma-4-e4b-it-4bit \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/test_gemma4e4b.jsonl \
    --n 5 --verbose

# Full run
python scripts/run_eval_jsonl.py \
    --model mlx-community/gemma-4-e4b-it-4bit \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/combined_gemma4e4b.jsonl

# Resume an interrupted run
python scripts/run_eval_jsonl.py \
    --model mlx-community/gemma-4-e4b-it-4bit \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/combined_gemma4e4b.jsonl \
    --resume
```

Models used in this study (all from [mlx-community](https://huggingface.co/mlx-community)):

| Model | HF repo | Size |
|---|---|---|
| Gemma 4 E4B | `mlx-community/gemma-4-e4b-it-4bit` | ~3 GB |
| Gemma 4 26B | `mlx-community/gemma-4-26b-it-4bit` | ~15 GB |
| Gemma 3 27B | `mlx-community/gemma-3-27b-it-qat-4bit` | ~15 GB |
| Llama 3.1 8B | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | ~5 GB |
| Poro-8B | `aciidix/Llama-Poro-2-8B-Instruct-mlx-4Bit` | ~5 GB |

### Local models — llama.cpp / GGUF (Apple Silicon · NVIDIA · AMD · CPU)

`run_llama_jsonl.py` uses [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
and supports any platform that can run GGUF models.

> **Python version:** llama-cpp-python requires **Python 3.14+** on macOS.
> Python 3.13 (conda / miniconda) causes a silent crash at model load time.
> Install Python 3.14 via `brew install python@3.14` and create the virtual
> environment explicitly: `python3.14 -m venv .venv`

Install llama-cpp-python for your hardware:

```bash
# Apple Silicon — Metal GPU (macOS M-series)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall

# NVIDIA GPU — CUDA (Linux / Windows)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

# AMD GPU — Vulkan (Linux / Windows)
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --force-reinstall

# CPU only (no GPU required, any platform)
pip install llama-cpp-python
```

Download a GGUF model from Hugging Face:

```python
from huggingface_hub import hf_hub_download

# Gemma 4 E4B (used in this study) — ~5 GB
hf_hub_download(
    repo_id="bartowski/google_gemma-4-E4B-it-GGUF",
    filename="google_gemma-4-E4B-it-Q4_K_M.gguf",
    local_dir="./models",
)

# Gemma 3 27B — ~17 GB
hf_hub_download(
    repo_id="bartowski/gemma-3-27b-it-GGUF",
    filename="gemma-3-27b-it-Q4_K_M.gguf",
    local_dir="./models",
)
```

Run evaluation:

```bash
# Smoke test — 5 items with live accuracy output
python scripts/run_llama_jsonl.py \
    --model models/google_gemma-4-E4B-it-Q4_K_M.gguf \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/test_gemma4e4b_llama.jsonl \
    --n 5 --verbose

# Full run — GPU (Metal / CUDA / Vulkan, all layers offloaded by default)
python scripts/run_llama_jsonl.py \
    --model models/google_gemma-4-E4B-it-Q4_K_M.gguf \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/combined_gemma4e4b_llama.jsonl

# Resume an interrupted run
python scripts/run_llama_jsonl.py \
    --model models/google_gemma-4-E4B-it-Q4_K_M.gguf \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/combined_gemma4e4b_llama.jsonl \
    --resume

# CPU only (no GPU)
python scripts/run_llama_jsonl.py \
    --model models/google_gemma-4-E4B-it-Q4_K_M.gguf \
    --n-gpu-layers 0 \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/combined_gemma4e4b_cpu.jsonl
```

### Scoring

```bash
python scripts/score_eval.py \
    --input outputs/test_openai.jsonl \
    --output results/raw/test_openai.json
```

---

## Data

**Evaluation dataset:** The fixed 1,146-item evaluation subset
(`data/finbench_combined_v1.jsonl`) is included in this repository.
Individual task datasets are available under
[TurkuNLP/finbenchv2-*](https://huggingface.co/TurkuNLP) on Hugging Face.

**Scored outputs:** All model responses and scores are published at
[jtompuri/finbench-eval-outputs](https://huggingface.co/datasets/jtompuri/finbench-eval-outputs)
on Hugging Face Datasets. This enables error analysis, new metrics
(e.g. LLM-as-a-judge), and reproduction without re-running API calls.

To export and upload your own scored outputs:

```bash
python scripts/export_hf_dataset.py \
    --out-dir hf_export \
    --outputs-dir outputs \
    --scores-dir results/raw \
    --push \
    --repo your-username/your-dataset-repo
```

---

## Repository structure

```
scripts/
    run_frontier_jsonl.py   # Run frontier API models (OpenAI, Anthropic, Google, OpenRouter)
    run_eval_jsonl.py       # Run local MLX models (Apple Silicon)
    run_llama_jsonl.py      # Run local GGUF models via llama.cpp (Windows / Linux / Intel Mac)
    frontier_adapters.py    # Provider abstraction + API adapters (single-call & batch)
    score_eval.py           # Scoring: accuracy, F1, Wilson CI
    eval_config.py          # Task baselines and normalisation formula
    normalize_answer.py     # Answer extraction (MCF letter/word, generative)
    export_hf_dataset.py    # Export scored outputs to Hugging Face Datasets format
    aggregate_results.py    # Aggregate per-task scores → tidy CSV
    plot_figures.py         # Reproduce all paper figures
    plot_style.py           # Shared matplotlib style
    build_subset_jsonl.py   # Build evaluation subsets from raw task data
    run_mlx_prompt.py       # Single-prompt MLX runner (smoke tests)
    run_llama_prompt.py     # Single-prompt llama.cpp runner (smoke tests)
    compute_bertscore_squad.py  # BERTScore validation for SQuAD FI
    analysis/
        final_summary.py        # Per-model summary table
        compare_runs.py         # Pairwise run comparison
        mcnemar_test.py         # McNemar significance tests (BH-corrected)
        analysis_normalized.py  # Normalised score analysis

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

# FIN-bench-eval

Evaluation pipeline for [FIN-bench-v2](https://github.com/TurkuNLP/FIN-bench-v2),
a Finnish language model benchmark covering 12 tasks across factual recall,
commonsense reasoning, reading comprehension, sentiment/semantic tasks,
and ethical alignment.

This repository accompanies the paper:

> **FIN-bench-v2 Evaluation: Comparing Frontier API Models and Local Inference
> on Finnish Benchmarks**  
> Janne Tompuri, University of Helsinki, 2026

Scored outputs for all 14 model configurations are available on
[Hugging Face Datasets](https://huggingface.co/datasets/jtompuri/finbench-eval-outputs)
(see [Data](#data)).

---

## Key results

| Model | Normalised score |
|---|---|
| Claude Opus 4.7 | **0.927** |
| Claude Opus 4.7 (CoT) | **0.927** |
| Gemini 3 Flash | 0.926 |
| Gemini 3.1 Pro | 0.914 |
| GPT-5.4 (CoT) | 0.907 |
| GPT-5.4 | 0.887 |
| Claude Sonnet 4.6 | 0.877 |
| Claude Sonnet 4.6 (CoT) | 0.874 |
| Gemma 4 26B *(best local)* | 0.826 |
| Gemma 3 27B | 0.802 |
| Gemma 4 E4B (CoT) | 0.749 |
| Gemma 4 E4B | 0.706 |
| Poro-8B | 0.605 |
| Llama 3.1 8B | 0.496 |

Scores are normalised: $(s - b)/(1 - b)$, where $b$ is the random-chance
baseline. Mean over 11 tasks (SQuAD excluded from aggregate).
Baselines v2 (2026-04-13); MCF-word extractor fix applied (2026-04-16).

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

# 3. Add API keys
cp .env.example .env
# Edit .env and fill in your keys
```

> **Choose your requirements file based on platform:**
> - **Frontier API models only:** `pip install -r requirements.txt`
> - **Apple Silicon (MLX):** `pip install -r requirements-mlx.txt`
> - **NVIDIA / AMD / CPU (llama.cpp):** install llama-cpp-python first, then
>   `pip install -r requirements-cuda.txt`. See the [llama.cpp section](#local-models--llamacpp--gguf-apple-silicon--nvidia--amd--cpu) below.
> - **Linux + NVIDIA (vLLM):** `sudo apt install python3-dev` then
>   `pip install -r requirements-vllm.txt`. See the [vLLM section](#local-models--vllm-linux--nvidia-cuda) below.

---

### Frontier models (API)

```bash
pip install -r requirements.txt
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
| `anthropic-adaptive-thinking` | Anthropic adaptive thinking for Claude Opus 4.7+ (CoT) |
| `google` | Google Gemini via AI Studio |
| `openrouter` | OpenRouter: unified access to third-party models |

Extended thinking (`anthropic-thinking`) is not supported on Claude Opus 4.7+.
Use `anthropic-adaptive-thinking` instead, which uses
`thinking={"type": "adaptive"}` and supports `temperature=0.0`.

Batch submission (50 % cost reduction, ~24 h turnaround) is supported for
`openai`, `anthropic`, and `anthropic-adaptive-thinking`:

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

---

### Local models: Apple Silicon (MLX)

`run_eval_jsonl.py` uses [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms)
and requires an Apple M-series chip. Models are downloaded automatically from
Hugging Face on first use.

> **Platform:** macOS with Apple Silicon (M1 or later) only.

```bash
pip install -r requirements-mlx.txt
```

```bash
# Smoke test: 5 items with live accuracy output
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

---

### Local models: llama.cpp / GGUF (Apple Silicon · NVIDIA · AMD · CPU)

`run_llama_jsonl.py` uses [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
and supports any platform that can run GGUF models.

#### Step 1: Platform prerequisites

**macOS:** llama-cpp-python requires **Python 3.14+**. Python 3.13 (conda /
miniconda) causes a silent crash at model load time. Create a fresh virtual
environment with Python 3.14 instead of the one from Quickstart:

```bash
brew install python@3.14
python3.14 -m venv .venv
source .venv/bin/activate
```

**Linux (Ubuntu + NVIDIA):** Install build tools and verify the GPU is visible:

```bash
sudo apt-get install -y cmake build-essential nvidia-cuda-toolkit
nvcc --version   # confirm CUDA is available
nvidia-smi       # confirm GPU is visible
```

#### Step 2: Install llama-cpp-python

Install llama-cpp-python **before** `pip install -r requirements-cuda.txt`.

**Apple Silicon (Metal):**

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-binary llama-cpp-python
```

**NVIDIA (CUDA, recommended: pre-built wheel, no compiler needed):**

```bash
# Check your CUDA version first
nvcc --version

# Install the matching wheel (cu121=12.1, cu122=12.2, cu123=12.3, cu124=12.4, cu125=12.5)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

Or build from source if a pre-built wheel is not available for your CUDA version:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-binary llama-cpp-python
```

**AMD (Vulkan):**

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-binary llama-cpp-python
```

**CPU only:**

```bash
pip install llama-cpp-python
```

#### Step 3: Install dependencies

```bash
pip install -r requirements-cuda.txt
```

#### Step 4: Download a GGUF model

```python
from huggingface_hub import hf_hub_download

# Gemma 4 E4B (used in this study, ~5 GB)
hf_hub_download(
    repo_id="bartowski/google_gemma-4-E4B-it-GGUF",
    filename="google_gemma-4-E4B-it-Q4_K_M.gguf",
    local_dir="./models",
)

# Gemma 3 27B (~17 GB)
hf_hub_download(
    repo_id="bartowski/gemma-3-27b-it-GGUF",
    filename="gemma-3-27b-it-Q4_K_M.gguf",
    local_dir="./models",
)
```

#### Step 5: Verify GPU support

Load a model and confirm GPU offloading in the output:

```bash
python -c "
from llama_cpp import Llama
Llama(model_path='models/google_gemma-4-E4B-it-Q4_K_M.gguf', n_gpu_layers=-1, verbose=True)
" 2>&1 | grep -iE "cuda|gpu|metal|offload|device"
```

If GPU is active you will see lines such as `llm_load_tensors: using CUDA`.

#### Step 6: Run evaluation

```bash
# Smoke test: 5 items with live accuracy output
python scripts/run_llama_jsonl.py \
    --model models/google_gemma-4-E4B-it-Q4_K_M.gguf \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/test_gemma4e4b_llama.jsonl \
    --n 5 --verbose

# Full run, GPU (all layers offloaded by default)
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

---

### Local models: vLLM (Linux + NVIDIA CUDA)

`run_vllm_jsonl.py` uses [vLLM](https://github.com/vllm-project/vllm) and loads
models directly from Hugging Face; no GGUF conversion is needed. vLLM's
PagedAttention allocates KV cache dynamically, making it faster than llama.cpp
for batched inference.

> **Platform:** Linux with NVIDIA CUDA only. Not supported on macOS or AMD.

System prerequisite (Python C headers for Triton runtime kernel compilation):

```bash
sudo apt install -y python3-dev
```

Then install vLLM and dependencies:

```bash
pip install -r requirements-vllm.txt
```

> `pip` will print a dependency conflict warning because vllm pins
> `transformers<5` but `requirements-vllm.txt` requests `transformers>=5.0.0`
> for Gemma 4 support. The warning is harmless; vllm imports and runs
> successfully with transformers 5.x.

```bash
# Smoke test: 5 items with live accuracy output
python scripts/run_vllm_jsonl.py \
    --model google/gemma-4-e4b-it \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/test_gemma4e4b_vllm.jsonl \
    --n 5 --verbose

# Full batch (recommended for runs ≲10 min wall clock).
# All prompts in one generate() call. Maximum throughput, no intermediate saves.
python scripts/run_vllm_jsonl.py \
    --model google/gemma-4-e4b-it \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/combined_gemma4e4b_vllm.jsonl \
    --batch

# Long or unstable runs: chunked batch with resume safety.
# Flushes after each chunk; supports --resume on interruption.
# At most N items lost if the run is interrupted.
python scripts/run_vllm_jsonl.py \
    --model google/gemma-4-e4b-it \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/combined_gemma4e4b_vllm.jsonl \
    --batch-size 200 --resume

# Multi-GPU (tensor parallelism)
python scripts/run_vllm_jsonl.py \
    --model google/gemma-4-e4b-it \
    --input data/finbench_combined_v1.jsonl \
    --output outputs/combined_gemma4e4b_vllm.jsonl \
    --batch \
    --tensor-parallel-size 2
```

Three inference modes (mutually exclusive):

| Mode | Flag | Flush | Resume | Use when |
|---|---|---|---|---|
| Full batch | `--batch` | at the end | ✗ | **recommended for runs ≲10 min wall clock** |
| Chunked batch | `--batch-size N` | after every N items | ✓ | long runs (>30 min), unstable hardware, or when you need progress visibility |
| Sequential | *(default)* | after every item | ✓ | development, smoke tests with `--n` |

#### Empirical scaling (RTX 3090, 1146 items, BF16)

Wall-clock time per batch size (engine init ~60 s constant; remainder is generation):

| `--batch-size` | Gemma 4 E4B | Llama 3.1 8B |
|---:|---:|---:|
| 50 | 254 s | 193 s |
| 100 | 199 s | 148 s |
| 200 | 163 s | 131 s |
| 500 | 154 s | 109 s |
| `--batch` (1146) | **147 s** | **94 s** |

Both models reach maximum throughput at `--batch`. Gemma 4 E4B saturates around
batch 200 (≤7 % gain beyond); Llama 3.1 8B keeps scaling and benefits most
from full batch (35 % faster than `--batch-size 200`). The chunked-batch
overhead is both model- and dataset-dependent, but for the canonical
1146-item benchmark, full batch is uniformly fastest. Use
`--batch-size N --resume` only when interruption recovery matters more than
throughput.

Key options:

| Flag | Default | Description |
|---|---|---|
| `--batch-size N` | — | Prompts per `generate()` call; flush after each chunk (recommended: 50–200) |
| `--batch` | off | All prompts in one `generate()` call; no intermediate saves |
| `--tensor-parallel-size N` | 1 | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization F` | 0.90 | Fraction of GPU memory for model + KV cache |
| `--max-model-len N` | 8192 | Maximum sequence length (prompt + response) |
| `--enable-thinking` | off | Activate chain-of-thought for Gemma 4 and compatible models |
| `--trust-remote-code` | off | Allow executing code from the model repository |
| `--quantization SCHEME` | none | On-the-fly quantization: `bitsandbytes`, `awq`, `gptq`, `fp8`, `compressed-tensors`. Default loads native precision. `bitsandbytes` requires `pip install bitsandbytes`; lower `--gpu-memory-utilization` to 0.80 to fit the BF16→4-bit conversion peak. |

---

### Scoring

```bash
python scripts/score_eval.py \
    --input outputs/combined_gemma4e4b_llama.jsonl \
    --output results/raw/score_gemma4e4b_llama_combined.json
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

See [`docs/repository_structure.md`](docs/repository_structure.md) for the full
script and data directory reference.

---

## Methodology notes

- **Generative scoring** throughout (no log-probabilities required),
  enabling evaluation of closed frontier APIs.
- Results are **not directly comparable** to official FIN-bench-v2
  log-probability scores.
- Random-chance baselines are task-specific; see `scripts/eval_config.py`.
- SQuAD FI uses token-level F1 and is excluded from the normalised aggregate.
  Multi-metric analysis (Voikko Lemma-F1, chrF, Finnish BERTScore) shows that
  verbosity, not morphology alone, is the primary cause of the cross-tier
  inversion: frontier models produce full grammatically inflected answers while
  gold spans are short extracted fragments. Finnish BERTScore
  (TurkuNLP/bert-base-finnish-cased-v1) gives the most faithful re-ranking,
  with frontier models gaining +0.30–0.47 and local models +0.30–0.43 over
  token-F1. Voikko lemmatisation alone adds only +0.016 for the most affected
  model (Claude Sonnet 4.6).

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

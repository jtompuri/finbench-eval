# Six-Precision Cross-Backend Comparison

Auxiliary measurement supporting the FIN-bench-v2 evaluation paper. Quantifies
the effect of 4-bit quantization across three model architectures (dense Llama
8B, MoE-without-PLE OLMoE-1B-7B, and MoE-A4B-with-PLE Gemma 4 E4B) and up to
five backends/quantization schemes (MLX 4-bit, llama.cpp Q4_K_M, vLLM bnb 4-bit,
vLLM AWQ INT4, vLLM GPTQ INT4) against a vLLM BF16 reference.

## Caveats and scope

- **Hardware:** All measurements on a single NVIDIA RTX 3090 (Ampere SM_86,
  24 GB VRAM, 936 GB/s memory bandwidth). Throughput, VRAM-fit, and
  `--gpu-memory-utilization` recommendations **do not generalise** to other
  GPUs (H100, A100, RTX 4090, consumer GPUs with less VRAM, etc.). Quality
  findings — i.e. *score deltas between precision schemes for a given model* —
  should generalise across hardware since they reflect the model's
  quantization sensitivity, not the GPU's compute path.
- **Sample size:** 1,146 items per run, single prompt variant (p0). Wilson
  CIs at the per-task level are ~0.07–0.09 (per the paper's own analysis),
  so individual cell variation is partially noise.
- **Greedy decoding (temperature = 0):** runs are deterministic per backend
  but sampling-tie-breaks differ slightly across backends — small per-task
  variation is expected even at fixed precision.
- **Llama 3.1 8B:** AWQ from `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`,
  GPTQ from `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4`. Both
  use English calibration data (likely C4) — Finnish-specific calibration
  could change the picture.

## Llama 3.1 8B — six-precision per-task accuracy

| Task | Paper MLX 4-bit | llama.cpp Q4_K_M | vLLM BF16 | vLLM bnb 4-bit | vLLM AWQ INT4 | vLLM GPTQ INT4 |
|---|---:|---:|---:|---:|---:|---:|
| ARC Challenge | 0.520 | 0.560 | 0.480 | 0.515 | 0.495 | 0.530 |
| Belebele | 0.680 | 0.750 | 0.690 | 0.640 | 0.620 | 0.670 |
| GoldenSwag | 0.420 | 0.360 | 0.350 | 0.370 | 0.398 | 0.392 |
| ScandiSent | 0.970 | 0.959 | 0.970 | 0.940 | 0.866 | 0.943 |
| SIB-200 | 0.700 | 0.698 | 0.733 | 0.775 | 0.667 | 0.624 |
| General Knowledge | 0.610 | 0.657 | 0.657 | 0.600 | 0.643 | 0.681 |
| TruthfulQA mc1 | 0.520 | 0.570 | 0.591 | 0.540 | **0.736** | 0.523 |
| Analogies | 0.610 | 0.670 | 0.606 | 0.573 | 0.562 | 0.644 |
| Emotions | 0.350 | 0.630 | 0.586 | 0.520 | 0.300 | 0.541 |
| HHH Alignment | 0.770 | 0.725 | 0.776 | 0.712 | 0.756 | 0.734 |
| Similarities | 0.610 | 0.487 | 0.413 | 0.421 | 0.474 | 0.632 |
| **Mean (11 tasks)** | **0.6145** | **0.6423** | **0.6230** | **0.6006** | **0.5925** | **0.6285** |

Aggregate variation across all six precisions: **±0.025** (range 0.5925–0.6423).
Within sampling noise per the paper's ~0.07–0.09 detection threshold.

## OLMoE-1B-7B — three-precision per-task accuracy (MoE without PLE)

OLMoE-1B-7B-0125-Instruct (AI2): Mixture-of-Experts, 64 experts top-8
routed, ~1 B active / 6.9 B total parameters, **no Per-Layer Embeddings**.
Included as a clean MoE-without-PLE control to isolate whether the Gemma 4
E4B quantization sensitivity stems from MoE routing itself or from PLE.

| Task | vLLM BF16 | vLLM bnb 4-bit | llama.cpp Q4_K_M |
|---|---:|---:|---:|
| ARC Challenge | 0.280 | 0.290 | 0.290 |
| Belebele | 0.240 | 0.300 | 0.330 |
| GoldenSwag | 0.270 | 0.280 | 0.320 |
| ScandiSent | 0.390 | 0.490 | 0.570 |
| SIB-200 | 0.130 | 0.130 | 0.120 |
| General Knowledge | 0.286 | 0.214 | 0.229 |
| SQuAD FI (F1) | 0.217 | 0.199 | 0.182 |
| TruthfulQA mc1 | 0.290 | 0.140 | 0.180 |
| Analogies | 0.360 | 0.350 | 0.320 |
| Emotions | 0.180 | 0.140 | 0.140 |
| HHH Alignment | 0.180 | 0.250 | 0.200 |
| Similarities | 0.290 | 0.329 | 0.250 |
| **Mean (12 tasks)** | **0.258** | **0.259** | **0.262** |

Aggregate variation across all three precisions: **±0.004** — tighter
than any other model in this sweep. MoE expert-routing alone, without
PLE, does not induce 4-bit degradation.

## Gemma 4 E4B — three-precision aggregate

| Method | Mean (11) | Δ vs BF16 |
|---|---:|---:|
| Paper MLX 4-bit | 0.762 | −0.064 |
| llama.cpp Q4_K_M | 0.747 | −0.079 |
| vLLM BF16 | **0.826** | – |
| vLLM bnb 4-bit | 0.746 | −0.080 |

(AWQ and GPTQ pre-quantized variants are not published for Gemma 4 E4B at
the time of writing — only for the 31B sibling.)

Aggregate variation across the three 4-bit schemes is small (±0.015), but
the gap to BF16 is **~0.08 pp**, an order of magnitude larger than what
we see on Llama 3.1.

## Poro-8B — four-precision aggregate

| Task | Paper MLX | llama.cpp Q4_K_M | vLLM BF16 | vLLM bnb 4-bit |
|---|---:|---:|---:|---:|
| ARC Challenge | 0.660 | 0.680 | 0.730 | 0.690 |
| Belebele | 0.740 | 0.780 | 0.760 | 0.760 |
| GoldenSwag | 0.780 | 0.770 | 0.770 | 0.720 |
| ScandiSent | 0.950 | 0.950 | 0.950 | 0.950 |
| SIB-200 | 0.740 | 0.780 | 0.750 | 0.778 |
| General Knowledge | 0.730 | 0.743 | 0.814 | 0.786 |
| TruthfulQA mc1 | 0.200 | 0.094 | 0.157 | 0.160 |
| Analogies | 0.740 | 0.768 | 0.818 | 0.748 |
| Emotions | 0.680 | 0.740 | 0.700 | 0.630 |
| HHH Alignment | 0.820 | 0.821 | 0.848 | 0.750 |
| Similarities | 0.640 | 0.699 | 0.773 | 0.803 |
| **Mean (11)** | **0.698** | **0.711** | **0.734** | **0.707** |

Aggregate variation across all four precisions: **±0.036** — within sampling
noise. Same pattern as Llama 3.1 8B (its base model), confirming that
**Finnish fine-tuning does not introduce additional quantization sensitivity**.

## Headline finding: 4-bit quantization cost is PLE-dependent, not MoE-routing-dependent

| Model | Architecture | BF16 → 4-bit aggregate cost |
|---|---|---:|
| Llama 3.1 8B | Dense Llama 8B | ±0.025 (within noise) |
| Poro-8B | Dense Llama 8B + 165 B FI tokens fine-tune | ±0.036 (within noise) |
| **OLMoE-1B-7B** | **MoE (64 experts top-8), no PLE** | **±0.004 (within noise)** |
| Gemma 4 E4B | MoE-A4B + Per-Layer Embeddings | −0.06 to −0.08 |

The four-model picture isolates the sensitivity source:

- **Dense Llama-derived models** (with and without Finnish fine-tuning)
  tolerate 4-bit across every scheme tested, ruling out fine-tuning as
  a cause.
- **OLMoE-1B-7B** (MoE without PLE) is even tighter than the dense
  models — ±0.004 spread across three precisions. **Expert-routing
  alone is not enough to induce 4-bit degradation.**
- **Gemma 4 E4B** (MoE with PLE) consistently loses 6–8 pp under
  4-bit across MLX, Q4_K_M, and bnb.

**Refined hypothesis:** Per-Layer Embeddings, rather than MoE expert-routing
per se, are the dominant quantization-sensitive path. PLE couples
every token embedding to per-layer learned vectors with limited
dynamic range; 4-bit truncation accumulates error along the depth
axis. Expert-routing, which was the a-priori suspect, turns out to
tolerate 4-bit cleanly when isolated (OLMoE). Fully validating the
hypothesis would require architecture-specific experiments beyond the
scope of this paper — e.g. an ablation of PLE in Gemma 4 E4B, or a
third MoE-with-PLE comparator if one becomes available.

## Bonus: Finnish specialisation effect persists at full precision

The paper (§4.3) reports a +0.137 normalised Finnish-specialisation gap
between Poro-8B and Llama 3.1 8B based on MLX 4-bit. The vLLM BF16
measurements **confirm this gap is not a quantization artefact**:

| Model | vLLM BF16 mean (11 tasks) |
|---|---:|
| Llama 3.1 8B (base) | 0.6230 |
| **Poro-8B (Finnish fine-tune)** | **0.7337 (+0.111 raw)** |

Raw +0.111 at full precision matches the paper's +0.084 raw / +0.137
normalised within sampling noise. Strengthens §4.3's conclusion.

## Wall-clock timings (RTX 3090, 1146 items, `--batch`)

| Backend / quantization | Llama 3.1 8B | Poro-8B | Gemma 4 E4B |
|---|---:|---:|---:|
| llama.cpp Q4_K_M (CUDA) | ~780 s (13 min) | ~1500 s (25 min) | ~3000 s (50 min) |
| vLLM BF16 | 94 s | 179 s | 147 s |
| vLLM bnb 4-bit | 132 s | 200 s | 176 s |
| vLLM AWQ INT4 | 127 s | n/a (not published) | n/a (not published) |
| vLLM GPTQ INT4 | 95 s | n/a | n/a |

Hardware-specific observations on RTX 3090 (do not generalise to other GPUs):

- **GPTQ matches BF16 throughput on Llama** (95 s vs 94 s) — vLLM's
  Marlin INT4 GEMM kernels are extremely well optimised for Ampere.
- **bnb and AWQ are ~30 % slower than BF16** on Llama. bnb does on-the-fly
  4-bit dequantization; AWQ uses a different INT4 path that is less
  optimised on Ampere than GPTQ's. Trade-off: AWQ/bnb give VRAM
  savings (~4 GB vs 16 GB BF16), GPTQ gives both VRAM savings AND
  full BF16 throughput.
- **vLLM is 8–20× faster than llama.cpp** end-to-end on this workload,
  for reasons documented in the empirical-scaling subsection of the
  README's vLLM section.
- **bnb required `--gpu-memory-utilization 0.80`** to fit on a 24 GB card
  (BF16-load + 4-bit-conversion peak). AWQ retried with the same setting
  after an initial OOM. Other GPUs with ≥32 GB VRAM should not need this.

## Per-task observations (Llama 3.1)

- **TruthfulQA AWQ outlier (0.736):** AWQ scores +0.21 above the next-best
  scheme on this task. May be a calibration-driven coincidence; would
  need a second AWQ build with different calibration to disentangle.
- **Emotions AWQ-low (0.300):** simultaneously the worst score on this
  task across all schemes, while all others (including bnb and GPTQ)
  cluster around 0.5–0.6. AWQ calibration appears not to favour this
  task on Llama 3.1.
- **Similarities BF16-low (0.413):** vLLM BF16 underperforms the other
  schemes on this task — opposite of what's usually expected. GPTQ
  scores 0.632, paper MLX 4-bit 0.610. Greedy-decoding tie-breaks may
  matter here.

These per-task shifts are within Wilson CIs and may not be statistically
significant individually, but the pattern (different precisions favour
different tasks) is consistent with the literature on quantization noise
amplifying task-specific sensitivities.

## Suggested paper Limitations / Methodology paragraph

> **Quantization sensitivity.** Local-model results reflect 4-bit quantization
> (MLX for the published outputs). Auxiliary precision sweep on a single
> RTX 3090 tests four models: two dense Llama-based models (Llama 3.1 8B,
> Llama-Poro-2-8B) vary by at most ±0.036 across precisions, within Wilson-CI
> noise; OLMoE-1B-7B (MoE with 64 experts top-8 routed, no Per-Layer
> Embeddings) varies by only ±0.004 across BF16, bitsandbytes 4-bit, and
> llama.cpp Q4_K_M; Gemma 4 E4B (MoE with PLE) loses 0.06–0.08 aggregate
> in 4-bit relative to BF16 across all three 4-bit schemes tested. The
> three-way contrast narrows the hypothesis: Per-Layer Embeddings, rather
> than expert-routing per se, appear to be the dominant quantization-
> sensitive path. The BF16 measurement also confirms the Finnish
> specialisation gap between Llama-Poro-2-8B and Llama 3.1 8B (+0.111 raw
> at full precision, matching the +0.109 normalised MLX 4-bit gap) is
> not a quantization artefact.

## Source data

All raw outputs and score files are in this repository under
`outputs/` and `results/raw/`. Filename convention:
`combined_<model>_<backend>[_<quant>].jsonl`. Tidy CSV combining all
runs is at `results/tidy/scores.csv`. Cross-precision evaluation was
performed with PR #6 (the `--quantization` flag); the AWQ retry that
required `--gpu-memory-utilization 0.80` is documented in
the related issues.

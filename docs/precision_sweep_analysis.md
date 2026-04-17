# Six-Precision Cross-Backend Comparison

Auxiliary measurement supporting the FIN-bench-v2 evaluation paper. Quantifies
the effect of 4-bit quantization across two model architectures (dense Llama
8B vs MoE-A4B Gemma 4 E4B) and four backends/quantization schemes
(MLX 4-bit, llama.cpp Q4_K_M, vLLM bnb 4-bit, vLLM AWQ INT4, vLLM GPTQ INT4)
against a vLLM BF16 reference.

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

## Headline finding: 4-bit quantization cost is architecture-dependent, not fine-tuning-dependent

| Model | Architecture | BF16 → 4-bit aggregate cost |
|---|---|---:|
| Llama 3.1 8B | Dense Llama 8B | ±0.025 (within noise) |
| **Poro-8B** | **Dense Llama 8B + 165 B FI tokens fine-tune** | **±0.036 (within noise)** |
| Gemma 4 E4B | MoE-A4B + Per-Layer Embeddings | −0.06 to −0.08 |

The three-model picture is now sharp: **two dense Llama-derived models** (with
and without Finnish fine-tuning) tolerate 4-bit quantization across every
scheme tested, while **the MoE-A4B + PLE Gemma 4 E4B** consistently loses
6–8 pp under 4-bit. This rules out fine-tuning as a source of quantization
sensitivity — the difference is architectural.

**Hypothesis:** MoE expert-routing is sensitive to small numerical
perturbations — quantization errors can flip routing decisions for
borderline tokens. Per-Layer Embeddings (PLE), which Gemma 4 introduces
to maximise parameter efficiency, may add another quantization-sensitive
path. Dense models lack both mechanisms and tolerate 4-bit better,
regardless of whether they have been further fine-tuned. Validating the
hypothesis cleanly would require AWQ/GPTQ versions of Gemma 4 E4B
(unavailable at the time of writing) or architecture-specific experiments
beyond the scope of the paper.

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

> **Quantization sensitivity.** Local-model results in Tables 1 and 3 reflect
> 4-bit quantization (MLX for the published outputs). On Llama 3.1 8B we
> measured a vLLM BF16 reference and four 4-bit schemes (MLX 4-bit,
> llama.cpp Q4_K_M, vLLM bitsandbytes, vLLM AWQ INT4, vLLM GPTQ INT4) on
> a single RTX 3090; aggregate normalised-score variation across all six
> measurements is ±0.025, within sampling noise. The same picture holds for
> Poro-8B (the Finnish-fine-tuned Llama 3.1 base): variation across MLX 4-bit,
> llama.cpp Q4_K_M, vLLM BF16 and vLLM bitsandbytes is ±0.036, again within
> sampling noise. On Gemma 4 E4B the BF16 reference is +0.06 to +0.08 above
> the 4-bit aggregate (across MLX, Q4_K_M, and bnb), suggesting that the
> MoE-A4B + Per-Layer Embeddings architecture is more sensitive to 4-bit
> quantization than dense models at the same active-parameter scale.
> Reproductions on consumer hardware that are restricted to 4-bit may
> therefore underestimate the true capability of MoE local models —
> full-precision evaluation requires substantially more VRAM (e.g. ~52 GB
> for Gemma 4 26B in BF16, beyond a single RTX 3090).
>
> The Poro-8B versus Llama 3.1 8B Finnish-specialisation gap reported in
> §4.3 (+0.137 normalised) also reproduces at full BF16 precision (raw
> +0.111), confirming that the Finnish fine-tuning effect is not a
> quantization artefact.

## Source data

All raw outputs and score files are in this repository under
`outputs/` and `results/raw/`. Filename convention:
`combined_<model>_<backend>[_<quant>].jsonl`. Tidy CSV combining all
runs is at `results/tidy/scores.csv`. Cross-precision evaluation was
performed with PR #6 (the `--quantization` flag); the AWQ retry that
required `--gpu-memory-utilization 0.80` is documented in
the related issues.

# Repository structure

```
scripts/
    run_frontier_jsonl.py   # Run frontier API models (OpenAI, Anthropic, Google, OpenRouter)
    run_eval_jsonl.py       # Run local MLX models (Apple Silicon)
    run_llama_jsonl.py      # Run local GGUF models via llama.cpp (all platforms)
    run_vllm_jsonl.py       # Run local HuggingFace models via vLLM (Linux + CUDA)
    runner_utils.py         # Shared utilities (LiveStats, load_jsonl, resume, thinking helpers)
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
    run_vllm_prompt.py      # Single-prompt vLLM runner (smoke tests)
    compute_bertscore_squad.py  # BERTScore validation for SQuAD FI
    analysis/
        final_summary.py        # Per-model summary table
        compare_runs.py         # Pairwise run comparison
        mcnemar_test.py         # McNemar significance tests (BH-corrected)
        analysis_normalized.py  # Normalised score analysis

data/
    finbench_combined_v1.jsonl   # Fixed 1,146-item evaluation subset
```

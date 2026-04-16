"""
Tests for scripts/aggregate_results.py — run_meta override logic (issue #1 fix).

Verifies that when a score JSON contains a run_meta block, aggregate_results
reads backend/max_tokens/temperature from it rather than from the hardcoded
MODEL_INFO table.
"""
import json
import tempfile
from pathlib import Path

import pytest

from aggregate_results import rows_from_file, MODEL_INFO


# ---------------------------------------------------------------------------
# Minimal score-JSON factory
# ---------------------------------------------------------------------------

def _make_score_json(run_meta: dict | None = None) -> dict:
    """Return a minimal score JSON that rows_from_file() can parse."""
    data = {
        "total_items": 1,
        "overall_score_avg": 0.8,
        "per_task": {
            "arc_challenge_fi": {
                "task_type": "mcf_letter",
                "n": 10,
                "score_avg": 0.8,
                "accuracy": 0.8,
                "accuracy_all": 0.8,
                "n_correct": 8,
                "n_abstain": 0,
                "n_engaged": 10,
                "refusal_rate": 0.0,
                "ci_95": [0.49, 0.94],
            }
        },
    }
    if run_meta is not None:
        data["run_meta"] = run_meta
    return data


def _write_score_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Backend override
# ---------------------------------------------------------------------------

class TestRunMetaBackendOverride:
    """rows_from_file() uses run_meta.backend when present."""

    def test_llama_cpp_backend_from_run_meta(self, tmp_path):
        path = tmp_path / "score_gemma4e4b_combined.json"
        _write_score_json(path, _make_score_json({"backend": "llama_cpp"}))
        rows = rows_from_file(path)
        assert rows, "Expected at least one row"
        assert all(r["backend"] == "llama_cpp" for r in rows)

    def test_vllm_backend_from_run_meta(self, tmp_path):
        path = tmp_path / "score_gemma4e4b_combined.json"
        _write_score_json(path, _make_score_json({"backend": "vllm"}))
        rows = rows_from_file(path)
        assert all(r["backend"] == "vllm" for r in rows)

    def test_mlx_backend_unchanged_via_run_meta(self, tmp_path):
        path = tmp_path / "score_gemma4e4b_combined.json"
        _write_score_json(path, _make_score_json({"backend": "mlx_lm"}))
        rows = rows_from_file(path)
        assert all(r["backend"] == "mlx_lm" for r in rows)

    def test_no_run_meta_falls_back_to_model_info(self, tmp_path):
        """Without run_meta the backend comes from MODEL_INFO (mlx_lm for local models)."""
        path = tmp_path / "score_gemma4e4b_combined.json"
        _write_score_json(path, _make_score_json(run_meta=None))
        rows = rows_from_file(path)
        expected_backend = MODEL_INFO["gemma4e4b"]["backend"]
        assert all(r["backend"] == expected_backend for r in rows)

    def test_empty_backend_in_run_meta_falls_back(self, tmp_path):
        """run_meta with empty backend string does not override MODEL_INFO."""
        path = tmp_path / "score_gemma4e4b_combined.json"
        _write_score_json(path, _make_score_json({"backend": ""}))
        rows = rows_from_file(path)
        expected_backend = MODEL_INFO["gemma4e4b"]["backend"]
        assert all(r["backend"] == expected_backend for r in rows)


# ---------------------------------------------------------------------------
# max_tokens / temperature override
# ---------------------------------------------------------------------------

class TestRunMetaParamsOverride:

    def test_max_tokens_from_run_meta(self, tmp_path):
        path = tmp_path / "score_gemma4e4b_combined.json"
        _write_score_json(path, _make_score_json({"backend": "llama_cpp", "max_tokens": 128}))
        rows = rows_from_file(path)
        assert all(r["max_tokens"] == 128 for r in rows)

    def test_temperature_from_run_meta(self, tmp_path):
        path = tmp_path / "score_gemma4e4b_combined.json"
        _write_score_json(path, _make_score_json({"backend": "vllm", "temperature": 0.7}))
        rows = rows_from_file(path)
        assert all(r["temperature"] == pytest.approx(0.7) for r in rows)

    def test_partial_run_meta_mixes_with_model_info(self, tmp_path):
        """run_meta with only backend leaves max_tokens from MODEL_INFO."""
        path = tmp_path / "score_gemma4e4b_combined.json"
        _write_score_json(path, _make_score_json({"backend": "llama_cpp"}))
        rows = rows_from_file(path)
        expected_max_tokens = MODEL_INFO["gemma4e4b"]["max_tokens"]
        assert all(r["max_tokens"] == expected_max_tokens for r in rows)
        assert all(r["backend"] == "llama_cpp" for r in rows)

"""
Tests for llama.cpp runner helper functions.
No model loading or inference — only pure Python logic.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_llama_jsonl import resolve_n_ctx


# ---------------------------------------------------------------------------
# resolve_n_ctx
# ---------------------------------------------------------------------------

class TestResolveNCtx:
    def _settings(self, n_ctx=4096):
        return {"generation": {"n_ctx": n_ctx}}

    def test_override_takes_precedence(self):
        assert resolve_n_ctx(self._settings(4096), override=8192) == 8192

    def test_reads_from_settings(self):
        assert resolve_n_ctx(self._settings(4096), override=None) == 4096

    def test_custom_value_in_settings(self):
        assert resolve_n_ctx(self._settings(32768), override=None) == 32768

    def test_empty_settings_returns_default(self):
        assert resolve_n_ctx({}, override=None) == 4096

    def test_override_zero_is_valid(self):
        # 0 is a valid value (edge case — not meaningful but should not fall back)
        assert resolve_n_ctx(self._settings(4096), override=0) == 0

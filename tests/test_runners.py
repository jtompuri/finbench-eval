"""
Tests for runner helper functions in run_llama_jsonl.py and run_eval_jsonl.py.
No model loading or inference — only pure Python logic.
"""
import pytest
from run_llama_jsonl import LiveStats, resolve_max_tokens


# ---------------------------------------------------------------------------
# resolve_max_tokens
# ---------------------------------------------------------------------------

class TestResolveMaxTokens:
    def _settings(self):
        return {
            "generation": {
                "max_tokens": {
                    "default": 512,
                    "gemma4": 1024,
                    "poro8b": 512,
                }
            }
        }

    def test_override_takes_precedence(self):
        assert resolve_max_tokens(self._settings(), "gemma4", override=256) == 256

    def test_model_key_lookup(self):
        assert resolve_max_tokens(self._settings(), "gemma4", override=None) == 1024

    def test_default_for_unknown_key(self):
        assert resolve_max_tokens(self._settings(), "unknown_model", override=None) == 512

    def test_default_when_no_key(self):
        assert resolve_max_tokens(self._settings(), None, override=None) == 512

    def test_integer_max_tokens_config(self):
        # max_tokens can also be a flat integer (not a dict)
        settings = {"generation": {"max_tokens": 768}}
        assert resolve_max_tokens(settings, "gemma4", override=None) == 768

    def test_empty_settings_returns_512(self):
        assert resolve_max_tokens({}, None, override=None) == 512


# ---------------------------------------------------------------------------
# LiveStats
# ---------------------------------------------------------------------------

class TestLiveStats:
    def _mcf_item(self, task, expected, task_type="mcf_letter"):
        return {
            "task": task,
            "task_type": task_type,
            "expected": expected,
            "expected_choices": ["positiivinen", "negatiivinen", "neutraali"],
        }

    def test_correct_mcf_letter(self):
        stats = LiveStats()
        result = stats.update(self._mcf_item("arc_challenge_fi", "A"), "Vastaus: A")
        assert "✓" in result
        assert stats.by_task["arc_challenge_fi"]["correct"] == 1

    def test_wrong_mcf_letter(self):
        stats = LiveStats()
        result = stats.update(self._mcf_item("arc_challenge_fi", "A"), "Vastaus: B")
        assert "✗" in result
        assert stats.by_task["arc_challenge_fi"]["correct"] == 0

    def test_correct_mcf_word(self):
        stats = LiveStats()
        item = self._mcf_item("scandisent_fi", "positiivinen", task_type="mcf_word")
        result = stats.update(item, "positiivinen")
        assert "✓" in result

    def test_gen_shows_snippet(self):
        stats = LiveStats()
        item = {"task": "squad_fi", "task_type": "gen", "expected": "Helsinki"}
        result = stats.update(item, "Helsinki on Suomen pääkaupunki")
        assert "→" in result

    def test_accumulates_across_items(self):
        stats = LiveStats()
        for _ in range(3):
            stats.update(self._mcf_item("arc_challenge_fi", "A"), "A")
        stats.update(self._mcf_item("arc_challenge_fi", "A"), "B")
        assert stats.by_task["arc_challenge_fi"]["n"] == 4
        assert stats.by_task["arc_challenge_fi"]["correct"] == 3

    def test_multiple_tasks(self):
        stats = LiveStats()
        stats.update(self._mcf_item("arc_challenge_fi", "A"), "A")
        stats.update(self._mcf_item("belebele_fin", "B"), "B")
        assert len(stats.by_task) == 2

    def test_summary_lines_format(self):
        stats = LiveStats()
        stats.update(self._mcf_item("arc_challenge_fi", "A"), "A")
        lines = stats.summary_lines()
        assert any("arc_challenge_fi" in line for line in lines)
        assert any("MCF overall" in line for line in lines)

    def test_summary_gen_task_no_accuracy(self):
        stats = LiveStats()
        item = {"task": "squad_fi", "task_type": "gen", "expected": "X"}
        stats.update(item, "X")
        lines = stats.summary_lines()
        assert any("gen" in line for line in lines)

"""
Tests for run_llama_prompt.py helper functions.
No model loading or inference — only pure Python logic.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_llama_prompt import _THINKING_DELIMITER, _THINKING_PREFIX


# ---------------------------------------------------------------------------
# BOS stripping in apply_chat_template
#
# The function strips a leading <bos> from the rendered template output
# to avoid a duplicate BOS warning when llama-cpp-python adds its own.
# We test the stripping logic directly without loading a model.
# ---------------------------------------------------------------------------

class TestBosStripping:
    """Verify the <bos> stripping logic used inside apply_chat_template."""

    def _strip(self, text: str) -> str:
        """Replicate the stripping logic from apply_chat_template."""
        if text.startswith("<bos>"):
            return text[len("<bos>"):]
        return text

    def test_leading_bos_stripped(self):
        assert self._strip("<bos><start_of_turn>user\nhello") == "<start_of_turn>user\nhello"

    def test_no_bos_unchanged(self):
        assert self._strip("<start_of_turn>user\nhello") == "<start_of_turn>user\nhello"

    def test_bos_only(self):
        assert self._strip("<bos>") == ""

    def test_bos_not_at_start_unchanged(self):
        text = "text<bos>more"
        assert self._strip(text) == text

    def test_empty_string_unchanged(self):
        assert self._strip("") == ""


# ---------------------------------------------------------------------------
# Thinking delimiter constants
# ---------------------------------------------------------------------------

class TestThinkingDelimiters:
    """Verify the thinking delimiter constants are set correctly."""

    def test_delimiter_value(self):
        assert _THINKING_DELIMITER == "<channel|>"

    def test_prefix_value(self):
        assert _THINKING_PREFIX == "<|channel>thought\n"

    def test_delimiter_splits_thinking_from_answer(self):
        raw = f"{_THINKING_PREFIX}some thoughts here{_THINKING_DELIMITER}final answer"
        parts = raw.split(_THINKING_DELIMITER, 1)
        thinking = parts[0].removeprefix(_THINKING_PREFIX).strip()
        answer = parts[1].strip()
        assert thinking == "some thoughts here"
        assert answer == "final answer"

    def test_no_delimiter_means_no_thinking(self):
        raw = "plain answer without thinking"
        assert _THINKING_DELIMITER not in raw

    def test_empty_thinking_block(self):
        raw = f"{_THINKING_PREFIX}{_THINKING_DELIMITER}answer"
        thinking = raw.split(_THINKING_DELIMITER, 1)[0].removeprefix(_THINKING_PREFIX).strip()
        assert thinking == ""

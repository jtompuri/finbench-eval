"""
Tests for run_llama_prompt.py helper functions.
No model loading or inference — only pure Python logic.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_llama_prompt import _THINKING_DELIMITER, _THINKING_PREFIX, _strip_stop_tokens, _STOP_TOKENS


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


# ---------------------------------------------------------------------------
# _strip_stop_tokens
#
# The unified code path uses llm() raw completion instead of
# create_chat_completion(), so we defensively strip trailing stop tokens
# that create_chat_completion() would have removed automatically.
# ---------------------------------------------------------------------------

class TestStripStopTokens:

    def test_clean_response_unchanged(self):
        assert _strip_stop_tokens("Vastaus on A.") == "Vastaus on A."

    def test_strips_end_of_turn(self):
        assert _strip_stop_tokens("Vastaus on A.<end_of_turn>") == "Vastaus on A."

    def test_strips_eos(self):
        assert _strip_stop_tokens("positiivinen<eos>") == "positiivinen"

    def test_strips_im_end(self):
        assert _strip_stop_tokens("answer<|im_end|>") == "answer"

    def test_strips_endoftext(self):
        assert _strip_stop_tokens("answer<|endoftext|>") == "answer"

    def test_strips_trailing_whitespace_before_stop_token(self):
        # strip() is called before token check, so whitespace before token is gone
        assert _strip_stop_tokens("answer <end_of_turn>") == "answer"

    def test_stop_token_in_middle_preserved(self):
        # Only trailing stop tokens are stripped — mid-text occurrences stay
        text = "choice A<end_of_turn> or choice B"
        assert _strip_stop_tokens(text) == text

    def test_empty_string(self):
        assert _strip_stop_tokens("") == ""

    def test_only_stop_token(self):
        assert _strip_stop_tokens("<end_of_turn>") == ""

    def test_all_stop_tokens_covered(self):
        """Every token in _STOP_TOKENS is stripped when it appears at the end."""
        for tok in _STOP_TOKENS:
            assert _strip_stop_tokens(f"answer{tok}") == "answer", \
                f"_strip_stop_tokens failed to strip '{tok}'"

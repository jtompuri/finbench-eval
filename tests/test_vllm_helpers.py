"""
Tests for vLLM runner helper functions.
No model loading or inference — only pure Python logic.

vLLM is Linux/CUDA only, so it is not installed in the development
environment on macOS. Tests that exercise run_batch() inject a fake
vllm module into sys.modules so the import guard inside run_batch()
is satisfied without requiring vLLM to actually be installed.
"""
import math
import sys
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Inject a minimal vllm stub so run_vllm_prompt imports without error
# ---------------------------------------------------------------------------
if "vllm" not in sys.modules:
    _vllm_stub = types.ModuleType("vllm")

    class _SamplingParams:
        def __init__(self, **kwargs):
            pass

    _vllm_stub.SamplingParams = _SamplingParams
    _vllm_stub.LLM = MagicMock()
    sys.modules["vllm"] = _vllm_stub

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_vllm_prompt import _parse_output, _THINKING_DELIMITER, _THINKING_PREFIX, run_batch


# ---------------------------------------------------------------------------
# Shared mock LLM used across multiple test classes
# ---------------------------------------------------------------------------

class _MockLLM:
    """Minimal LLM stub — returns 'mock response' for every prompt."""

    class _Output:
        class _Choice:
            text = "mock response"
        outputs = [_Choice()]

    def get_tokenizer(self):
        class Tok:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]
        return Tok()

    def generate(self, prompts, sampling_params, **kwargs):
        return [_MockLLM._Output() for _ in prompts]


# ---------------------------------------------------------------------------
# _parse_output
# ---------------------------------------------------------------------------

class TestParseOutput:
    def test_plain_response_no_thinking(self):
        response, thinking = _parse_output("Vastaus on A.")
        assert response == "Vastaus on A."
        assert thinking is None

    def test_thinking_block_extracted(self):
        raw = f"{_THINKING_PREFIX}mietin tässä\n{_THINKING_DELIMITER}Vastaus on B."
        response, thinking = _parse_output(raw)
        assert "B" in response
        assert thinking == "mietin tässä"

    def test_empty_thinking_block(self):
        raw = f"{_THINKING_PREFIX}{_THINKING_DELIMITER}A"
        response, thinking = _parse_output(raw)
        assert thinking == ""

    def test_empty_string(self):
        response, thinking = _parse_output("")
        assert response == ""
        assert thinking is None

    def test_delimiter_only(self):
        response, thinking = _parse_output(_THINKING_DELIMITER)
        assert thinking == ""

    def test_thinking_stripped_from_response(self):
        raw = f"{_THINKING_PREFIX}thoughts{_THINKING_DELIMITER}final"
        response, _ = _parse_output(raw)
        assert "thoughts" not in response
        assert _THINKING_PREFIX not in response


# ---------------------------------------------------------------------------
# run_batch result structure (without actual model — tests shape only)
# ---------------------------------------------------------------------------

class TestRunBatchStructure:
    """Verify run_batch returns correctly shaped results using a mock LLM."""

    def test_returns_one_result_per_prompt(self):
        results = run_batch(
            _MockLLM(),
            prompts=["prompt 1", "prompt 2", "prompt 3"],
            model_path="mock/model",
        )
        assert len(results) == 3

    def test_result_has_required_fields(self):
        results = run_batch(
            _MockLLM(),
            prompts=["prompt"],
            model_path="mock/model",
        )
        r = results[0]
        assert "response" in r
        assert "prompt" in r
        assert "model" in r
        assert "elapsed_s" in r
        assert "generation_kwargs" in r

    def test_prompts_preserved_in_results(self):
        prompts = ["kysymys 1", "kysymys 2"]
        results = run_batch(_MockLLM(), prompts=prompts, model_path="mock/model")
        assert results[0]["prompt"] == "kysymys 1"
        assert results[1]["prompt"] == "kysymys 2"

    def test_empty_batch_returns_empty_list(self):
        results = run_batch(_MockLLM(), prompts=[], model_path="mock/model")
        assert results == []

    def test_generation_kwargs_recorded(self):
        results = run_batch(
            _MockLLM(),
            prompts=["p"],
            model_path="mock/model",
            max_tokens=256,
            temperature=0.5,
        )
        kwargs = results[0]["generation_kwargs"]
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.5


# ---------------------------------------------------------------------------
# Chunked batch mode — simulates the --batch-size logic in run_vllm_jsonl.py
# ---------------------------------------------------------------------------

def _run_chunked(llm, prompts, batch_size, **kwargs):
    """Replicate the chunked batch loop from run_vllm_jsonl.py."""
    all_results = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        all_results.extend(run_batch(llm, prompts=chunk, **kwargs))
    return all_results


class TestChunkedBatch:
    """Verify chunked batch mode splits prompts and collects results correctly."""

    PROMPTS = [f"prompt {i}" for i in range(7)]

    def test_all_prompts_covered(self):
        """No items are dropped between chunk boundaries."""
        results = _run_chunked(
            _MockLLM(), self.PROMPTS, batch_size=3, model_path="mock/model"
        )
        assert len(results) == len(self.PROMPTS)

    def test_order_preserved_across_chunks(self):
        """Prompt-to-result mapping is stable across chunk boundaries."""
        results = _run_chunked(
            _MockLLM(), self.PROMPTS, batch_size=3, model_path="mock/model"
        )
        for i, (prompt, result) in enumerate(zip(self.PROMPTS, results)):
            assert result["prompt"] == prompt, f"Order mismatch at index {i}"

    def test_correct_number_of_chunks(self):
        """math.ceil(N / batch_size) chunks are produced."""
        batch_size = 3
        expected_chunks = math.ceil(len(self.PROMPTS) / batch_size)
        call_count = 0
        llm = _MockLLM()
        original_generate = llm.generate

        def counting_generate(prompts, sampling_params, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_generate(prompts, sampling_params, **kwargs)

        llm.generate = counting_generate
        _run_chunked(llm, self.PROMPTS, batch_size=batch_size, model_path="mock/model")
        assert call_count == expected_chunks

    def test_last_chunk_may_be_smaller(self):
        """Final chunk contains only the remaining items (not padded)."""
        prompts = [f"p{i}" for i in range(5)]
        results = _run_chunked(
            _MockLLM(), prompts, batch_size=3, model_path="mock/model"
        )
        # 5 prompts, batch 3 → chunks of 3 and 2
        assert len(results) == 5

    def test_batch_size_larger_than_dataset(self):
        """Single chunk when batch_size >= number of prompts."""
        prompts = ["a", "b", "c"]
        call_count = 0
        llm = _MockLLM()
        original_generate = llm.generate

        def counting_generate(p, sp, **kw):
            nonlocal call_count
            call_count += 1
            return original_generate(p, sp, **kw)

        llm.generate = counting_generate
        results = _run_chunked(llm, prompts, batch_size=100, model_path="mock/model")
        assert call_count == 1
        assert len(results) == 3

    def test_batch_size_one_equals_sequential(self):
        """batch_size=1 produces one generate() call per prompt."""
        results = _run_chunked(
            _MockLLM(), self.PROMPTS, batch_size=1, model_path="mock/model"
        )
        assert len(results) == len(self.PROMPTS)
        for prompt, result in zip(self.PROMPTS, results):
            assert result["prompt"] == prompt

    def test_empty_prompt_list(self):
        """Empty input returns empty result list."""
        results = _run_chunked(
            _MockLLM(), [], batch_size=50, model_path="mock/model"
        )
        assert results == []


# ---------------------------------------------------------------------------
# batch_mode label (mirrors run_vllm_jsonl.py logic)
# ---------------------------------------------------------------------------

class TestBatchModeLabel:
    """Verify the batch_mode string written to run_meta is correct."""

    def _label(self, batch=False, batch_size=None):
        """Replicate the label logic from run_vllm_jsonl.py."""
        if batch:
            return "full"
        elif batch_size:
            return f"chunked/{batch_size}"
        else:
            return "sequential"

    def test_sequential_label(self):
        assert self._label() == "sequential"

    def test_full_batch_label(self):
        assert self._label(batch=True) == "full"

    def test_chunked_label_includes_size(self):
        assert self._label(batch_size=100) == "chunked/100"

    def test_chunked_label_different_sizes(self):
        assert self._label(batch_size=50)  == "chunked/50"
        assert self._label(batch_size=200) == "chunked/200"

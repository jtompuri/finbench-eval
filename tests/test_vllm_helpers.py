"""
Tests for vLLM runner helper functions.
No model loading or inference — only pure Python logic.

vLLM is Linux/CUDA only, so it is not installed in the development
environment on macOS. Tests that exercise run_batch() inject a fake
vllm module into sys.modules so the import guard inside run_batch()
is satisfied without requiring vLLM to actually be installed.
"""
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
from run_vllm_prompt import _parse_output, _THINKING_DELIMITER, _THINKING_PREFIX


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

    class _MockOutput:
        class _Choice:
            text = "mock response"
        outputs = [_Choice()]

    class _MockLLM:
        def get_tokenizer(self):
            class Tok:
                def apply_chat_template(self, messages, **kwargs):
                    return messages[0]["content"]
            return Tok()

        def generate(self, prompts, sampling_params, **kwargs):
            return [TestRunBatchStructure._MockOutput() for _ in prompts]

    def test_returns_one_result_per_prompt(self):
        from run_vllm_prompt import run_batch
        results = run_batch(
            self._MockLLM(),
            prompts=["prompt 1", "prompt 2", "prompt 3"],
            model_path="mock/model",
        )
        assert len(results) == 3

    def test_result_has_required_fields(self):
        from run_vllm_prompt import run_batch
        results = run_batch(
            self._MockLLM(),
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
        from run_vllm_prompt import run_batch
        prompts = ["kysymys 1", "kysymys 2"]
        results = run_batch(self._MockLLM(), prompts=prompts, model_path="mock/model")
        assert results[0]["prompt"] == "kysymys 1"
        assert results[1]["prompt"] == "kysymys 2"

    def test_empty_batch_returns_empty_list(self):
        from run_vllm_prompt import run_batch
        results = run_batch(self._MockLLM(), prompts=[], model_path="mock/model")
        assert results == []

    def test_generation_kwargs_recorded(self):
        from run_vllm_prompt import run_batch
        results = run_batch(
            self._MockLLM(),
            prompts=["p"],
            model_path="mock/model",
            max_tokens=256,
            temperature=0.5,
        )
        kwargs = results[0]["generation_kwargs"]
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.5

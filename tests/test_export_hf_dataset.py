"""
Tests for scripts/export_hf_dataset.py — merge logic and schema validation.
"""
import json
import pytest
from pathlib import Path
from export_hf_dataset import merge


# ---------------------------------------------------------------------------
# merge()
# ---------------------------------------------------------------------------

class TestMerge:
    def _output_item(self, **kwargs):
        base = {
            "id": "arc_challenge_fi_000",
            "task": "arc_challenge_fi",
            "task_type": "mcf_letter",
            "prompt": "Mikä on Maan satelliitti?",
            "expected": "A",
            "response": "Vastaus: A",
        }
        base.update(kwargs)
        return base

    def _score_item(self, **kwargs):
        base = {
            "id": "arc_challenge_fi_000",
            "task": "arc_challenge_fi",
            "task_type": "mcf_letter",
            "extracted": "A",
            "abstain": False,
            "correct": True,
            "score": 1.0,
        }
        base.update(kwargs)
        return base

    def test_schema_keys_present(self):
        result = merge(self._output_item(), self._score_item(), "TestModel")
        expected_keys = {"id", "task", "task_type", "model", "prompt",
                         "expected", "response", "extracted", "abstain",
                         "correct", "score"}
        assert set(result.keys()) == expected_keys

    def test_model_display_name(self):
        result = merge(self._output_item(), self._score_item(), "Claude Sonnet 4.6")
        assert result["model"] == "Claude Sonnet 4.6"

    def test_score_fields_from_score_item(self):
        result = merge(self._output_item(), self._score_item(score=0.0, correct=False), "M")
        assert result["score"] == pytest.approx(0.0)
        assert result["correct"] is False

    def test_prompt_from_output_item(self):
        result = merge(self._output_item(prompt="Test prompt"), self._score_item(), "M")
        assert result["prompt"] == "Test prompt"

    def test_empty_output_item_uses_score_fields(self):
        # Gemini 3.1 Pro case: no output JSONL, only score JSON
        result = merge({}, self._score_item(), "Gemini 3.1 Pro")
        assert result["id"] == "arc_challenge_fi_000"
        assert result["task"] == "arc_challenge_fi"
        assert result["prompt"] == ""
        assert result["response"] == ""

    def test_response_falls_back_to_response_raw(self):
        score = self._score_item()
        score["response_raw"] = "raw model output"
        result = merge({}, score, "M")
        assert result["response"] == "raw model output"

    def test_abstain_field(self):
        result = merge(self._output_item(), self._score_item(abstain=True), "M")
        assert result["abstain"] is True

    def test_id_from_score_item_takes_priority(self):
        output = self._output_item(id="old_id")
        score = self._score_item(id="correct_id")
        result = merge(output, score, "M")
        assert result["id"] == "correct_id"

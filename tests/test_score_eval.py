"""
Tests for scripts/score_eval.py — Wilson CI, bootstrap CI, and score_item.
"""
import pytest
from score_eval import bootstrap_ci, mean_se_ci, score_item, wilson_ci


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

class TestWilsonCi:
    def test_perfect_accuracy(self):
        lo, hi = wilson_ci(100, 100)
        assert lo > 0.9
        assert hi == pytest.approx(1.0)

    def test_zero_accuracy(self):
        lo, hi = wilson_ci(0, 100)
        assert lo == pytest.approx(0.0)
        assert hi < 0.1

    def test_zero_n(self):
        lo, hi = wilson_ci(0, 0)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(1.0)

    def test_ci_contains_proportion(self):
        lo, hi = wilson_ci(75, 100)
        assert lo <= 0.75 <= hi

    def test_ci_is_ordered(self):
        lo, hi = wilson_ci(50, 100)
        assert lo < hi

    def test_large_n_narrow_ci(self):
        lo, hi = wilson_ci(500, 1000)
        assert (hi - lo) < 0.07  # should be narrow for large n


# ---------------------------------------------------------------------------
# mean_se_ci
# ---------------------------------------------------------------------------

class TestMeanSeCi:
    def test_single_value(self):
        lo, hi = mean_se_ci([0.8])
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(1.0)

    def test_identical_values(self):
        lo, hi = mean_se_ci([0.5, 0.5, 0.5])
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_ci_contains_mean(self):
        values = [0.3, 0.5, 0.7, 0.6, 0.4]
        mean = sum(values) / len(values)
        lo, hi = mean_se_ci(values)
        assert lo <= mean <= hi


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCi:
    def test_deterministic_with_seed(self):
        values = [0.0, 1.0] * 50
        lo1, hi1 = bootstrap_ci(values, seed=42)
        lo2, hi2 = bootstrap_ci(values, seed=42)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_ci_contains_mean(self):
        values = [0.6] * 80 + [0.4] * 20
        mean = sum(values) / len(values)
        lo, hi = bootstrap_ci(values)
        assert lo <= mean <= hi

    def test_single_value(self):
        lo, hi = bootstrap_ci([0.5])
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_item
# ---------------------------------------------------------------------------

class TestScoreItem:
    def _make_item(self, task_type, expected, response, **kwargs):
        return {
            "id": "test_001",
            "task": "arc_challenge_fi",
            "task_type": task_type,
            "expected": expected,
            "response": response,
            **kwargs,
        }

    # MCF letter
    def test_mcf_letter_correct(self):
        item = self._make_item("mcf_letter", "A", "Vastaus: A")
        result = score_item(item)
        assert result["correct"] is True
        assert result["score"] == pytest.approx(1.0)
        assert result["extracted"] == "A"

    def test_mcf_letter_wrong(self):
        item = self._make_item("mcf_letter", "A", "Vastaus: B")
        result = score_item(item)
        assert result["correct"] is False
        assert result["score"] == pytest.approx(0.0)

    def test_mcf_letter_abstain(self):
        item = self._make_item("mcf_letter", "A", "En osaa vastata tähän kysymykseen.")
        result = score_item(item)
        assert result["abstain"] is True
        assert result["correct"] is False

    def test_mcf_letter_bold_correct(self):
        item = self._make_item("mcf_letter", "C", "**C**")
        result = score_item(item)
        assert result["correct"] is True

    # MCF word
    def test_mcf_word_correct(self):
        item = self._make_item(
            "mcf_word", "positiivinen", "Vastaus: positiivinen",
            expected_choices=["positiivinen", "negatiivinen", "neutraali"],
        )
        result = score_item(item)
        assert result["correct"] is True
        assert result["score"] == pytest.approx(1.0)

    def test_mcf_word_wrong(self):
        item = self._make_item(
            "mcf_word", "positiivinen", "negatiivinen",
            expected_choices=["positiivinen", "negatiivinen", "neutraali"],
        )
        result = score_item(item)
        assert result["correct"] is False

    # Generative (SQuAD-style) — gen tasks return exact_match + f1, not correct
    def test_gen_exact_match(self):
        item = self._make_item("gen", "Helsinki", "Helsinki")
        result = score_item(item)
        assert result["exact_match"] is True
        assert result["score"] == pytest.approx(1.0)

    def test_gen_f1_partial(self):
        item = self._make_item(
            "gen", "Helsinki on Suomen pääkaupunki",
            "Helsinki on pääkaupunki",
        )
        result = score_item(item)
        assert 0.0 < result["score"] < 1.0

    def test_gen_no_match(self):
        item = self._make_item("gen", "Helsinki", "Tampere")
        result = score_item(item)
        assert result["score"] == pytest.approx(0.0)

    def test_gen_thinking_stripped(self):
        response = "<|channel>thought\nLet me think\n<channel|>Helsinki"
        item = self._make_item("gen", "Helsinki", response)
        result = score_item(item)
        assert result["exact_match"] is True

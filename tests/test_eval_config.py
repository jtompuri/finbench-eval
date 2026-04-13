"""
Tests for scripts/eval_config.py — baselines, normalisation, and score helpers.
"""
import pytest
from eval_config import BASELINES, normalized, primary_ci, primary_score


class TestBaselines:
    def test_all_tasks_present(self):
        expected_tasks = {
            "arc_challenge_fi", "belebele_fin", "goldenswag_fi",
            "scandisent_fi", "sib200_fi", "finbench_general_knowledge",
            "squad_fi", "truthfulqa_fi_mc1", "finbench_analogies",
            "finbench_emotions", "finbench_hhh_alignment", "finbench_similarities",
        }
        assert expected_tasks == set(BASELINES.keys())

    def test_baselines_in_range(self):
        for task, b in BASELINES.items():
            assert 0.0 <= b < 1.0, f"{task} baseline {b} out of range [0, 1)"

    def test_mcf4_baseline(self):
        # 4-choice MCF tasks should have 0.25 baseline
        for task in ("arc_challenge_fi", "belebele_fin", "goldenswag_fi"):
            assert BASELINES[task] == pytest.approx(0.25)

    def test_squad_baseline_zero(self):
        assert BASELINES["squad_fi"] == pytest.approx(0.0)

    def test_binary_baseline(self):
        # 2-class sentiment task
        assert BASELINES["scandisent_fi"] == pytest.approx(0.5)


class TestNormalized:
    def test_perfect_score(self):
        # score=1.0, baseline=0.25 → (1 - 0.25) / (1 - 0.25) = 1.0
        assert normalized(1.0, "arc_challenge_fi") == pytest.approx(1.0)

    def test_at_baseline(self):
        # score=0.25, baseline=0.25 → 0.0
        assert normalized(0.25, "arc_challenge_fi") == pytest.approx(0.0)

    def test_below_baseline(self):
        assert normalized(0.0, "arc_challenge_fi") < 0.0

    def test_unknown_task_uses_zero_baseline(self):
        # Unknown task → baseline=0.0 → normalized == raw score
        assert normalized(0.8, "unknown_task") == pytest.approx(0.8)

    def test_squad_no_baseline_adjustment(self):
        # squad_fi baseline=0.0 → normalized == raw score
        assert normalized(0.6, "squad_fi") == pytest.approx(0.6)

    def test_formula_correctness(self):
        # (0.8 - 0.25) / (1 - 0.25) = 0.55 / 0.75 ≈ 0.7333
        assert normalized(0.8, "arc_challenge_fi") == pytest.approx(0.55 / 0.75)


class TestPrimaryScore:
    def test_mcf_letter_uses_accuracy(self):
        data = {"task_type": "mcf_letter", "accuracy": 0.85, "score_avg": 0.70}
        assert primary_score(data) == pytest.approx(0.85)

    def test_mcf_word_uses_accuracy(self):
        data = {"task_type": "mcf_word", "accuracy": 0.90, "score_avg": 0.75}
        assert primary_score(data) == pytest.approx(0.90)

    def test_gen_uses_f1(self):
        data = {"task_type": "gen", "f1_avg": 0.60, "score_avg": 0.55}
        assert primary_score(data) == pytest.approx(0.60)

    def test_mcf_fallback_to_score_avg(self):
        data = {"task_type": "mcf_letter", "score_avg": 0.70}
        assert primary_score(data) == pytest.approx(0.70)

    def test_gen_fallback_to_score_avg(self):
        data = {"task_type": "gen", "score_avg": 0.55}
        assert primary_score(data) == pytest.approx(0.55)

    def test_missing_all_returns_zero(self):
        assert primary_score({"task_type": "mcf_letter"}) == pytest.approx(0.0)


class TestPrimaryCi:
    def test_returns_tuple(self):
        data = {"ci_95": [0.70, 0.90]}
        lo, hi = primary_ci(data)
        assert lo == pytest.approx(0.70)
        assert hi == pytest.approx(0.90)

    def test_default_when_missing(self):
        lo, hi = primary_ci({})
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(1.0)

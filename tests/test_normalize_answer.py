"""
Tests for scripts/normalize_answer.py — answer extraction and normalization.

These tests cover the critical path: every evaluation item is scored through
these functions, so edge cases here directly affect benchmark results.
"""
import pytest
from normalize_answer import (
    best_f1_against_list,
    exact_match,
    extract_final_answer,
    extract_mcf_letter,
    extract_mcf_word,
    normalize_for_exact_match,
    strip_markdown,
    strip_punctuation,
    token_f1,
)


# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------

class TestExtractFinalAnswer:
    def test_no_delimiter_returns_unchanged(self):
        assert extract_final_answer("Vastaus on A.") == "Vastaus on A."

    def test_strips_thinking_block(self):
        text = "<|channel>thought\nLet me think...\n<channel|>The answer is B."
        assert extract_final_answer(text) == "The answer is B."

    def test_empty_string(self):
        assert extract_final_answer("") == ""

    def test_delimiter_only(self):
        assert extract_final_answer("<channel|>") == ""

    def test_whitespace_stripped_after_delimiter(self):
        text = "<|channel>thought\nthinking\n<channel|>  A  "
        assert extract_final_answer(text) == "A"


# ---------------------------------------------------------------------------
# strip_markdown
# ---------------------------------------------------------------------------

class TestStripMarkdown:
    def test_bold(self):
        assert strip_markdown("**A**") == "A"

    def test_italic(self):
        assert strip_markdown("*B*") == "B"

    def test_nested_bold_italic(self):
        assert strip_markdown("**vastaus on *C***") == "vastaus on C"

    def test_no_markdown(self):
        assert strip_markdown("plain text") == "plain text"


# ---------------------------------------------------------------------------
# strip_punctuation
# ---------------------------------------------------------------------------

class TestStripPunctuation:
    def test_trailing_period(self):
        assert strip_punctuation("Oikein.") == "Oikein"

    def test_leading_quote(self):
        assert strip_punctuation('"teksti"') == "teksti"

    def test_em_dash(self):
        assert strip_punctuation("–vastaus–") == "vastaus"

    def test_no_punctuation(self):
        assert strip_punctuation("teksti") == "teksti"


# ---------------------------------------------------------------------------
# extract_mcf_letter
# ---------------------------------------------------------------------------

class TestExtractMcfLetter:
    def test_plain_letter_start(self):
        assert extract_mcf_letter("A") == "A"

    def test_letter_with_period(self):
        assert extract_mcf_letter("B.") == "B"

    def test_vastaus_prefix(self):
        assert extract_mcf_letter("Vastaus: C") == "C"

    def test_lowercase_normalized(self):
        assert extract_mcf_letter("d") == "D"

    def test_letter_in_sentence(self):
        assert extract_mcf_letter("Oikea vastaus on B koska...") == "B"

    def test_bold_letter(self):
        assert extract_mcf_letter("**A**") == "A"

    def test_after_thinking_block(self):
        text = "<|channel>thought\nLet me reason\n<channel|>B"
        assert extract_mcf_letter(text) == "B"

    def test_empty_response(self):
        assert extract_mcf_letter("") == ""

    def test_no_valid_letter(self):
        assert extract_mcf_letter("En tiedä vastausta.") == ""

    def test_e_not_extracted(self):
        # E is not a valid MCF option (only A–D)
        assert extract_mcf_letter("E") == ""

    def test_newline_before_letter(self):
        assert extract_mcf_letter("Ajatellaanpa...\nC") == "C"


# ---------------------------------------------------------------------------
# extract_mcf_word
# ---------------------------------------------------------------------------

CHOICES = ["positiivinen", "negatiivinen", "neutraali"]

class TestExtractMcfWord:
    def test_exact_match(self):
        assert extract_mcf_word("Vastaus: positiivinen", CHOICES) == "positiivinen"

    def test_case_insensitive(self):
        assert extract_mcf_word("POSITIIVINEN", CHOICES) == "positiivinen"

    def test_answer_line_priority(self):
        # Model explains why negative is wrong, but final answer line is positive
        text = "Negatiivinen ei sovi tähän. Vastaus: positiivinen"
        assert extract_mcf_word(text, CHOICES) == "positiivinen"

    def test_word_overlap_fallback(self):
        choices = ["economic growth", "social development", "political change"]
        # Response uses most of the content words from the first choice
        resp = "This is about economic growth and its impact on society."
        assert extract_mcf_word(resp, choices) == "economic growth"

    def test_no_match_returns_empty(self):
        assert extract_mcf_word("En osaa sanoa.", CHOICES) == ""

    def test_empty_choices(self):
        assert extract_mcf_word("positiivinen", []) == ""

    def test_empty_response(self):
        assert extract_mcf_word("", CHOICES) == ""

    def test_after_thinking_block(self):
        text = "<|channel>thought\nnegatiivinen\n<channel|>Vastaus: neutraali"
        assert extract_mcf_word(text, CHOICES) == "neutraali"


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------

class TestTokenF1:
    def test_identical(self):
        assert token_f1("Helsinki on pääkaupunki", "Helsinki on pääkaupunki") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("kissa istuu", "koira juoksee") == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = token_f1("Helsinki on kaupunki", "Helsinki on Suomen pääkaupunki")
        assert 0.0 < score < 1.0

    def test_empty_prediction(self):
        assert token_f1("", "vastaus") == pytest.approx(0.0)

    def test_empty_reference(self):
        assert token_f1("vastaus", "") == pytest.approx(0.0)

    def test_both_empty(self):
        assert token_f1("", "") == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert token_f1("HELSINKI", "helsinki") == pytest.approx(1.0)

    def test_punctuation_stripped(self):
        assert token_f1("Helsinki.", "Helsinki") == pytest.approx(1.0)

    def test_symmetric(self):
        a, b = "kissa istuu matolla", "matolla istuu kissa"
        assert token_f1(a, b) == pytest.approx(token_f1(b, a))


# ---------------------------------------------------------------------------
# best_f1_against_list
# ---------------------------------------------------------------------------

class TestBestF1AgainstList:
    def test_best_of_multiple(self):
        refs = ["väärä vastaus", "Helsinki on pääkaupunki", "toinen väärä"]
        pred = "Helsinki on pääkaupunki"
        assert best_f1_against_list(pred, refs) == pytest.approx(1.0)

    def test_empty_list(self):
        assert best_f1_against_list("vastaus", []) == pytest.approx(0.0)

    def test_single_reference(self):
        assert best_f1_against_list("a b c", ["a b c"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_identical(self):
        assert exact_match("Helsinki", "Helsinki") is True

    def test_different(self):
        assert exact_match("Helsinki", "Tampere") is False

    def test_case_sensitive_default(self):
        assert exact_match("helsinki", "Helsinki") is False

    def test_case_insensitive_flag(self):
        assert exact_match("helsinki", "Helsinki", lowercase_flag=True) is True

    def test_strips_markdown(self):
        assert exact_match("**Helsinki**", "Helsinki") is True

    def test_strips_whitespace(self):
        assert exact_match("  Helsinki  ", "Helsinki") is True

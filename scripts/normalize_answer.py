#!/usr/bin/env python3
"""
Answer normalization helpers for FIN-bench-v2 smoke tests and comparable subset scoring.
Lightweight and transparent — not a full FIN-bench-v2 normalization stack.
"""
import re
from collections import Counter


# ---------------------------------------------------------------------------
# Thinking block and formatting
# ---------------------------------------------------------------------------

def extract_final_answer(text: str) -> str:
    """
    Extract the final answer from a response that may contain a thinking block.

    Gemma 4 with reasoning enabled wraps chain-of-thought in:
        <|channel>thought
        [reasoning...]
        <channel|>[final answer]

    If the delimiter is present, return only the text after <channel|>.
    If not present, return the text unchanged.
    """
    delimiter = "<channel|>"
    if delimiter in text:
        return text.split(delimiter, 1)[1].strip()
    return text


def strip_markdown(text: str) -> str:
    """Remove common markdown formatting (bold, italic)."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    return text


def strip_whitespace(text: str) -> str:
    return text.strip()


def lowercase(text: str) -> str:
    return text.lower()


def strip_punctuation(text: str) -> str:
    return text.strip(".,;:!?\"'()-–—")


# ---------------------------------------------------------------------------
# MCF answer extraction
# ---------------------------------------------------------------------------

def extract_mcf_letter(text: str) -> str:
    """
    Extract the first A/B/C/D letter from a model response.
    Returns empty string if no letter found.
    """
    text = extract_final_answer(text)
    text = strip_markdown(text).strip()
    # Look for a standalone letter at the start, or after "Vastaus:" / newline
    match = re.search(r"(?:^|Vastaus:\s*|[\n\r])([A-Da-d])(?:\s|$|\.|:)", text)
    if match:
        return match.group(1).upper()
    # Fallback: first standalone letter in the response
    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        return match.group(1).upper()
    return ""


# Markers that introduce reasoning/explanation sections in verbose model responses.
# Everything after these markers is about why certain choices are wrong — matching
# against that section would extract the REJECTED choices, not the chosen one.
_EXPLANATION_MARKER_RE = re.compile(
    r"\n+\*{0,2}(?:Perustelu|Selitys|Koska|Huomio|Huom)\b",
    re.IGNORECASE,
)


def _trim_to_answer_section(text: str) -> str:
    """Return only the answer-bearing portion of a verbose model response.

    Finnish-language models often generate responses of the form:
        "Valitsen vaihtoehdon: [CHOSEN CHOICE]

         Perustelu: ... [WRONG CHOICE] on väärä koska ..."

    Passing the full text to Pass 1/2 would match the WRONG CHOICE (mentioned
    verbatim in the explanation) instead of the chosen one.  Trimming at the
    first explanation marker restricts matching to the part of the response
    where the answer is actually stated.
    """
    m = _EXPLANATION_MARKER_RE.search(text)
    if m:
        return text[: m.start()].strip()
    return text


def extract_mcf_word(text: str, expected_choices: list) -> str:
    """
    Find the best matching choice in the model response.

    Strategy (in order):
    0. Answer-line priority — look for choices only in the "Vastaus[:]" prefix
       line before searching the full response. This prevents wrong matches when
       the model explains why other choices are incorrect (and thus mentions them
       in its reasoning).
    1. Exact substring match — answer section only (text before any explanation
       marker such as "Perustelu:" or "Selitys:"). Restricting to the answer
       section prevents matching wrong choices that the model quotes while
       explaining why it rejected them.
    2. Word-overlap fallback — same answer section, handles paraphrases and
       near-misses (e.g. a single missing or misspelled word).  Computes the
       fraction of each choice's content words that appear in the response
       (recall-style overlap).  The choice with the highest overlap wins if it
       exceeds OVERLAP_THRESHOLD; otherwise returns empty string.

    Returns the matched choice string (lowercased), or empty string if none found.
    """
    OVERLAP_THRESHOLD = 0.40   # ≥40 % of choice words must appear in response

    text = extract_final_answer(text)
    text_clean = strip_markdown(text).strip()
    text_norm = re.sub(r"\s+", " ", text_clean.lower())

    # Answer section: text before the first explanation marker.  Pass 1 and 2
    # operate on this shorter string; Pass 0 already targets the first line only.
    answer_section = _trim_to_answer_section(text_clean)
    answer_norm = re.sub(r"\s+", " ", answer_section.lower())

    # --- Pass 0: answer-line priority ---
    # Extract first line that starts with "Vastaus" (handles "Vastaus:", "Vastaus on")
    answer_line_match = re.match(
        r"(?:Oikea vastaus on|Vastaus(?:\s+on)?)[:\s]+(.+?)(?:\n|$)",
        text_clean, re.IGNORECASE
    )
    if answer_line_match:
        answer_line = re.sub(r"\s+", " ", answer_line_match.group(1).lower().strip())
        for choice in expected_choices:
            choice_norm = re.sub(r"\s+", " ", choice.lower().strip())
            if choice_norm in answer_line:
                return choice.lower()

    # --- Pass 1: exact substring match (answer section only) ---
    for choice in expected_choices:
        choice_norm = re.sub(r"\s+", " ", choice.lower().strip())
        if choice_norm in answer_norm:
            return choice.lower()

    # --- Pass 2: word-overlap fallback (answer section only) ---
    # Ignore very short stop-words (≤2 chars) to avoid noise
    def content_words(s: str) -> list:
        return [w for w in re.sub(r"[^\w\s]", "", s.lower()).split() if len(w) > 2]

    answer_words = set(content_words(answer_norm))
    best_choice, best_score = "", 0.0
    for choice in expected_choices:
        cwords = content_words(choice)
        if not cwords:
            continue
        overlap = sum(1 for w in cwords if w in answer_words) / len(cwords)
        if overlap > best_score:
            best_score, best_choice = overlap, choice

    if best_score >= OVERLAP_THRESHOLD:
        return best_choice.lower()
    return ""


# ---------------------------------------------------------------------------
# Generative scoring
# ---------------------------------------------------------------------------

def normalize_for_exact_match(text: str, lowercase_flag: bool = False,
                               strip_punct: bool = False,
                               strip_md: bool = True) -> str:
    """Apply normalization steps in a transparent, configurable order."""
    text = extract_final_answer(text)
    text = strip_whitespace(text)
    if strip_md:
        text = strip_markdown(text)
        text = strip_whitespace(text)
    if lowercase_flag:
        text = lowercase(text)
    if strip_punct:
        text = strip_punctuation(text)
    return text


def exact_match(prediction: str, reference: str, lowercase_flag: bool = False,
                strip_punct: bool = False) -> bool:
    pred = normalize_for_exact_match(prediction, lowercase_flag=lowercase_flag,
                                     strip_punct=strip_punct)
    ref = normalize_for_exact_match(reference, lowercase_flag=lowercase_flag,
                                    strip_punct=strip_punct)
    return pred == ref


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference (SQuAD-style)."""
    pred_tokens = normalize_for_exact_match(
        prediction, lowercase_flag=True, strip_punct=True
    ).split()
    ref_tokens = normalize_for_exact_match(
        reference, lowercase_flag=True, strip_punct=True
    ).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def best_f1_against_list(prediction: str, references: list) -> float:
    """Return the highest token F1 against any reference in the list."""
    if not references:
        return 0.0
    return max(token_f1(prediction, ref) for ref in references)

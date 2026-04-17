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

    Supports multiple thinking-block formats that vary by backend:
      - MLX Gemma 4:        <|channel>thought...<channel|>[answer]
      - Ollama Gemma 4:     <|think|>...</think>[answer]  or  <|think|>...<|end|>[answer]
      - Ollama Gemma 4:     <|think|>...<turn|>\n[answer]<turn|>   (some items)
      - OpenAI-compat:      <think>...</think>[answer]

    Opening tags differ across backends, but all end with one of:
        </think>, <|end|>, <channel|>
    We locate the last occurrence of any of these closing markers and return
    everything after it. This is robust to asymmetric open/close tags.

    Fallback for <turn|>-terminated Ollama outputs: if the text contains a
    `<|think|>` open marker but none of the standard closers, treat <turn|>
    as a segment separator and return the last non-empty segment (trailing
    <turn|> is stripped first so "...<turn|>\nB<turn|>" → "B").

    If no marker is present at all, return the text unchanged.
    """
    close_markers = ("</think>", "<|end|>", "<channel|>")
    best_end = -1
    best_mark = ""
    for mark in close_markers:
        idx = text.rfind(mark)
        if idx > best_end:
            best_end = idx
            best_mark = mark
    if best_end >= 0:
        return text[best_end + len(best_mark):].strip()

    # Fallback: <turn|>-only termination (Ollama Gemma 4 edge case).
    # Gate on <|think|> to avoid over-stripping responses that merely
    # mention <turn|> inside normal text.
    if "<|think|>" in text and "<turn|>" in text:
        stripped = text.rstrip()
        # Strip trailing <turn|> tokens (possibly separated by whitespace)
        while stripped.endswith("<turn|>"):
            stripped = stripped[: -len("<turn|>")].rstrip()
        last_turn = stripped.rfind("<turn|>")
        if last_turn >= 0:
            return stripped[last_turn + len("<turn|>"):].strip()

    return text.strip()


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
#
# Patterns covered:
#   "**Perustelu:**"  "**Perustelut:**"  "Perustelu:" (Finnish: justification)
#   "**Selitys:**"    "Selitys:"         (Finnish: explanation)
#   "**Koska**"       "Koska:"           (Finnish: because/since)
#   "**Huomio:**"     "Huomio:"          (Finnish: note)
#   "### Selitys:"    "### Miksi?"       (Markdown headings with explanation keywords)
#
# NOTE: the heading pattern uses the same Finnish keyword list (not bare "#{1,3}\s")
# to avoid matching answer headings like "## **Tyttö** ✅" or "## Vastaus: **X**"
# that Claude (Anthropic) generates — where the chosen answer IS the heading text.
_EXPLANATION_MARKER_RE = re.compile(
    r"\n+(?:"
    r"\*{0,2}(?:Perustelu|Selitys|Koska|Huomio|Huom|Miksi)\w*"     # Finnish explanation words (bold or plain)
    r"|#{1,3}\s+(?:Perustelu|Selitys|Koska|Huomio|Huom|Miksi)\w*"  # same words as Markdown headings
    r")",
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
    1b. Numbered/ordinal choice reference — "Vaihtoehto 2 on paras" or
       "Toinen vaihtoehto on..." mapped to expected_choices index.  Catches
       Gemma 4 CoT responses that reference choices by position instead of
       quoting their text.
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

    # --- Pass 1b: numbered/ordinal choice reference ---
    # Some models (especially Gemma 4 CoT) say "Vaihtoehto 2 on paras" or
    # "Toinen vaihtoehto on paras vastaus" instead of quoting the choice text.
    # Map Finnish ordinals and numbered references to choice indices — BUT
    # only when the reference appears in a positive-selection context.
    #
    # Negative-context example to reject:
    #   "Paras vastaus on: '[choice A text]'. ... Toinen vaihtoehto ei ole hyvä."
    # Here "Toinen vaihtoehto" refers to the REJECTED choice. Matching it
    # would flip a correct extraction to the wrong choice.
    _ORDINALS_FI = {
        "ensimmäinen": 0, "toinen": 1, "kolmas": 2, "neljäs": 3,
        "viides": 4, "kuudes": 5,
    }
    _POSITIVE_CTX_RE = re.compile(
        r"(paras|oikea|parempi|sopivin|valitsen|suosittelen|hyväksyn)"
    )
    _NEGATIVE_CTX_RE = re.compile(
        r"(ei ole|ei sovi|on huono|on väärä|ei ole hyvä|ei ole paras|"
        r"vaikkei|valitettavasti|ei kannata|ei voi valita|kannata valita|"
        r"on vaarallinen|on vastuuton|en koskaan|vahingoittamis|"
        r"on epäasialli|on sopimat|ei ole turvalli|on epäeetti)"
    )

    def _select_by_ref(start: int, end: int, idx: int) -> str:
        """Return choice if context around ref is positive, else empty.

        Negative check uses a short 20-char window after the ordinal so
        that rejections like "toinen vaihtoehto ei ole hyvä" are caught,
        but negatives ABOUT the OTHER ordinal 30+ chars later do not
        disqualify an otherwise-positive reference.

        Positive check uses a wider 40-char ±window because selection
        keywords like "paras vastaus on" often precede the ordinal.
        """
        before = answer_norm[max(0, start - 40):start]
        after_short = answer_norm[end:min(len(answer_norm), end + 20)]
        after_wide = answer_norm[end:min(len(answer_norm), end + 40)]
        # Reject if immediate after context is negative
        if _NEGATIVE_CTX_RE.search(after_short):
            return ""
        # Accept only if positive keyword appears in ±40 char window
        if _POSITIVE_CTX_RE.search(before + " " + after_wide):
            if 0 <= idx < len(expected_choices):
                return expected_choices[idx].lower()
        return ""

    # Collect all ordinal/numbered refs with their positions
    refs = []
    for m in re.finditer(r"vaihtoehto\s+(\d)\b", answer_norm):
        refs.append((m.start(), m.end(), int(m.group(1)) - 1))
    for m in re.finditer(
        r"\b(ensimmäinen|toinen|kolmas|neljäs|viides|kuudes)\s+vaihtoehto\b",
        answer_norm,
    ):
        refs.append((m.start(), m.end(), _ORDINALS_FI[m.group(1)]))
    refs.sort()  # process in document order
    for start, end, idx in refs:
        picked = _select_by_ref(start, end, idx)
        if picked:
            return picked

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

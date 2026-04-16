#!/usr/bin/env python3
"""
Frontier model API adapters for FIN-bench-v2 evaluation pipeline.

Each single-call adapter accepts (prompt, model_id, max_tokens, temperature)
and returns a dict compatible with the JSONL output schema used by run_eval_jsonl.py:

    {
        "model":             str,
        "response":          str,   # full raw response text
        "generation_kwargs": dict,
        "elapsed_s":         float, # None for batch results
    }

Environment variables required:
    OPENAI_API_KEY      — OpenAI API key
    ANTHROPIC_API_KEY   — Anthropic API key
    GOOGLE_API_KEY      — Google AI Studio API key
    OPENROUTER_API_KEY  — OpenRouter API key (used for openrouter provider)

Batch functionality:
    OpenAI:       Batch Files + Batch Jobs API  (50% cost reduction, ~24 h turnaround)
    Anthropic:    Message Batches API           (50% cost reduction, ~24 h turnaround)
    Google:       Concurrent calls via ThreadPoolExecutor (no batch endpoint in AI Studio)
    OpenRouter:   Concurrent calls via ThreadPoolExecutor (no batch endpoint)
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(
            f"Required environment variable not set: {var}\n"
            f"Add it to your shell profile or .env file."
        )
    return val


def _gen_kwargs(max_tokens: int, temperature: float) -> dict:
    return {"max_tokens": max_tokens, "temperature": temperature}


# Models that reject the temperature parameter entirely
_ANTHROPIC_NO_TEMPERATURE = frozenset({"claude-opus-4-7"})


def _is_retryable(exc: Exception) -> bool:
    """
    Return True if the exception represents a transient API error worth retrying.

    Checks the exception's status_code attribute first (reliable for all SDK clients),
    then falls back to keyword matching in the message string.
    Non-retryable errors (401 auth, 403 permission, 404 not found) are never retried.
    """
    # Use SDK-provided status code if available
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code is not None:
        try:
            code = int(status_code)
        except (TypeError, ValueError):
            code = 0
        if code in (401, 403, 404):
            return False   # auth / permission / not-found — retrying won't help
        if code in (429, 500, 502, 503, 529):
            return True    # rate limit or server error — retry

    # Fallback: keyword search in message (covers network-level errors)
    msg = str(exc).lower()
    return any(k in msg for k in (
        "rate limit", "too many requests",
        "timeout", "timed out",
        "connection", "network",
        "overloaded", "service unavailable",
    ))


_MAX_RETRY_WAIT_S = 60.0  # Never sleep longer than this between retries


def _is_daily_quota_error(exc: Exception) -> bool:
    """Return True if the error is a hard daily quota limit (not worth retrying)."""
    msg = str(exc)
    return any(k in msg for k in (
        "per_day", "PerDay", "per_model_per_day",
        "GenerateRequestsPerDay",
    ))


def _suggested_wait(exc: Exception, default: float) -> float:
    """
    Extract the provider's suggested retry delay from the error message if present.
    Google RESOURCE_EXHAUSTED errors include 'retry in Xs' in the message.
    Falls back to `default` if no suggestion is found.
    Capped at _MAX_RETRY_WAIT_S to avoid sleeping for hours on daily quota errors.
    """
    import re
    msg = str(exc)
    m = re.search(r'retry in (\d+(?:\.\d+)?)\s*s', msg, re.IGNORECASE)
    if m:
        suggested = float(m.group(1)) + 2.0   # add 2 s safety buffer
        return min(suggested, _MAX_RETRY_WAIT_S)
    return min(default, _MAX_RETRY_WAIT_S)


def _retry(fn, retries: int = 3, backoff: float = 5.0):
    """
    Call fn() with exponential-backoff retry on transient errors.
    For quota/rate-limit errors the provider's suggested wait time is used
    when available (e.g. Google includes 'Please retry in Xs' in the message),
    capped at _MAX_RETRY_WAIT_S.
    Daily quota errors (RPD) are treated as non-retryable and fail immediately.
    Raises immediately on non-retryable errors (auth, not-found, etc.).
    """
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if not _is_retryable(exc) or _is_daily_quota_error(exc) or attempt == retries - 1:
                raise
            default_wait = backoff * (2 ** attempt)
            wait = _suggested_wait(exc, default_wait)
            print(f"\nTransient error (attempt {attempt + 1}/{retries}): {exc}")
            print(f"Retrying in {wait:.0f}s ...")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Single-call adapters  (smoke tests and sequential runs)
# ---------------------------------------------------------------------------

def run_openai_prompt(
    prompt: str,
    model_id: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """Run a single prompt through the OpenAI Chat Completions API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))

    def _call():
        t0 = time.time()
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = round(time.time() - t0, 3)
        response = completion.choices[0].message.content or ""
        return {
            "model": model_id,
            "response": response,
            "generation_kwargs": _gen_kwargs(max_tokens, temperature),
            "elapsed_s": elapsed,
        }

    return _retry(_call)


def run_openai_thinking_prompt(
    prompt: str,
    model_id: str,
    max_tokens: int = 2048,  # kattaa reasoning + output yhteensä; vastaa Clauden budget_tokens=1024 + max_tokens=2048
    temperature: float = 1.0,  # unused — Responses API does not accept temperature
) -> dict:
    """
    Run a single prompt through the OpenAI Responses API with reasoning enabled.

    The Responses API is the only OpenAI endpoint that exposes reasoning tokens for
    GPT-5.4. The Chat Completions API returns reasoning_tokens=0 for this model.

    Note: temperature is not a supported parameter in the Responses API — the value
    passed here is ignored and recorded as 1.0 in generation_kwargs for documentation.
    reasoning_effort is hardcoded to 'high'.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))

    def _call():
        t0 = time.time()
        response = client.responses.create(
            model=model_id,
            input=prompt,
            reasoning={"effort": "high"},
            max_output_tokens=max_tokens,
        )
        elapsed = round(time.time() - t0, 3)

        output_text = ""
        for item in response.output:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []):
                    if getattr(c, "type", "") == "output_text":
                        output_text = c.text

        usage = getattr(response, "usage", None)
        reasoning_tokens = (
            getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
        )

        return {
            "model": model_id,
            "response": output_text,
            "generation_kwargs": {
                "max_tokens": max_tokens,
                "temperature": 1.0,
                "reasoning_effort": "high",
                "api": "responses",
                "reasoning_tokens_sample": reasoning_tokens,
            },
            "elapsed_s": elapsed,
        }

    return _retry(_call)


def run_anthropic_prompt(
    prompt: str,
    model_id: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> dict:
    """Run a single prompt through the Anthropic Messages API (standard, no thinking)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required: pip install anthropic")

    client = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))

    def _call():
        t0 = time.time()
        kwargs = dict(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if model_id not in _ANTHROPIC_NO_TEMPERATURE:
            kwargs["temperature"] = temperature
        message = client.messages.create(**kwargs)
        elapsed = round(time.time() - t0, 3)
        # Filter for text blocks explicitly — avoids mis-reading thinking or tool_use blocks
        text_blocks = [b for b in (message.content or [])
                       if getattr(b, "type", None) == "text"]
        response = text_blocks[0].text if text_blocks else ""
        return {
            "model": model_id,
            "response": response,
            "generation_kwargs": _gen_kwargs(max_tokens, temperature),
            "elapsed_s": elapsed,
        }

    return _retry(_call)


def run_anthropic_thinking_prompt(
    prompt: str,
    model_id: str,
    max_tokens: int = 2048,
    temperature: float = 1.0,  # required for extended thinking; value is ignored
) -> dict:
    """
    Run a single prompt through the Anthropic Messages API with extended thinking.

    budget_tokens is hardcoded to 1024 (minimum allowed); max_tokens must be > budget_tokens.
    temperature must be 1 for extended thinking — the value passed here is ignored.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required: pip install anthropic")

    BUDGET_TOKENS = 1024
    if max_tokens <= BUDGET_TOKENS:
        max_tokens = BUDGET_TOKENS + 512

    client = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))

    def _call():
        t0 = time.time()
        message = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=1,
            thinking={"type": "enabled", "budget_tokens": BUDGET_TOKENS},
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = round(time.time() - t0, 3)
        text_blocks = [b for b in (message.content or [])
                       if getattr(b, "type", None) == "text"]
        response = text_blocks[0].text if text_blocks else ""
        return {
            "model": model_id,
            "response": response,
            "generation_kwargs": {
                "max_tokens": max_tokens,
                "temperature": 1,
                "thinking_budget_tokens": BUDGET_TOKENS,
            },
            "elapsed_s": elapsed,
        }

    return _retry(_call)


def run_google_prompt(
    prompt: str,
    model_id: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """Run a single prompt through the Google Gemini API (AI Studio)."""
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise ImportError(
            "google-genai package required: pip install google-genai"
        )

    client = genai.Client(api_key=_require_env("GOOGLE_API_KEY"))

    def _call():
        t0 = time.time()
        result = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        elapsed = round(time.time() - t0, 3)
        response = result.text or ""
        return {
            "model": model_id,
            "response": response,
            "generation_kwargs": _gen_kwargs(max_tokens, temperature),
            "elapsed_s": elapsed,
        }

    return _retry(_call)


# ---------------------------------------------------------------------------
# Single-call adapter — OpenRouter
# ---------------------------------------------------------------------------

def run_openrouter_prompt(
    prompt: str,
    model_id: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """
    Run a single prompt through the OpenRouter API (OpenAI-compatible endpoint).

    OpenRouter routes requests to third-party providers (Google, Meta, etc.) via
    a unified OpenAI-compatible interface. No daily request quotas apply from the
    caller side; billing is credit-based per token.

    Recommended model ID: google/gemini-3.1-pro-preview-20260219
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_require_env("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://github.com/finbench-project",
            "X-Title": "FIN-bench-v2 evaluation",
        },
    )

    def _call():
        t0 = time.time()
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = round(time.time() - t0, 3)
        response = completion.choices[0].message.content or ""
        return {
            "model": model_id,
            "response": response,
            "generation_kwargs": _gen_kwargs(max_tokens, temperature),
            "elapsed_s": elapsed,
        }

    return _retry(_call)


# ---------------------------------------------------------------------------
# Batch — OpenAI
# ---------------------------------------------------------------------------

def submit_openai_batch(
    items: list,
    model_id: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """
    Submit a batch job to the OpenAI Batch API.

    Returns a metadata dict that must be saved to disk (as JSON) so that
    fetch_openai_batch() can retrieve the results later.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))

    batch_requests = [
        {
            "custom_id": item["id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_id,
                "messages": [{"role": "user", "content": item["prompt"]}],
                "max_completion_tokens": max_tokens,
                "temperature": temperature,
            },
        }
        for item in items
    ]

    file_bytes = "\n".join(json.dumps(r) for r in batch_requests).encode("utf-8")
    uploaded = client.files.create(
        file=("batch_input.jsonl", file_bytes), purpose="batch"
    )

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "FIN-bench-v2 evaluation"},
    )

    meta = {
        "provider": "openai",
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "model_id": model_id,
        "n_items": len(items),
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": batch.status,
        "generation_kwargs": _gen_kwargs(max_tokens, temperature),
    }
    print(f"OpenAI batch submitted: {batch.id}  ({len(items)} items, model={model_id})")
    return meta


def poll_openai_batch(batch_id: str) -> dict:
    """Return current status of an OpenAI batch job."""
    from openai import OpenAI

    client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    batch = client.batches.retrieve(batch_id)
    rc = batch.request_counts
    return {
        "provider": "openai",
        "batch_id": batch_id,
        "status": batch.status,
        "output_file_id": batch.output_file_id,
        "request_counts": {
            "total": rc.total if rc else None,
            "completed": rc.completed if rc else None,
            "failed": rc.failed if rc else None,
        },
    }


def fetch_openai_batch(batch_id: str, output_file_id: str) -> list:
    """
    Download completed OpenAI batch results.

    Returns a list of partial records:
        {"id": str, "response": str, "elapsed_s": None, "model": str,
         "error": str | None}

    Failed items have response="" and error set to the error message.
    The caller should write these with a run_meta flag so they are not
    silently scored as 0.
    """
    from openai import OpenAI

    client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    content = client.files.content(output_file_id).text

    results = []
    n_errors = 0
    for line in content.strip().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        item_id = rec["custom_id"]
        error_body = rec.get("error")
        resp = rec.get("response", {})

        if error_body or resp.get("status_code", 200) >= 400:
            error_msg = str(error_body) if error_body else f"HTTP {resp.get('status_code')}"
            results.append({
                "id": item_id,
                "response": "",
                "elapsed_s": None,
                "model": "",
                "generation_kwargs": {},
                "error": error_msg,
            })
            n_errors += 1
        else:
            body = resp.get("body", {})
            choices = body.get("choices", [])
            response_text = choices[0]["message"]["content"] if choices else ""
            results.append({
                "id": item_id,
                "response": response_text,
                "elapsed_s": None,
                "model": body.get("model", ""),
                "generation_kwargs": {},
                "error": None,
            })

    if n_errors:
        print(f"WARNING: {n_errors}/{len(results)} batch items had errors — "
              f"check output for items with error field set.")
    return results


# ---------------------------------------------------------------------------
# Batch — Anthropic
# ---------------------------------------------------------------------------

def submit_anthropic_batch(
    items: list,
    model_id: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> dict:
    """
    Submit a batch job to the Anthropic Message Batches API.

    Returns a metadata dict to be saved as JSON for later retrieval.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required: pip install anthropic")

    client = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))

    def _make_params(item):
        p = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": item["prompt"]}],
        }
        if model_id not in _ANTHROPIC_NO_TEMPERATURE:
            p["temperature"] = temperature
        return p

    requests = [
        {"custom_id": item["id"], "params": _make_params(item)}
        for item in items
    ]

    batch = client.messages.batches.create(requests=requests)

    meta = {
        "provider": "anthropic",
        "batch_id": batch.id,
        "model_id": model_id,
        "n_items": len(items),
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": batch.processing_status,
        "generation_kwargs": _gen_kwargs(max_tokens, temperature),
    }
    print(f"Anthropic batch submitted: {batch.id}  ({len(items)} items, model={model_id})")
    return meta


def submit_anthropic_thinking_batch(
    items: list,
    model_id: str,
    max_tokens: int = 2048,
    temperature: float = 1.0,  # required for extended thinking
) -> dict:
    """
    Submit a batch job to the Anthropic Message Batches API with extended thinking.

    budget_tokens is hardcoded to 1024 (minimum allowed); max_tokens must be > budget_tokens.
    temperature must be 1 for extended thinking — the value passed here is ignored.
    Returns a metadata dict compatible with poll_anthropic_batch() and fetch_anthropic_batch().
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required: pip install anthropic")

    BUDGET_TOKENS = 1024
    if max_tokens <= BUDGET_TOKENS:
        max_tokens = BUDGET_TOKENS + 512

    client = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))

    requests = [
        {
            "custom_id": item["id"],
            "params": {
                "model": model_id,
                "max_tokens": max_tokens,
                "temperature": 1,
                "thinking": {"type": "enabled", "budget_tokens": BUDGET_TOKENS},
                "messages": [{"role": "user", "content": item["prompt"]}],
            },
        }
        for item in items
    ]

    batch = client.messages.batches.create(requests=requests)

    meta = {
        "provider": "anthropic",   # stored as "anthropic" so poll/fetch work unchanged
        "batch_id": batch.id,
        "model_id": model_id,
        "n_items": len(items),
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": batch.processing_status,
        "generation_kwargs": {
            "max_tokens": max_tokens,
            "temperature": 1,
            "thinking_budget_tokens": BUDGET_TOKENS,
        },
    }
    print(f"Anthropic thinking batch submitted: {batch.id}  ({len(items)} items, model={model_id})")
    return meta


def run_anthropic_adaptive_thinking_prompt(
    prompt: str,
    model_id: str,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    effort: str = "high",
) -> dict:
    """
    Run a single prompt through the Anthropic Messages API with adaptive thinking.

    Adaptive thinking (thinking={"type": "adaptive"}) lets the model decide whether
    to think and how much. Supports temperature=0.0 (unlike extended thinking).
    Use output_config={"effort": effort} to control reasoning depth.
    Note: extended thinking (budget_tokens) is NOT supported on Opus 4.7+; use this instead.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required: pip install anthropic")

    client = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))

    def _call():
        t0 = time.time()
        kwargs = dict(
            model=model_id,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}],
        )
        if model_id not in _ANTHROPIC_NO_TEMPERATURE:
            kwargs["temperature"] = temperature
        message = client.messages.create(**kwargs)
        elapsed = round(time.time() - t0, 3)
        text_blocks = [b for b in (message.content or [])
                       if getattr(b, "type", None) == "text"]
        response = text_blocks[0].text if text_blocks else ""
        return {
            "model": model_id,
            "response": response,
            "generation_kwargs": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "thinking_type": "adaptive",
                "thinking_effort": effort,
            },
            "elapsed_s": elapsed,
        }

    return _retry(_call)


def submit_anthropic_adaptive_thinking_batch(
    items: list,
    model_id: str,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    effort: str = "high",
) -> dict:
    """
    Submit a batch job to the Anthropic Message Batches API with adaptive thinking.

    Uses thinking={"type": "adaptive"} and output_config={"effort": effort}.
    Supports temperature=0.0 (unlike extended thinking which requires temperature=1).
    Recommended for Opus 4.7+ where extended thinking (budget_tokens) is not supported.
    Returns a metadata dict compatible with poll_anthropic_batch() and fetch_anthropic_batch().
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required: pip install anthropic")

    client = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))

    def _make_params(item):
        p = {
            "model": model_id,
            "max_tokens": max_tokens,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
            "messages": [{"role": "user", "content": item["prompt"]}],
        }
        if model_id not in _ANTHROPIC_NO_TEMPERATURE:
            p["temperature"] = temperature
        return p

    requests = [
        {"custom_id": item["id"], "params": _make_params(item)}
        for item in items
    ]

    batch = client.messages.batches.create(requests=requests)

    meta = {
        "provider": "anthropic",   # stored as "anthropic" so poll/fetch work unchanged
        "batch_id": batch.id,
        "model_id": model_id,
        "n_items": len(items),
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": batch.processing_status,
        "generation_kwargs": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking_type": "adaptive",
            "thinking_effort": effort,
        },
    }
    print(f"Anthropic adaptive thinking batch submitted: {batch.id}  ({len(items)} items, model={model_id})")
    return meta


def poll_anthropic_batch(batch_id: str) -> dict:
    """Return current status of an Anthropic batch job."""
    import anthropic

    client = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
    batch = client.messages.batches.retrieve(batch_id)
    rc = batch.request_counts
    return {
        "provider": "anthropic",
        "batch_id": batch_id,
        "status": batch.processing_status,
        "request_counts": {
            "processing": rc.processing if rc else None,
            "succeeded": rc.succeeded if rc else None,
            "errored": rc.errored if rc else None,
        },
    }


def fetch_anthropic_batch(batch_id: str) -> list:
    """
    Stream completed Anthropic batch results.

    Returns a list of partial records:
        {"id": str, "response": str, "elapsed_s": None, "model": str,
         "error": str | None}

    Failed or expired items have response="" and error set.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))

    results = []
    n_errors = 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            content = result.result.message.content or []
            text_blocks = [b for b in content if getattr(b, "type", None) == "text"]
            response_text = text_blocks[0].text if text_blocks else ""
            results.append({
                "id": result.custom_id,
                "response": response_text,
                "elapsed_s": None,
                "model": "",
                "generation_kwargs": {},
                "error": None,
            })
        else:
            # "errored" or "expired"
            error_msg = result.result.type
            results.append({
                "id": result.custom_id,
                "response": "",
                "elapsed_s": None,
                "model": "",
                "generation_kwargs": {},
                "error": error_msg,
            })
            n_errors += 1

    if n_errors:
        print(f"WARNING: {n_errors}/{len(results)} batch items had errors — "
              f"check output for items with error field set.")
    return results


# ---------------------------------------------------------------------------
# Concurrent runner — Google
# (AI Studio has no batch endpoint; thread-pool concurrent calls are equivalent)
# ---------------------------------------------------------------------------

def run_google_concurrent(
    items: list,
    model_id: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    concurrency: int = 5,
) -> list:
    """
    Run multiple Google Gemini prompts concurrently using a thread pool.

    Google AI Studio does not offer a batch endpoint, so concurrent synchronous
    calls are the practical equivalent. Results are returned in original item order.

    Returns a list of partial records:
        {"id": str, "response": str, "elapsed_s": float, "model": str,
         "generation_kwargs": dict, "error": str | None}

    Items that fail after retries have response="" and error set.
    """
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise ImportError(
            "google-genai package required: pip install google-genai"
        )

    client = genai.Client(api_key=_require_env("GOOGLE_API_KEY"))
    gen_kw = _gen_kwargs(max_tokens, temperature)

    def call_one(item: dict) -> dict:
        def _call():
            t0 = time.time()
            result = client.models.generate_content(
                model=model_id,
                contents=item["prompt"],
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            elapsed = round(time.time() - t0, 3)
            return result.text or "", elapsed

        try:
            response, elapsed = _retry(_call)
            return {
                "id": item["id"],
                "response": response,
                "elapsed_s": elapsed,
                "model": model_id,
                "generation_kwargs": gen_kw,
                "error": None,
            }
        except Exception as exc:
            return {
                "id": item["id"],
                "response": "",
                "elapsed_s": None,
                "model": model_id,
                "generation_kwargs": gen_kw,
                "error": str(exc),
            }

    # Build id→index map for ordering results
    order = {item["id"]: idx for idx, item in enumerate(items)}
    results_unordered = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_id = {executor.submit(call_one, item): item["id"] for item in items}
        for future in tqdm(as_completed(future_to_id), total=len(items),
                           desc=f"google/{model_id} (concurrency={concurrency})"):
            results_unordered.append(future.result())

    results_unordered.sort(key=lambda r: order.get(r["id"], 999_999))

    n_errors = sum(1 for r in results_unordered if r.get("error"))
    if n_errors:
        print(f"WARNING: {n_errors}/{len(results_unordered)} items failed after retries.")

    return results_unordered


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

class Provider:
    """
    Abstract base class for API provider adapters.

    Each concrete subclass wraps the low-level adapter functions above and
    exposes a uniform interface so that run_frontier_jsonl.py can dispatch
    without per-provider if/elif chains.
    """

    #: True if the provider supports asynchronous batch submission.
    supports_batch: bool = False

    def call_single(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Run one prompt and return the standard result dict."""
        raise NotImplementedError

    def call_batch_submit(
        self,
        items: list,
        model_id: str,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Submit a batch job and return metadata dict."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support batch submission")

    def poll_batch(self, batch_id: str) -> dict:
        """Return current batch status dict."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support batch polling")

    def fetch_batch(self, batch_id: str, output_file_id: str | None = None) -> list:
        """Download completed batch results as a list of partial records."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support batch fetch")

    def batch_is_complete(self, status: dict) -> bool:
        """Return True when the batch has finished processing (success or failure)."""
        raise NotImplementedError

    def batch_is_failed(self, status: dict) -> bool:
        """Return True when the batch ended in a terminal failure state."""
        return False


class OpenAIProvider(Provider):
    """Standard OpenAI Chat Completions (no reasoning)."""

    supports_batch = True

    def call_single(self, prompt, model_id, max_tokens, temperature):
        return run_openai_prompt(prompt, model_id, max_tokens, temperature)

    def call_batch_submit(self, items, model_id, max_tokens, temperature):
        return submit_openai_batch(items, model_id, max_tokens, temperature)

    def poll_batch(self, batch_id):
        return poll_openai_batch(batch_id)

    def fetch_batch(self, batch_id, output_file_id=None):
        return fetch_openai_batch(batch_id, output_file_id)

    def batch_is_complete(self, status):
        return status.get("status") == "completed"

    def batch_is_failed(self, status):
        return status.get("status") in {"failed", "cancelled", "expired"}


class OpenAIThinkingProvider(Provider):
    """OpenAI Responses API with reasoning enabled (GPT-5.4 CoT)."""

    supports_batch = False   # no batch endpoint for Responses API

    def call_single(self, prompt, model_id, max_tokens, temperature):
        return run_openai_thinking_prompt(prompt, model_id, max_tokens, temperature)


class AnthropicProvider(Provider):
    """Standard Anthropic Messages API (no extended thinking)."""

    supports_batch = True

    def call_single(self, prompt, model_id, max_tokens, temperature):
        return run_anthropic_prompt(prompt, model_id, max_tokens, temperature)

    def call_batch_submit(self, items, model_id, max_tokens, temperature):
        return submit_anthropic_batch(items, model_id, max_tokens, temperature)

    def poll_batch(self, batch_id):
        return poll_anthropic_batch(batch_id)

    def fetch_batch(self, batch_id, output_file_id=None):
        return fetch_anthropic_batch(batch_id)

    def batch_is_complete(self, status):
        return status.get("status") == "ended"


class AnthropicThinkingProvider(Provider):
    """Anthropic Messages API with extended thinking (Claude CoT)."""

    supports_batch = True

    def call_single(self, prompt, model_id, max_tokens, temperature):
        return run_anthropic_thinking_prompt(prompt, model_id, max_tokens, temperature)

    def call_batch_submit(self, items, model_id, max_tokens, temperature):
        return submit_anthropic_thinking_batch(items, model_id, max_tokens, temperature)

    def poll_batch(self, batch_id):
        return poll_anthropic_batch(batch_id)

    def fetch_batch(self, batch_id, output_file_id=None):
        return fetch_anthropic_batch(batch_id)

    def batch_is_complete(self, status):
        return status.get("status") == "ended"


class AnthropicAdaptiveThinkingProvider(Provider):
    """Anthropic Messages API with adaptive thinking (Claude Opus 4.7+).

    Uses thinking={"type": "adaptive"} instead of extended thinking (budget_tokens),
    which is not supported on Opus 4.7+. Supports temperature=0.0.
    """

    supports_batch = True

    def call_single(self, prompt, model_id, max_tokens, temperature):
        return run_anthropic_adaptive_thinking_prompt(prompt, model_id, max_tokens, temperature)

    def call_batch_submit(self, items, model_id, max_tokens, temperature):
        return submit_anthropic_adaptive_thinking_batch(items, model_id, max_tokens, temperature)

    def poll_batch(self, batch_id):
        return poll_anthropic_batch(batch_id)

    def fetch_batch(self, batch_id, output_file_id=None):
        return fetch_anthropic_batch(batch_id)

    def batch_is_complete(self, status):
        return status.get("status") == "ended"


class GoogleProvider(Provider):
    """Google Gemini via AI Studio (concurrent calls only — no batch endpoint)."""

    supports_batch = False

    def call_single(self, prompt, model_id, max_tokens, temperature):
        return run_google_prompt(prompt, model_id, max_tokens, temperature)


class OpenRouterProvider(Provider):
    """OpenRouter API — OpenAI-compatible endpoint for third-party models."""

    supports_batch = False

    def call_single(self, prompt, model_id, max_tokens, temperature):
        return run_openrouter_prompt(prompt, model_id, max_tokens, temperature)


#: Registry mapping CLI provider names to Provider instances.
PROVIDER_REGISTRY: dict[str, Provider] = {
    "openai":                        OpenAIProvider(),
    "openai-thinking":               OpenAIThinkingProvider(),
    "anthropic":                     AnthropicProvider(),
    "anthropic-thinking":            AnthropicThinkingProvider(),
    "anthropic-adaptive-thinking":   AnthropicAdaptiveThinkingProvider(),
    "google":                        GoogleProvider(),
    "openrouter":                    OpenRouterProvider(),
}


def get_provider(name: str) -> Provider:
    """
    Return the Provider instance for the given CLI name.

    Raises ValueError for unknown names so callers get a clear error
    instead of a confusing AttributeError.
    """
    if name not in PROVIDER_REGISTRY:
        available = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return PROVIDER_REGISTRY[name]

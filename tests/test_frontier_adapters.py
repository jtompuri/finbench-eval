"""
Tests for scripts/frontier_adapters.py — retry logic, Provider registry,
and Provider interface. No live API calls are made.
"""
import pytest
from frontier_adapters import (
    PROVIDER_REGISTRY,
    AnthropicProvider,
    AnthropicThinkingProvider,
    GoogleProvider,
    OpenAIProvider,
    OpenAIThinkingProvider,
    OpenRouterProvider,
    get_provider,
    _is_retryable,
    _is_daily_quota_error,
    _suggested_wait,
)


# ---------------------------------------------------------------------------
# _is_retryable
# ---------------------------------------------------------------------------

class TestIsRetryable:
    def _exc(self, msg="", status_code=None):
        e = Exception(msg)
        if status_code is not None:
            e.status_code = status_code
        return e

    def test_rate_limit_429(self):
        assert _is_retryable(self._exc(status_code=429)) is True

    def test_server_error_500(self):
        assert _is_retryable(self._exc(status_code=500)) is True

    def test_overloaded_529(self):
        assert _is_retryable(self._exc(status_code=529)) is True

    def test_auth_401_not_retryable(self):
        assert _is_retryable(self._exc(status_code=401)) is False

    def test_forbidden_403_not_retryable(self):
        assert _is_retryable(self._exc(status_code=403)) is False

    def test_not_found_404_not_retryable(self):
        assert _is_retryable(self._exc(status_code=404)) is False

    def test_rate_limit_message_fallback(self):
        assert _is_retryable(self._exc("rate limit exceeded")) is True

    def test_timeout_message_fallback(self):
        assert _is_retryable(self._exc("request timed out")) is True

    def test_connection_error_fallback(self):
        assert _is_retryable(self._exc("connection reset")) is True

    def test_generic_error_not_retryable(self):
        assert _is_retryable(self._exc("invalid model")) is False


# ---------------------------------------------------------------------------
# _is_daily_quota_error
# ---------------------------------------------------------------------------

class TestIsDailyQuotaError:
    def test_per_day_keyword(self):
        assert _is_daily_quota_error(Exception("per_day limit reached")) is True

    def test_generate_requests_per_day(self):
        assert _is_daily_quota_error(Exception("GenerateRequestsPerDay exceeded")) is True

    def test_regular_rate_limit_not_quota(self):
        assert _is_daily_quota_error(Exception("rate limit exceeded")) is False


# ---------------------------------------------------------------------------
# _suggested_wait
# ---------------------------------------------------------------------------

class TestSuggestedWait:
    def test_extracts_seconds(self):
        exc = Exception("Resource exhausted, retry in 30s")
        assert _suggested_wait(exc, default=5.0) == pytest.approx(32.0)  # +2s buffer

    def test_falls_back_to_default(self):
        exc = Exception("some error with no retry hint")
        assert _suggested_wait(exc, default=10.0) == pytest.approx(10.0)

    def test_capped_at_max(self):
        exc = Exception("retry in 120s")
        result = _suggested_wait(exc, default=5.0)
        assert result <= 60.0  # _MAX_RETRY_WAIT_S


# ---------------------------------------------------------------------------
# Provider registry and get_provider
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    def test_all_providers_registered(self):
        expected = {"openai", "openai-thinking", "anthropic", "anthropic-thinking",
                    "google", "openrouter"}
        assert set(PROVIDER_REGISTRY.keys()) == expected

    def test_get_provider_openai(self):
        assert isinstance(get_provider("openai"), OpenAIProvider)

    def test_get_provider_openai_thinking(self):
        assert isinstance(get_provider("openai-thinking"), OpenAIThinkingProvider)

    def test_get_provider_anthropic(self):
        assert isinstance(get_provider("anthropic"), AnthropicProvider)

    def test_get_provider_anthropic_thinking(self):
        assert isinstance(get_provider("anthropic-thinking"), AnthropicThinkingProvider)

    def test_get_provider_google(self):
        assert isinstance(get_provider("google"), GoogleProvider)

    def test_get_provider_openrouter(self):
        assert isinstance(get_provider("openrouter"), OpenRouterProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")


# ---------------------------------------------------------------------------
# Provider.supports_batch
# ---------------------------------------------------------------------------

class TestProviderBatchSupport:
    def test_openai_supports_batch(self):
        assert get_provider("openai").supports_batch is True

    def test_anthropic_supports_batch(self):
        assert get_provider("anthropic").supports_batch is True

    def test_anthropic_thinking_supports_batch(self):
        assert get_provider("anthropic-thinking").supports_batch is True

    def test_openai_thinking_no_batch(self):
        assert get_provider("openai-thinking").supports_batch is False

    def test_google_no_batch(self):
        assert get_provider("google").supports_batch is False

    def test_openrouter_no_batch(self):
        assert get_provider("openrouter").supports_batch is False


# ---------------------------------------------------------------------------
# Provider.batch_is_complete / batch_is_failed
# ---------------------------------------------------------------------------

class TestProviderBatchStatus:
    def test_openai_complete(self):
        prov = get_provider("openai")
        assert prov.batch_is_complete({"status": "completed"}) is True
        assert prov.batch_is_complete({"status": "in_progress"}) is False

    def test_openai_failed(self):
        prov = get_provider("openai")
        assert prov.batch_is_failed({"status": "failed"}) is True
        assert prov.batch_is_failed({"status": "cancelled"}) is True
        assert prov.batch_is_failed({"status": "expired"}) is True
        assert prov.batch_is_failed({"status": "completed"}) is False

    def test_anthropic_complete(self):
        prov = get_provider("anthropic")
        assert prov.batch_is_complete({"status": "ended"}) is True
        assert prov.batch_is_complete({"status": "in_progress"}) is False

    def test_anthropic_not_failed_by_default(self):
        prov = get_provider("anthropic")
        assert prov.batch_is_failed({"status": "ended"}) is False

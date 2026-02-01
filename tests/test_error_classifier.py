"""
Direct tests for ErrorClassifier — pattern matching, severity, recovery hints.
"""
import pytest
import json

from airos.errors import (
    ErrorClassifier, ClassifiedError,
    ErrorCategory, ErrorSeverity,
    AirOSError, BudgetExceededError, TimeoutExceededError,
    RecoveryError, ConfigurationError, ProviderError, MedicError,
)
from airos.fuse import LoopError


# ── Classification by error type ─────────────────────────────────────────

class TestErrorClassification:

    def test_json_decode_error(self):
        try:
            json.loads("{bad json")
        except json.JSONDecodeError as e:
            result = ErrorClassifier.classify(e)
            assert result.category == ErrorCategory.JSON_PARSE
            assert result.severity == ErrorSeverity.LOW
            assert result.recoverable is True

    def test_validation_error_keyword(self):
        err = Exception("Sentinel Alert: Output validation failed: missing name")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.SCHEMA_VALIDATION

    def test_type_error(self):
        err = TypeError("cannot convert string to int")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.TYPE_MISMATCH

    def test_timeout_error(self):
        err = Exception("connection timed out after 30s")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.NETWORK_TIMEOUT

    def test_rate_limit_error(self):
        err = Exception("429 too many requests — rate limit exceeded")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.RATE_LIMIT

    def test_api_500_error(self):
        err = Exception("500 Internal Server Error from API")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.API_ERROR

    def test_auth_error(self):
        err = Exception("401 unauthorized: invalid api key")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.AUTHENTICATION
        assert result.recoverable is False

    def test_context_length_error(self):
        err = Exception("context_length_exceeded: maximum token limit reached")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.CONTEXT_LENGTH

    def test_content_filter_error(self):
        err = Exception("content filter triggered: inappropriate content blocked")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.CONTENT_FILTER
        assert result.severity == ErrorSeverity.HIGH
        # CONTENT_FILTER is not in NON_RECOVERABLE, so it's technically recoverable
        assert result.recoverable is True

    def test_memory_error(self):
        err = MemoryError("out of memory")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.MEMORY
        assert result.severity == ErrorSeverity.CRITICAL
        assert result.recoverable is False

    def test_loop_error(self):
        err = LoopError("Fuse Tripped: Loop detected. State repeated 3 times.")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.LOOP_DETECTED
        assert result.recoverable is False

    def test_key_error(self):
        err = KeyError("missing_key")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.MISSING_FIELD

    def test_none_attribute_error(self):
        err = AttributeError("'NoneType' object has no attribute 'get'")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.NULL_REFERENCE

    def test_model_overload(self):
        err = Exception("model is overloaded, try again later")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.MODEL_OVERLOAD
        assert result.severity == ErrorSeverity.LOW

    def test_unknown_error(self):
        err = Exception("some completely random error xyz123")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.UNKNOWN
        assert result.severity == ErrorSeverity.MEDIUM

    def test_empty_message(self):
        err = Exception("")
        result = ErrorClassifier.classify(err)
        assert result.category == ErrorCategory.UNKNOWN


# ── ClassifiedError metadata ─────────────────────────────────────────────

class TestClassifiedErrorMetadata:

    def test_to_dict(self):
        err = ValueError("bad value")
        classified = ErrorClassifier.classify(err)
        d = classified.to_dict()
        assert "category" in d
        assert "severity" in d
        assert "message" in d
        assert "recoverable" in d
        assert "original_type" in d
        assert d["original_type"] == "ValueError"

    def test_str_representation(self):
        err = ValueError("test error")
        classified = ErrorClassifier.classify(err)
        s = str(classified)
        assert "[" in s  # [category:severity] format

    def test_context_passed_through(self):
        err = ValueError("test")
        classified = ErrorClassifier.classify(err, context={"node_id": "my_node"})
        assert classified.context["node_id"] == "my_node"

    def test_context_defaults_to_empty(self):
        err = ValueError("test")
        classified = ErrorClassifier.classify(err)
        assert classified.context == {}


# ── Severity mapping ─────────────────────────────────────────────────────

class TestSeverityMapping:

    def test_all_categories_have_severity(self):
        for cat in ErrorCategory:
            assert cat in ErrorClassifier.SEVERITY_MAP

    def test_all_categories_have_strategy(self):
        for cat in ErrorCategory:
            assert cat in ErrorClassifier.STRATEGY_MAP


# ── Recovery hints ───────────────────────────────────────────────────────

class TestRecoveryHints:

    def test_json_parse_hint(self):
        err = json.JSONDecodeError("error", "", 0)
        classified = ErrorClassifier.classify(err)
        hint = ErrorClassifier.get_recovery_prompt_hint(classified)
        assert "JSON" in hint

    def test_schema_validation_hint(self):
        err = Exception("Sentinel Alert: validation failed")
        classified = ErrorClassifier.classify(err)
        hint = ErrorClassifier.get_recovery_prompt_hint(classified)
        assert "schema" in hint

    def test_unknown_category_hint(self):
        err = Exception("some random error")
        classified = ErrorClassifier.classify(err)
        hint = ErrorClassifier.get_recovery_prompt_hint(classified)
        assert "error" in hint.lower()

    def test_context_length_hint(self):
        err = Exception("context_length_exceeded")
        classified = ErrorClassifier.classify(err)
        hint = ErrorClassifier.get_recovery_prompt_hint(classified)
        assert "shorter" in hint


# ── Non-recoverable set ─────────────────────────────────────────────────

class TestNonRecoverable:

    def test_non_recoverable_set_correct(self):
        assert ErrorCategory.AUTHENTICATION in ErrorClassifier.NON_RECOVERABLE
        assert ErrorCategory.MEMORY in ErrorClassifier.NON_RECOVERABLE
        assert ErrorCategory.LOOP_DETECTED in ErrorClassifier.NON_RECOVERABLE

    def test_recoverable_not_in_set(self):
        assert ErrorCategory.JSON_PARSE not in ErrorClassifier.NON_RECOVERABLE
        assert ErrorCategory.SCHEMA_VALIDATION not in ErrorClassifier.NON_RECOVERABLE
        assert ErrorCategory.RATE_LIMIT not in ErrorClassifier.NON_RECOVERABLE


# ── AirOS custom error types ────────────────────────────────────────────

class TestCustomErrorTypes:

    def test_airos_error_base(self):
        err = AirOSError("base error")
        assert isinstance(err, Exception)
        assert str(err) == "base error"

    def test_budget_exceeded_attributes(self):
        err = BudgetExceededError("over budget", spent=5.0, limit=3.0)
        assert err.spent == 5.0
        assert err.limit == 3.0
        assert isinstance(err, AirOSError)

    def test_timeout_exceeded_attributes(self):
        err = TimeoutExceededError("too slow", elapsed=10.0, limit=5.0)
        assert err.elapsed == 10.0
        assert err.limit == 5.0
        assert isinstance(err, AirOSError)

    def test_recovery_error_classified(self):
        classified = ErrorClassifier.classify(ValueError("x"))
        err = RecoveryError("failed", classified_error=classified)
        assert err.classified_error == classified

    def test_provider_error_attributes(self):
        orig = ConnectionError("down")
        err = ProviderError("provider failed", provider="openai", original_error=orig)
        assert err.provider == "openai"
        assert err.original_error == orig

    def test_medic_error_is_airos_error(self):
        err = MedicError("medic failed")
        assert isinstance(err, AirOSError)

    def test_budget_exceeded_default_attributes(self):
        err = BudgetExceededError("over")
        assert err.spent == 0.0
        assert err.limit == 0.0

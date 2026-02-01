"""
Edge case tests for Medic (LLM recovery).

Covers: no LLM, empty LLM response, non-recoverable errors,
max attempts, parse edge cases, stats tracking.
"""
import json
import pytest
from pydantic import BaseModel

from airos.medic import Medic, RecoveryResult
from airos.errors import MedicError, ErrorClassifier, ErrorCategory, ErrorSeverity
from airos.fuse import LoopError


class SimpleOutput(BaseModel):
    result: str
    score: float


# ── No LLM callable ─────────────────────────────────────────────────────

class TestMedicNoLLM:

    def test_no_llm_reraises_or_repairs(self):
        """Without explicit LLM, Medic either re-raises (no fallback)
        or repairs (Groq fallback from env)."""
        medic = Medic(llm_callable=None)
        err = ValueError("something broke")
        try:
            result = medic.attempt_recovery(
                error=err, input_state={}, raw_output=None,
                node_id="test", recovery_attempts=1, schema=SimpleOutput,
            )
            # If Groq fallback repaired it, that's valid behavior
            assert isinstance(result, dict)
        except (ValueError, MedicError):
            pass  # Expected if no fallback available

    def test_no_explicit_llm_behavior(self):
        """Medic() without llm_callable may use Groq fallback or re-raise."""
        medic = Medic()
        err = TypeError("bad type")
        try:
            result = medic.attempt_recovery(
                error=err, input_state={}, raw_output=None,
                node_id="test", recovery_attempts=1,
            )
            # Groq fallback repaired it
            assert result is not None
        except (TypeError, MedicError):
            pass  # No fallback available


# ── Max attempts exceeded ────────────────────────────────────────────────

class TestMedicMaxAttempts:

    def test_exceeds_max_attempts(self):
        medic = Medic(llm_callable=lambda p: '{"result":"ok","score":1.0}', max_recovery_attempts=2)
        with pytest.raises(MedicError, match="Exceeded 2"):
            medic.attempt_recovery(
                error=ValueError("test"), input_state={}, raw_output=None,
                node_id="test", recovery_attempts=3, schema=SimpleOutput,
            )

    def test_at_max_attempts_still_works(self):
        """Attempt 2 of max 2 should still try."""
        medic = Medic(llm_callable=lambda p: '{"result":"ok","score":1.0}', max_recovery_attempts=2)
        result = medic.attempt_recovery(
            error=ValueError("test"), input_state={}, raw_output=None,
            node_id="test", recovery_attempts=2, schema=SimpleOutput,
        )
        assert result["result"] == "ok"

    def test_attempt_1_works(self):
        medic = Medic(llm_callable=lambda p: '{"result":"fixed","score":0.9}', max_recovery_attempts=2)
        result = medic.attempt_recovery(
            error=ValueError("test"), input_state={}, raw_output=None,
            node_id="test", recovery_attempts=1, schema=SimpleOutput,
        )
        assert result["result"] == "fixed"


# ── Non-recoverable errors ──────────────────────────────────────────────

class TestMedicNonRecoverable:

    def test_loop_error_not_recovered(self):
        """LoopError is classified as non-recoverable."""
        medic = Medic(llm_callable=lambda p: '{"result":"ok","score":1.0}')
        with pytest.raises(LoopError):
            medic.attempt_recovery(
                error=LoopError("Fuse Tripped: Loop detected"),
                input_state={}, raw_output=None,
                node_id="test", recovery_attempts=1,
            )

    def test_auth_error_not_recovered(self):
        """Authentication errors are non-recoverable."""
        medic = Medic(llm_callable=lambda p: '{"result":"ok","score":1.0}')
        err = Exception("401 unauthorized: invalid api key")
        with pytest.raises(Exception, match="unauthorized"):
            medic.attempt_recovery(
                error=err, input_state={}, raw_output=None,
                node_id="test", recovery_attempts=1,
            )

    def test_memory_error_not_recovered(self):
        medic = Medic(llm_callable=lambda p: '{"result":"ok","score":1.0}')
        err = MemoryError("out of memory")
        with pytest.raises(MemoryError):
            medic.attempt_recovery(
                error=err, input_state={}, raw_output=None,
                node_id="test", recovery_attempts=1,
            )


# ── LLM response parsing ────────────────────────────────────────────────

class TestMedicParsing:

    def test_clean_json(self):
        medic = Medic(llm_callable=lambda p: '{"result":"ok","score":1.0}')
        result = medic.attempt_recovery(
            error=ValueError("bad"), input_state={}, raw_output=None,
            node_id="test", recovery_attempts=1, schema=SimpleOutput,
        )
        assert result["result"] == "ok"

    def test_markdown_wrapped_json(self):
        response = '```json\n{"result":"ok","score":1.0}\n```'
        medic = Medic(llm_callable=lambda p: response)
        result = medic.attempt_recovery(
            error=ValueError("bad"), input_state={}, raw_output=None,
            node_id="test", recovery_attempts=1, schema=SimpleOutput,
        )
        assert result["result"] == "ok"

    def test_backtick_wrapped_json(self):
        response = '```\n{"result":"ok","score":1.0}\n```'
        medic = Medic(llm_callable=lambda p: response)
        result = medic.attempt_recovery(
            error=ValueError("bad"), input_state={}, raw_output=None,
            node_id="test", recovery_attempts=1, schema=SimpleOutput,
        )
        assert result["result"] == "ok"

    def test_llm_returns_garbage_raises(self):
        medic = Medic(llm_callable=lambda p: "this is not json at all")
        with pytest.raises(MedicError, match="Repair failed"):
            medic.attempt_recovery(
                error=ValueError("bad"), input_state={}, raw_output=None,
                node_id="test", recovery_attempts=1, schema=SimpleOutput,
            )

    def test_llm_returns_empty_string_raises(self):
        medic = Medic(llm_callable=lambda p: "")
        with pytest.raises(MedicError):
            medic.attempt_recovery(
                error=ValueError("bad"), input_state={}, raw_output=None,
                node_id="test", recovery_attempts=1, schema=SimpleOutput,
            )

    def test_llm_returns_none_raises(self):
        """LLM returns None — json.loads(None) fails."""
        def bad_llm(prompt):
            return None
        medic = Medic(llm_callable=bad_llm)
        with pytest.raises((MedicError, TypeError)):
            medic.attempt_recovery(
                error=ValueError("bad"), input_state={}, raw_output=None,
                node_id="test", recovery_attempts=1, schema=SimpleOutput,
            )

    def test_llm_raises_exception(self):
        """LLM callable itself throws."""
        def failing_llm(prompt):
            raise ConnectionError("API down")
        medic = Medic(llm_callable=failing_llm)
        with pytest.raises(MedicError, match="Repair failed"):
            medic.attempt_recovery(
                error=ValueError("bad"), input_state={}, raw_output=None,
                node_id="test", recovery_attempts=1, schema=SimpleOutput,
            )


# ── Stats tracking ───────────────────────────────────────────────────────

class TestMedicStats:

    def test_initial_stats(self):
        medic = Medic(llm_callable=lambda p: '{"result":"ok","score":1.0}')
        stats = medic.get_stats()
        assert stats["total_attempts"] == 0
        assert stats["successful"] == 0
        assert stats["total_tokens"] == 0

    def test_stats_after_recovery(self):
        medic = Medic(llm_callable=lambda p: '{"result":"ok","score":1.0}')
        medic.attempt_recovery(
            error=ValueError("bad"), input_state={}, raw_output=None,
            node_id="test", recovery_attempts=1, schema=SimpleOutput,
        )
        stats = medic.get_stats()
        assert stats["total_attempts"] >= 1
        assert stats["successful"] >= 1

    def test_recovery_history(self):
        medic = Medic(llm_callable=lambda p: '{"result":"ok","score":1.0}')
        medic.attempt_recovery(
            error=ValueError("bad"), input_state={}, raw_output=None,
            node_id="test", recovery_attempts=1, schema=SimpleOutput,
        )
        history = medic.recovery_history
        assert len(history) >= 1
        assert all(isinstance(r, RecoveryResult) for r in history)

    def test_total_tokens_tracked(self):
        medic = Medic(llm_callable=lambda p: '{"result":"ok","score":1.0}')
        medic.attempt_recovery(
            error=ValueError("bad"), input_state={"x": "y"}, raw_output=None,
            node_id="test", recovery_attempts=1, schema=SimpleOutput,
        )
        assert medic.total_tokens_used >= 0

    def test_total_cost_tracked(self):
        medic = Medic(llm_callable=lambda p: '{"result":"ok","score":1.0}')
        medic.attempt_recovery(
            error=ValueError("bad"), input_state={"x": "y"}, raw_output=None,
            node_id="test", recovery_attempts=1, schema=SimpleOutput,
        )
        assert medic.total_cost >= 0

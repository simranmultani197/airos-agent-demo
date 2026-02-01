"""
Edge case tests for Fuse (loop detection).

Covers: zero/negative limits, None state, large objects, non-serializable,
empty history, hash collisions, unicode.
"""
import pytest

from airos.fuse import Fuse, LoopError


class TestFuseInitEdgeCases:

    def test_default_limit(self):
        fuse = Fuse()
        assert fuse.limit == 3

    def test_limit_of_one(self):
        """Limit=1 means the very first repeated state trips."""
        fuse = Fuse(limit=1)
        state = {"key": "value"}
        h = fuse._hash_state(state)
        # First occurrence in history — trips immediately
        with pytest.raises(LoopError):
            fuse.check([h], state)

    def test_limit_of_zero(self):
        """Limit=0 — any state always trips since count >= 0."""
        fuse = Fuse(limit=0)
        with pytest.raises(LoopError):
            fuse.check([], {"key": "value"})

    def test_negative_limit(self):
        """Negative limit — always trips because count(0) >= -5 is True."""
        fuse = Fuse(limit=-5)
        with pytest.raises(LoopError):
            fuse.check([], {"key": "value"})

    def test_very_large_limit(self):
        fuse = Fuse(limit=1000000)
        fuse.check([], {"key": "value"})  # passes


class TestFuseCheckEdgeCases:

    def test_empty_history_passes(self):
        fuse = Fuse(limit=3)
        fuse.check([], {"data": "test"})

    def test_none_state(self):
        fuse = Fuse(limit=3)
        fuse.check([], None)  # str(None) hashes fine

    def test_empty_dict_state(self):
        fuse = Fuse(limit=3)
        h = fuse._hash_state({})
        fuse.check([h, h], {})  # 2 times, limit 3 — passes
        with pytest.raises(LoopError):
            fuse.check([h, h, h], {})  # 3 times — trips

    def test_empty_string_state(self):
        fuse = Fuse(limit=3)
        fuse.check([], "")

    def test_boolean_state(self):
        fuse = Fuse(limit=3)
        fuse.check([], True)
        fuse.check([], False)

    def test_integer_state(self):
        fuse = Fuse(limit=3)
        fuse.check([], 42)

    def test_list_state(self):
        fuse = Fuse(limit=3)
        fuse.check([], [1, 2, 3])

    def test_nested_dict_state(self):
        fuse = Fuse(limit=3)
        state = {"a": {"b": {"c": [1, 2, {"d": "deep"}]}}}
        fuse.check([], state)

    def test_different_states_no_loop(self):
        """Different states should never trip."""
        fuse = Fuse(limit=2)
        states = [{"i": i} for i in range(100)]
        history = []
        for s in states:
            fuse.check(history, s)
            history.append(fuse._hash_state(s))

    def test_loop_error_message(self):
        fuse = Fuse(limit=2)
        state = {"x": 1}
        h = fuse._hash_state(state)
        with pytest.raises(LoopError, match="Loop detected"):
            fuse.check([h, h], state)


class TestFuseHashEdgeCases:

    def test_hash_deterministic(self):
        fuse = Fuse()
        state = {"key": "value", "num": 42}
        h1 = fuse._hash_state(state)
        h2 = fuse._hash_state(state)
        assert h1 == h2

    def test_hash_key_order_independent(self):
        """JSON sort_keys makes key order irrelevant."""
        fuse = Fuse()
        h1 = fuse._hash_state({"a": 1, "b": 2})
        h2 = fuse._hash_state({"b": 2, "a": 1})
        assert h1 == h2

    def test_hash_different_values_differ(self):
        fuse = Fuse()
        h1 = fuse._hash_state({"a": 1})
        h2 = fuse._hash_state({"a": 2})
        assert h1 != h2

    def test_hash_none(self):
        fuse = Fuse()
        h = fuse._hash_state(None)
        assert isinstance(h, str) and len(h) > 0

    def test_hash_large_object(self):
        fuse = Fuse()
        state = {"data": "x" * 100000}
        h = fuse._hash_state(state)
        assert isinstance(h, str) and len(h) == 64  # sha256 hex

    def test_hash_unicode(self):
        fuse = Fuse()
        h = fuse._hash_state({"text": "こんにちは世界 🌍"})
        assert isinstance(h, str) and len(h) == 64

    def test_hash_non_serializable_fallback(self):
        """Non-JSON-serializable objects use str() fallback."""
        fuse = Fuse()

        class Custom:
            def __str__(self):
                return "custom_object"

        h = fuse._hash_state(Custom())
        assert isinstance(h, str) and len(h) > 0

    def test_hash_bytes_object(self):
        fuse = Fuse()
        h = fuse._hash_state(b"raw bytes")
        assert isinstance(h, str) and len(h) > 0

    def test_hash_set_object(self):
        """Sets aren't JSON serializable — uses fallback."""
        fuse = Fuse()
        h = fuse._hash_state({1, 2, 3})
        assert isinstance(h, str)

    def test_hash_float_nan(self):
        fuse = Fuse()
        h = fuse._hash_state({"val": float('nan')})
        assert isinstance(h, str)

    def test_hash_float_inf(self):
        fuse = Fuse()
        h = fuse._hash_state({"val": float('inf')})
        assert isinstance(h, str)

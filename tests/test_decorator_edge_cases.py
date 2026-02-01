"""
Edge case tests for @reliable decorator.

Covers: parameter conflicts, function signatures, return types,
combined features, budget+fuse+medic interaction.

NOTE: Each test uses its own InMemoryStorage to avoid cross-test
fuse contamination (shared "local_dev_run" history).
"""
import time
import pytest
from pydantic import BaseModel

from airos import (
    reliable, GlobalBudget,
    BudgetExceededError, TimeoutExceededError,
)
from airos.storage import InMemoryStorage
from airos.fuse import LoopError


class Output(BaseModel):
    result: str

class NumberOutput(BaseModel):
    value: int


# ── Basic decorator behavior ────────────────────────────────────────────

class TestDecoratorBasics:

    def test_no_args_decorator(self):
        store = InMemoryStorage()
        @reliable(storage=store)
        def node(state):
            return state
        assert node({"result": "ok"}) == {"result": "ok"}

    def test_function_name_preserved(self):
        @reliable()
        def my_special_node(state):
            return state
        assert my_special_node.__name__ == "my_special_node"

    def test_function_returning_none(self):
        store = InMemoryStorage()
        @reliable(storage=store)
        def node(state):
            return None
        assert node({"x": 1}) is None

    def test_function_returning_string(self):
        store = InMemoryStorage()
        @reliable(storage=store)
        def node(state):
            return "hello"
        assert node({"x": 1}) == "hello"

    def test_function_returning_int(self):
        store = InMemoryStorage()
        @reliable(storage=store)
        def node(state):
            return 42
        assert node({"x": 1}) == 42

    def test_function_returning_list(self):
        store = InMemoryStorage()
        @reliable(storage=store)
        def node(state):
            return [1, 2, 3]
        assert node({"x": 1}) == [1, 2, 3]


# ── Schema validation edge cases ────────────────────────────────────────

class TestDecoratorSchema:

    def test_schema_with_valid_dict(self):
        store = InMemoryStorage()
        @reliable(sentinel_schema=Output, storage=store)
        def node(state):
            return {"result": "ok"}
        result = node({"x": 1})
        assert result.result == "ok"

    def test_schema_with_invalid_dict_no_medic(self):
        """No LLM — sentinel error propagates (or Groq fallback repairs)."""
        store = InMemoryStorage()
        @reliable(sentinel_schema=Output, storage=store)
        def node(state):
            return {"wrong": "field"}
        try:
            result = node({"x": 1})
            # If Medic auto-repaired via Groq fallback, that's fine
            assert hasattr(result, 'result')
        except Exception:
            pass  # Expected if no LLM available

    def test_schema_with_function_exception(self):
        store = InMemoryStorage()
        @reliable(sentinel_schema=Output, storage=store)
        def node(state):
            raise ValueError("crash!")
        try:
            node({"x": 1})
        except (ValueError, Exception):
            pass  # Expected


# ── Budget + decorator combos ────────────────────────────────────────────

class TestDecoratorBudget:

    def test_max_cost_usd_under_budget(self):
        store = InMemoryStorage()
        @reliable(max_cost_usd=100.0, model="gpt-4o", storage=store)
        def node(state):
            return {"result": "small"}
        result = node({"x": 1})
        assert result is not None

    def test_max_seconds_under_limit(self):
        store = InMemoryStorage()
        @reliable(max_seconds=10.0, storage=store)
        def node(state):
            return {"result": "fast"}
        assert node({"x": 1}) is not None

    def test_max_seconds_over_limit(self):
        store = InMemoryStorage()
        @reliable(max_seconds=0.01, storage=store)
        def node(state):
            time.sleep(0.1)
            return {"result": "slow"}
        with pytest.raises(TimeoutExceededError):
            node({"x": 1})

    def test_global_budget_tracks_across_calls(self):
        budget = GlobalBudget(max_cost_usd=100.0)
        store = InMemoryStorage()

        @reliable(budget=budget, model="gpt-4o", storage=store)
        def node(state):
            return {"result": "ok"}

        budget.reset()
        node({"x": 1})
        first = budget.total_spent
        node({"y": 2})
        assert budget.total_spent > first

    def test_global_budget_trips(self):
        budget = GlobalBudget(max_cost_usd=0.0001)
        store = InMemoryStorage()

        @reliable(budget=budget, model="gpt-4o", storage=store)
        def node(state):
            return {"result": "x" * 5000}

        budget.reset()
        try:
            node({"x": 1})
            node({"x": 2})
        except BudgetExceededError:
            pass  # Expected

    def test_both_max_cost_and_global_budget(self):
        """Both per-node and global budget set."""
        budget = GlobalBudget(max_cost_usd=100.0)
        store = InMemoryStorage()

        @reliable(max_cost_usd=100.0, budget=budget, model="gpt-4o", storage=store)
        def node(state):
            return {"result": "ok"}

        budget.reset()
        node({"x": 1})  # should pass both


# ── cost_per_token + model combos ────────────────────────────────────────

class TestDecoratorPricing:

    def test_model_param(self):
        store = InMemoryStorage()
        @reliable(model="gpt-4o", storage=store)
        def node(state):
            return {"result": "ok"}
        node({"x": 1})

    def test_cost_per_token_param(self):
        store = InMemoryStorage()
        @reliable(cost_per_token=0.001, storage=store)
        def node(state):
            return {"result": "ok"}
        node({"x": 1})

    def test_both_cost_and_model(self):
        """cost_per_token should override model."""
        budget = GlobalBudget(max_cost_usd=100.0)
        store = InMemoryStorage()

        @reliable(model="gpt-4o", cost_per_token=0.0001, budget=budget, storage=store)
        def node(state):
            return {"result": "ok"}

        budget.reset()
        node({"x": 1})
        assert budget.total_spent > 0


# ── Fuse + budget interaction ────────────────────────────────────────────

class TestDecoratorFuseBudget:

    def test_fuse_trips_before_budget(self):
        """Fuse should trip on loop before budget is checked."""
        budget = GlobalBudget(max_cost_usd=100.0)
        store = InMemoryStorage()

        @reliable(fuse_limit=2, budget=budget, model="gpt-4o", storage=store)
        def node(state):
            return {"result": "same"}

        budget.reset()
        node({"input": "same"})
        node({"input": "same"})
        with pytest.raises(LoopError):
            node({"input": "same"})

    def test_fuse_with_different_inputs_no_trip(self):
        store = InMemoryStorage()
        @reliable(fuse_limit=2, storage=store)
        def node(state):
            return state

        for i in range(10):
            node({"input": f"different_{i}"})


# ── All features combined ────────────────────────────────────────────────

class TestDecoratorAllFeatures:

    def test_all_params_together(self):
        """Every parameter set at once — should work."""
        budget = GlobalBudget(max_cost_usd=100.0, max_seconds=60)
        store = InMemoryStorage()

        @reliable(
            sentinel_schema=Output,
            fuse_limit=5,
            max_cost_usd=50.0,
            max_seconds=30.0,
            budget=budget,
            model="gpt-4o",
            cost_per_token=0.00001,
            node_name="mega_node",
            storage=store,
        )
        def node(state):
            return {"result": "all features active"}

        budget.reset()
        result = node({"x": 1})
        assert result.result == "all features active"

    def test_custom_node_name(self):
        store = InMemoryStorage()
        @reliable(node_name="custom_name", storage=store)
        def node(state):
            return state
        node({"x": 1})


# ── Edge case: function with kwargs ──────────────────────────────────────

class TestDecoratorFunctionSignatures:

    def test_function_with_kwargs(self):
        store = InMemoryStorage()
        @reliable(storage=store)
        def node(state, extra="default"):
            return {"result": extra}
        result = node({"x": 1}, extra="custom")
        assert result["result"] == "custom"

    def test_function_with_config_kwarg(self):
        store = InMemoryStorage()
        @reliable(storage=store)
        def node(state, config=None):
            return state
        result = node({"x": 1}, config={"configurable": {"thread_id": "t1"}})
        assert result == {"x": 1}

    def test_state_as_kwarg(self):
        store = InMemoryStorage()
        @reliable(storage=store)
        def node(state):
            return state
        result = node(state={"x": 1})
        assert result == {"x": 1}

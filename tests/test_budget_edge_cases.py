"""
Edge case tests for BudgetFuse, TimeoutFuse, GlobalBudget.

Covers: zero, negative, NaN, Inf, exact boundaries, thread safety,
concurrent reset, float accumulation.
"""
import math
import time
import threading
import pytest

from airos.budget import BudgetFuse, TimeoutFuse, GlobalBudget
from airos.errors import BudgetExceededError, TimeoutExceededError


# ── BudgetFuse ───────────────────────────────────────────────────────────

class TestBudgetFuseEdgeCases:

    def test_zero_budget_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            BudgetFuse(max_cost_usd=0)

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            BudgetFuse(max_cost_usd=-1.0)

    def test_very_small_budget(self):
        fuse = BudgetFuse(max_cost_usd=1e-10)
        fuse.check(0.0)  # under budget

    def test_very_large_budget(self):
        fuse = BudgetFuse(max_cost_usd=1e12)
        fuse.check(1e11)  # still under

    def test_zero_cost_passes(self):
        fuse = BudgetFuse(max_cost_usd=1.0)
        fuse.check(0.0)

    def test_negative_cost_passes(self):
        fuse = BudgetFuse(max_cost_usd=1.0)
        fuse.check(-5.0)  # under budget

    def test_exact_budget_trips(self):
        fuse = BudgetFuse(max_cost_usd=1.0)
        with pytest.raises(BudgetExceededError) as exc:
            fuse.check(1.0)
        assert exc.value.spent == 1.0
        assert exc.value.limit == 1.0

    def test_slightly_under_passes(self):
        fuse = BudgetFuse(max_cost_usd=1.0)
        fuse.check(0.9999999999)

    def test_nan_cost_passes(self):
        """NaN >= 1.0 is False in Python."""
        fuse = BudgetFuse(max_cost_usd=1.0)
        fuse.check(float('nan'))

    def test_inf_cost_trips(self):
        fuse = BudgetFuse(max_cost_usd=1e12)
        with pytest.raises(BudgetExceededError):
            fuse.check(float('inf'))

    def test_error_message_contains_amounts(self):
        fuse = BudgetFuse(max_cost_usd=5.0)
        with pytest.raises(BudgetExceededError, match="10.0000"):
            fuse.check(10.0)


# ── TimeoutFuse ──────────────────────────────────────────────────────────

class TestTimeoutFuseEdgeCases:

    def test_zero_timeout_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            TimeoutFuse(max_seconds=0)

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            TimeoutFuse(max_seconds=-10.0)

    def test_very_small_timeout_trips(self):
        fuse = TimeoutFuse(max_seconds=1e-10)
        start = time.time() - 1.0
        with pytest.raises(TimeoutExceededError):
            fuse.check(start)

    def test_future_start_time_passes(self):
        """Elapsed is negative — under timeout."""
        fuse = TimeoutFuse(max_seconds=1.0)
        fuse.check(time.time() + 1000)

    def test_exact_timeout_trips(self):
        fuse = TimeoutFuse(max_seconds=1.0)
        start = time.time() - 1.5
        with pytest.raises(TimeoutExceededError) as exc:
            fuse.check(start)
        assert exc.value.elapsed >= 1.0
        assert exc.value.limit == 1.0

    def test_very_large_timeout_never_trips(self):
        fuse = TimeoutFuse(max_seconds=1e10)
        fuse.check(time.time())

    def test_error_attributes(self):
        fuse = TimeoutFuse(max_seconds=5.0)
        start = time.time() - 10.0
        with pytest.raises(TimeoutExceededError) as exc:
            fuse.check(start)
        assert exc.value.elapsed >= 10.0
        assert exc.value.limit == 5.0


# ── GlobalBudget ─────────────────────────────────────────────────────────

class TestGlobalBudgetEdgeCases:

    def test_zero_cost_raises(self):
        with pytest.raises(ValueError):
            GlobalBudget(max_cost_usd=0)

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError):
            GlobalBudget(max_cost_usd=-1.0)

    def test_zero_seconds_raises(self):
        with pytest.raises(ValueError):
            GlobalBudget(max_cost_usd=1.0, max_seconds=0)

    def test_negative_seconds_raises(self):
        with pytest.raises(ValueError):
            GlobalBudget(max_cost_usd=1.0, max_seconds=-5.0)

    def test_none_seconds_unlimited_time(self):
        budget = GlobalBudget(max_cost_usd=1.0, max_seconds=None)
        budget.check_time()  # no error

    def test_record_negative_cost(self):
        budget = GlobalBudget(max_cost_usd=10.0)
        budget.record_cost(5.0)
        budget.record_cost(-2.0)
        assert budget.total_spent == 3.0

    def test_record_zero_cost(self):
        budget = GlobalBudget(max_cost_usd=10.0)
        budget.record_cost(0.0)
        assert budget.total_spent == 0.0

    def test_record_nan_cost(self):
        budget = GlobalBudget(max_cost_usd=10.0)
        budget.record_cost(float('nan'))
        assert math.isnan(budget.total_spent)

    def test_record_inf_trips(self):
        budget = GlobalBudget(max_cost_usd=10.0)
        budget.record_cost(float('inf'))
        with pytest.raises(BudgetExceededError):
            budget.check_cost()

    def test_remaining_never_negative(self):
        budget = GlobalBudget(max_cost_usd=1.0)
        budget.record_cost(5.0)
        assert budget.remaining == 0.0

    def test_remaining_with_no_spend(self):
        budget = GlobalBudget(max_cost_usd=10.0)
        assert budget.remaining == 10.0

    def test_exact_limit_trips(self):
        budget = GlobalBudget(max_cost_usd=1.0)
        budget.record_cost(1.0)
        with pytest.raises(BudgetExceededError):
            budget.check_cost()

    def test_both_limits_exceeded(self):
        budget = GlobalBudget(max_cost_usd=0.001, max_seconds=0.001)
        budget.record_cost(1.0)
        time.sleep(0.01)
        with pytest.raises(BudgetExceededError):
            budget.check_cost()
        with pytest.raises(TimeoutExceededError):
            budget.check_time()

    def test_reset_clears_everything(self):
        budget = GlobalBudget(max_cost_usd=10.0, max_seconds=60)
        budget.record_cost(5.0)
        time.sleep(0.01)
        budget.reset()
        assert budget.total_spent == 0.0
        assert budget.elapsed_seconds < 0.1

    def test_high_contention_100_threads(self):
        budget = GlobalBudget(max_cost_usd=1e10)
        errors = []

        def record():
            try:
                for _ in range(100):
                    budget.record_cost(1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert budget.total_spent == 10000.0

    def test_reset_during_concurrent_recording(self):
        budget = GlobalBudget(max_cost_usd=1e10)
        stop = threading.Event()
        errors = []

        def record():
            while not stop.is_set():
                try:
                    budget.record_cost(0.001)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=record) for _ in range(10)]
        for t in threads:
            t.start()
        time.sleep(0.05)
        budget.reset()
        time.sleep(0.05)
        stop.set()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_float_accumulation_precision(self):
        budget = GlobalBudget(max_cost_usd=100.0)
        for _ in range(10000):
            budget.record_cost(0.001)
        assert abs(budget.total_spent - 10.0) < 0.01

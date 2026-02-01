"""
Concurrent stress tests.

Covers: shared GlobalBudget under heavy threading, multiple decorated
nodes running in parallel, storage under concurrent writes.
"""
import threading
import time
import pytest

from airos import reliable, GlobalBudget, BudgetExceededError
from airos.storage import InMemoryStorage


# ── Shared budget under thread contention ────────────────────────────────

class TestConcurrentBudget:

    def test_50_threads_shared_budget(self):
        """50 threads all decrementing the same budget."""
        budget = GlobalBudget(max_cost_usd=1e10)
        errors = []

        @reliable(budget=budget, model="gpt-4o")
        def node(state):
            return {"result": "ok"}

        def run_node(thread_id):
            try:
                for i in range(10):
                    node({"thread": thread_id, "iter": i})
            except Exception as e:
                errors.append(e)

        budget.reset()
        threads = [threading.Thread(target=run_node, args=(t,)) for t in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert budget.total_spent > 0

    def test_concurrent_budget_eventually_trips(self):
        """Many threads with tight budget — should eventually trip."""
        budget = GlobalBudget(max_cost_usd=0.001)
        trip_count = 0
        lock = threading.Lock()

        @reliable(budget=budget, model="gpt-4o")
        def node(state):
            return {"result": "x" * 1000}

        def run_node():
            nonlocal trip_count
            try:
                for _ in range(20):
                    node({"data": "test"})
            except BudgetExceededError:
                with lock:
                    trip_count += 1

        budget.reset()
        threads = [threading.Thread(target=run_node) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least some threads should have been tripped
        assert trip_count > 0


# ── Shared storage under thread contention ───────────────────────────────

class TestConcurrentStorage:

    def test_concurrent_writes_to_storage(self):
        """Multiple threads writing traces to same storage."""
        storage = InMemoryStorage()
        errors = []

        @reliable(storage=storage)
        def node(state):
            return state

        def run_node(thread_id):
            try:
                for i in range(20):
                    node({"thread": thread_id, "iter": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_node, args=(t,)) for t in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_reads_and_writes(self):
        """Reads and writes happening simultaneously."""
        storage = InMemoryStorage()
        errors = []

        @reliable(storage=storage)
        def node(state):
            return state

        def writer(thread_id):
            try:
                for i in range(20):
                    node({"thread": thread_id, "iter": i})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    storage.get_run_history("local_dev_run")
                    storage.get_run_cost("local_dev_run")
            except Exception as e:
                errors.append(e)

        threads = []
        for t in range(10):
            threads.append(threading.Thread(target=writer, args=(t,)))
        for _ in range(5):
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ── Rapid reset/record cycles ───────────────────────────────────────────

class TestConcurrentResetCycles:

    def test_rapid_reset_record_cycle(self):
        """Alternating reset and record in tight loop."""
        budget = GlobalBudget(max_cost_usd=1e10)
        errors = []

        def reset_loop():
            try:
                for _ in range(100):
                    budget.reset()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def record_loop():
            try:
                for _ in range(100):
                    budget.record_cost(0.01)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=reset_loop)
        t2 = threading.Thread(target=record_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0
        # No crash — that's the test

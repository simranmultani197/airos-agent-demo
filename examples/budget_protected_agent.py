"""
Example 3: Budget-Protected Agent — Dollar + time circuit breakers.

Shows max_cost_usd and max_seconds stopping runaway nodes.
No real LLM needed — uses simulated costs.

Run:
    python examples/budget_protected_agent.py
"""
import time
from pydantic import BaseModel
from airos import reliable, BudgetExceededError, TimeoutExceededError


class AnalysisOutput(BaseModel):
    result: str
    score: float


# ── Per-node dollar limit ────────────────────────────────────────────────

@reliable(
    sentinel_schema=AnalysisOutput,
    max_cost_usd=0.10,
    model="gpt-4o",
)
def cheap_node(state: dict) -> dict:
    """This node has a $0.10 budget. Small inputs stay under budget."""
    return {"result": state.get("data", "analyzed"), "score": 0.95}


# ── Per-node time limit ─────────────────────────────────────────────────

@reliable(
    sentinel_schema=AnalysisOutput,
    max_seconds=2,
)
def slow_node(state: dict) -> dict:
    """This node has a 2-second timeout. Sleeps to demonstrate the trip."""
    delay = state.get("delay", 0)
    if delay > 0:
        time.sleep(delay)
    return {"result": "done", "score": 1.0}


def main():
    print("=" * 50)
    print("AirOS Demo: Budget-Protected Agent")
    print("=" * 50)

    # Test 1: Under budget — passes fine
    print("\n[Test 1] Node under budget ($0.10 limit):")
    result = cheap_node({"data": "small input"})
    print(f"  Result: {result}")

    # Test 2: Time limit — fast call passes
    print("\n[Test 2] Fast node (2s timeout):")
    result = slow_node({"delay": 0})
    print(f"  Result: {result}")

    # Test 3: Time limit — slow call trips
    print("\n[Test 3] Slow node (2s timeout, 3s delay):")
    try:
        result = slow_node({"delay": 3})
        print(f"  Result: {result}")
    except TimeoutExceededError as e:
        print(f"  Tripped! {e}")
        print(f"  Elapsed: {e.elapsed:.1f}s, Limit: {e.limit:.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()

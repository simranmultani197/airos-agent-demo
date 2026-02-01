"""
Example 4: Multi-Node Global Budget — Shared budget across an agent graph.

Shows GlobalBudget tracking cost across multiple nodes and tripping
when the combined cost exceeds the limit.

No real LLM needed — uses simulated function calls.

Run:
    python examples/multi_node_budget.py
"""
from pydantic import BaseModel
from airos import reliable, GlobalBudget, BudgetExceededError, TimeoutExceededError


class StepOutput(BaseModel):
    step: str
    data: str


# Shared budget: $0.50 total, 30 second time limit
budget = GlobalBudget(max_cost_usd=0.50, max_seconds=30)


@reliable(sentinel_schema=StepOutput, budget=budget, model="gpt-4o")
def step_one(state: dict) -> dict:
    return {"step": "research", "data": f"researched {state['topic']}"}


@reliable(sentinel_schema=StepOutput, budget=budget, model="gpt-4o")
def step_two(state: dict) -> dict:
    return {"step": "analyze", "data": f"analyzed {state['data']}"}


@reliable(sentinel_schema=StepOutput, budget=budget, model="gpt-4o")
def step_three(state: dict) -> dict:
    return {"step": "format", "data": f"formatted {state['data']}"}


def main():
    print("=" * 50)
    print("AirOS Demo: Multi-Node Global Budget")
    print("=" * 50)

    # Run 1: Normal pipeline — all 3 steps under budget
    print("\n[Run 1] Normal pipeline ($0.50 budget):")
    budget.reset()

    r1 = step_one({"topic": "AI safety"})
    r1 = r1.model_dump() if hasattr(r1, "model_dump") else r1
    print(f"  Step 1: {r1['step']} — spent ${budget.total_spent:.6f}")

    r2 = step_two({"data": r1["data"]})
    r2 = r2.model_dump() if hasattr(r2, "model_dump") else r2
    print(f"  Step 2: {r2['step']} — spent ${budget.total_spent:.6f}")

    r3 = step_three({"data": r2["data"]})
    r3 = r3.model_dump() if hasattr(r3, "model_dump") else r3
    print(f"  Step 3: {r3['step']} — spent ${budget.total_spent:.6f}")

    print(f"  Total: ${budget.total_spent:.6f} | Remaining: ${budget.remaining:.6f}")

    # Run 2: Tight budget — should trip
    print("\n[Run 2] Tight budget ($0.0001):")
    tight_budget = GlobalBudget(max_cost_usd=0.0001, max_seconds=30)

    @reliable(sentinel_schema=StepOutput, budget=tight_budget, model="gpt-4o")
    def expensive_step(state: dict) -> dict:
        # Return a large output to generate cost
        return {"step": "expensive", "data": "x" * 5000}

    try:
        r = expensive_step({"topic": "test"})
        print(f"  Step 1 done — spent ${tight_budget.total_spent:.6f}")

        # Second call should trip — budget already used
        r = expensive_step({"topic": "test2"})
        print(f"  Step 2 done — spent ${tight_budget.total_spent:.6f}")
    except BudgetExceededError as e:
        print(f"  Budget tripped! Spent ${e.spent:.6f} of ${e.limit:.6f}")

    # Run 3: Check reset works
    print("\n[Run 3] Budget reset and reuse:")
    budget.reset()
    print(f"  After reset: spent ${budget.total_spent:.6f}, remaining ${budget.remaining:.6f}")

    r = step_one({"topic": "reset test"})
    r = r.model_dump() if hasattr(r, "model_dump") else r
    print(f"  After one step: spent ${budget.total_spent:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

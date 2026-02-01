"""
Example 5: Cost Tracking — Model-aware pricing and CostCalculator.

Demonstrates the pricing system:
- Built-in pricing table for 25+ models
- Custom cost_per_token override
- CostCalculator for direct cost estimation

No real LLM needed.

Run:
    python examples/cost_tracking_agent.py
"""
from pydantic import BaseModel
from airos import (
    reliable,
    CostCalculator,
    get_model_pricing,
    MODEL_PRICING,
    GlobalBudget,
)


class Output(BaseModel):
    result: str


def main():
    print("=" * 50)
    print("AirOS Demo: Cost Tracking & Pricing")
    print("=" * 50)

    # ── Part 1: Explore the pricing table ────────────────────────────────
    print("\n[Part 1] Built-in model pricing:\n")
    models_to_show = [
        "gpt-4o", "gpt-4o-mini", "gpt-4",
        "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus",
        "llama-3.3-70b-versatile", "gemini-2.0-flash",
    ]
    print(f"  {'Model':<30} {'Input $/1M':<15} {'Output $/1M':<15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    for name in models_to_show:
        p = get_model_pricing(name)
        inp = p.input_per_token * 1_000_000
        out = p.output_per_token * 1_000_000
        print(f"  {name:<30} ${inp:<14.2f} ${out:<14.2f}")

    print(f"\n  Total models in table: {len(MODEL_PRICING)}")

    # ── Part 2: CostCalculator usage ─────────────────────────────────────
    print("\n[Part 2] CostCalculator:\n")

    # Model-based
    calc = CostCalculator(model="gpt-4o")
    cost = calc.calculate(input_tokens=1000, output_tokens=500)
    print(f"  GPT-4o: 1000 input + 500 output = ${cost:.6f}")

    calc_mini = CostCalculator(model="gpt-4o-mini")
    cost_mini = calc_mini.calculate(input_tokens=1000, output_tokens=500)
    print(f"  GPT-4o-mini: same tokens = ${cost_mini:.6f}")
    print(f"  Savings with mini: {(1 - cost_mini/cost)*100:.0f}%")

    # Custom rate
    calc_custom = CostCalculator(cost_per_token=0.00005)
    cost_custom = calc_custom.calculate(input_tokens=1000, output_tokens=500)
    print(f"  Custom ($50/1M): same tokens = ${cost_custom:.6f}")

    # Estimate from objects
    print("\n  Estimate from objects:")
    tokens, est_cost = calc.estimate_from_objects(
        {"prompt": "What is the meaning of life?"},
        {"answer": "42", "confidence": 0.99}
    )
    print(f"  Estimated tokens: {tokens}, cost: ${est_cost:.6f}")

    # ── Part 3: Model-aware @reliable ────────────────────────────────────
    print("\n[Part 3] @reliable with model pricing:\n")

    budget = GlobalBudget(max_cost_usd=1.0)

    @reliable(sentinel_schema=Output, budget=budget, model="gpt-4o")
    def gpt4o_node(state: dict) -> dict:
        return {"result": "expensive output " * 50}

    @reliable(sentinel_schema=Output, budget=budget, model="gpt-4o-mini")
    def mini_node(state: dict) -> dict:
        return {"result": "cheap output " * 50}

    budget.reset()
    gpt4o_node({"prompt": "test"})
    cost_after_gpt4o = budget.total_spent

    mini_node({"prompt": "test"})
    cost_after_mini = budget.total_spent - cost_after_gpt4o

    print(f"  GPT-4o node cost:      ${cost_after_gpt4o:.6f}")
    print(f"  GPT-4o-mini node cost: ${cost_after_mini:.6f}")
    print(f"  Total pipeline cost:   ${budget.total_spent:.6f}")

    # ── Part 4: Custom cost_per_token override ───────────────────────────
    print("\n[Part 4] Custom cost_per_token override:\n")

    budget2 = GlobalBudget(max_cost_usd=1.0)

    @reliable(sentinel_schema=Output, budget=budget2, cost_per_token=0.0001)
    def custom_priced_node(state: dict) -> dict:
        return {"result": "output with custom pricing"}

    budget2.reset()
    custom_priced_node({"input": "test"})
    print(f"  Custom-priced node cost: ${budget2.total_spent:.6f}")
    print(f"  (using $100/1M token rate)")

    print("\nDone.")


if __name__ == "__main__":
    main()

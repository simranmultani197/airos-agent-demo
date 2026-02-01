"""
Example 8: Full Pipeline — All AirOS features in one pipeline.

Chains 3 nodes with every feature:
  Fuse + Sentinel + Medic + Budget + Pricing + Storage

No real LLM needed — uses a mock LLM for self-healing demo.

Run:
    python examples/full_pipeline_agent.py
"""
import json
from pydantic import BaseModel
from airos import (
    reliable, GlobalBudget,
    BudgetExceededError, TimeoutExceededError,
)
from airos.storage import InMemoryStorage
from airos.fuse import LoopError


# ── Schemas ──────────────────────────────────────────────────────────────

class ResearchOutput(BaseModel):
    topic: str
    findings: list[str]

class SummaryOutput(BaseModel):
    summary: str
    word_count: int

class FinalOutput(BaseModel):
    title: str
    body: str


# ── Mock LLM ─────────────────────────────────────────────────────────────

def mock_llm(prompt: str) -> str:
    """Simulates an LLM that returns valid JSON for repair."""
    if "ResearchOutput" in prompt or "findings" in prompt:
        return json.dumps({
            "topic": "AI safety",
            "findings": ["Finding 1", "Finding 2", "Finding 3"]
        })
    elif "SummaryOutput" in prompt or "summary" in prompt:
        return json.dumps({
            "summary": "AI safety is important for responsible development.",
            "word_count": 7
        })
    elif "FinalOutput" in prompt or "title" in prompt:
        return json.dumps({
            "title": "AI Safety Report",
            "body": "AI safety is important for responsible development."
        })
    return json.dumps({"result": "fallback"})


# ── Pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("AirOS Demo: Full Pipeline (All Features)")
    print("=" * 50)

    storage = InMemoryStorage()
    budget = GlobalBudget(max_cost_usd=1.0, max_seconds=30)

    @reliable(
        sentinel_schema=ResearchOutput,
        llm_callable=mock_llm,
        fuse_limit=3,
        budget=budget,
        model="gpt-4o",
        max_cost_usd=0.50,
        max_seconds=10,
        node_name="research",
        storage=storage,
    )
    def research_node(state):
        return {"topic": state["topic"], "findings": ["F1", "F2", "F3"]}

    @reliable(
        sentinel_schema=SummaryOutput,
        llm_callable=mock_llm,
        fuse_limit=3,
        budget=budget,
        model="gpt-4o-mini",
        node_name="summarize",
        storage=storage,
    )
    def summarize_node(state):
        text = " ".join(state.get("findings", []))
        return {"summary": f"Summary of: {text}", "word_count": len(text.split())}

    @reliable(
        sentinel_schema=FinalOutput,
        llm_callable=mock_llm,
        fuse_limit=3,
        budget=budget,
        model="claude-3-5-haiku",
        node_name="format",
        storage=storage,
    )
    def format_node(state):
        return {
            "title": f"Report: {state.get('topic', 'Unknown')}",
            "body": state.get("summary", "No summary"),
        }

    # ── Run pipeline ─────────────────────────────────────────────────────
    print("\n[Pipeline] Running research → summarize → format\n")
    budget.reset()

    # Step 1
    print("[1/3] Research...")
    r1 = research_node({"topic": "AI safety"})
    r1_dict = r1.model_dump() if hasattr(r1, "model_dump") else r1
    print(f"  Findings: {len(r1_dict['findings'])}")
    print(f"  $ spent: ${budget.total_spent:.6f}")

    # Step 2
    print("\n[2/3] Summarize...")
    r2 = summarize_node({**r1_dict, "topic": "AI safety"})
    r2_dict = r2.model_dump() if hasattr(r2, "model_dump") else r2
    print(f"  Words: {r2_dict['word_count']}")
    print(f"  $ spent: ${budget.total_spent:.6f}")

    # Step 3
    print("\n[3/3] Format...")
    r3 = format_node({**r2_dict, "topic": "AI safety"})
    r3_dict = r3.model_dump() if hasattr(r3, "model_dump") else r3
    print(f"  Title: {r3_dict['title']}")
    print(f"  $ spent: ${budget.total_spent:.6f}")

    # ── Results ──────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Pipeline Results:")
    print(f"  Budget: ${budget.total_spent:.6f} / ${budget.max_cost_usd:.2f}")
    print(f"  Remaining: ${budget.remaining:.6f}")
    print(f"  Time: {budget.elapsed_seconds:.2f}s")

    # ── Storage traces ───────────────────────────────────────────────────
    print(f"\nStorage Traces:")
    history = storage.get_run_history("local_dev_run")
    for h in history:
        print(f"  [{h['status']}] {h['node_id']}")
    print(f"  Total traces: {len(history)}")
    print(f"  Total cost tracked: ${storage.get_run_cost('local_dev_run'):.6f}")

    # ── Fuse demo ────────────────────────────────────────────────────────
    print(f"\nFuse Demo (loop detection):")
    try:
        for i in range(5):
            research_node({"topic": "AI safety"})
            print(f"  Call {i+1}: OK")
    except LoopError:
        print(f"  Fuse tripped! Loop detected.")

    # ── Budget trip demo ─────────────────────────────────────────────────
    print(f"\nBudget Trip Demo:")
    tight_budget = GlobalBudget(max_cost_usd=0.0001, max_seconds=30)
    tight_storage = InMemoryStorage()

    @reliable(
        sentinel_schema=FinalOutput,
        budget=tight_budget,
        model="gpt-4o",
        storage=tight_storage,
    )
    def expensive_node(state):
        return {"title": "big", "body": "x" * 5000}

    tight_budget.reset()
    try:
        expensive_node({"input": "test1"})
        print(f"  Call 1: ${tight_budget.total_spent:.6f}")
        expensive_node({"input": "test2"})
        print(f"  Call 2: ${tight_budget.total_spent:.6f}")
    except BudgetExceededError as e:
        print(f"  Budget tripped! ${e.spent:.6f} >= ${e.limit:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

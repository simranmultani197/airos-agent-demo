"""
Simple pipeline that chains research → summarize → format.

No framework dependency — just plain function calls with AirOS protection.
"""
from .nodes import research_node, summarize_node, format_node
from .config import BUDGET


def _to_dict(result):
    """Convert Pydantic model to dict if needed."""
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


def run_pipeline(topic: str) -> dict:
    """
    Run the full agent pipeline for a given topic.

    Each node is protected by:
    - Fuse (loop detection)
    - Sentinel (schema validation)
    - Medic (LLM auto-repair)
    - Budget (shared cost + time limit)
    - Pricing (model-aware cost tracking)

    Returns the final formatted output.
    """
    # Reset budget for a fresh run
    BUDGET.reset()

    print(f"[pipeline] Starting research on: {topic}")
    print(f"[pipeline] Budget: ${BUDGET.remaining:.2f} | Time limit: 60s\n")

    # Step 1: Research
    print("[1/3] Researching...")
    research = _to_dict(research_node({"topic": topic}))
    print(f"  Found {len(research.get('findings', []))} findings")
    print(f"  $ Spent so far: ${BUDGET.total_spent:.6f}\n")

    # Step 2: Summarize
    print("[2/3] Summarizing...")
    summary = _to_dict(summarize_node({**research, "topic": topic}))
    print(f"  Generated {len(summary.get('key_points', []))} key points")
    print(f"  $ Spent so far: ${BUDGET.total_spent:.6f}\n")

    # Step 3: Format
    print("[3/3] Formatting...")
    result = _to_dict(format_node({**summary, "topic": topic}))
    print(f"  Output: {result.get('word_count', 0)} words")
    print(f"  $ Total cost: ${BUDGET.total_spent:.6f}")
    print(f"  Total time: {BUDGET.elapsed_seconds:.1f}s\n")

    return result

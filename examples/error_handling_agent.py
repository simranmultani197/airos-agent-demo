"""
Example 6: Error Handling — ErrorClassifier + recovery strategies.

Shows how AirOS classifies errors and picks recovery strategies.
No LLM required.

Run:
    python examples/error_handling_agent.py
"""
import json
from airos.errors import (
    ErrorClassifier, ErrorCategory, ErrorSeverity,
    AirOSError, BudgetExceededError, TimeoutExceededError,
)
from airos.fuse import LoopError


def main():
    print("=" * 50)
    print("AirOS Demo: Error Classification & Handling")
    print("=" * 50)

    # ── Part 1: Classify different error types ───────────────────────────
    print("\n[Part 1] Error classification:\n")

    errors = [
        json.JSONDecodeError("Expecting value", "", 0),
        ValueError("Sentinel Alert: Output validation failed: missing field 'name'"),
        TypeError("cannot convert string to int"),
        Exception("429 too many requests — rate limit exceeded"),
        Exception("401 unauthorized: invalid api key"),
        LoopError("Fuse Tripped: Loop detected. State repeated 3 times."),
        MemoryError("out of memory"),
        Exception("context_length_exceeded: maximum token limit reached"),
        Exception("model is overloaded, try again later"),
        Exception("totally unknown error xyz123"),
    ]

    print(f"  {'Error Type':<35} {'Category':<25} {'Severity':<10} {'Recoverable':<12} {'Strategy'}")
    print(f"  {'-'*35} {'-'*25} {'-'*10} {'-'*12} {'-'*20}")

    for err in errors:
        classified = ErrorClassifier.classify(err)
        print(
            f"  {type(err).__name__:<35} "
            f"{classified.category.value:<25} "
            f"{classified.severity.value:<10} "
            f"{'Yes' if classified.recoverable else 'No':<12} "
            f"{classified.suggested_strategy}"
        )

    # ── Part 2: Recovery hints ───────────────────────────────────────────
    print("\n[Part 2] Recovery hints for LLM repair:\n")

    hint_errors = [
        json.JSONDecodeError("bad json", "", 0),
        ValueError("Sentinel Alert: validation failed"),
        Exception("context_length_exceeded"),
    ]

    for err in hint_errors:
        classified = ErrorClassifier.classify(err)
        hint = ErrorClassifier.get_recovery_prompt_hint(classified)
        print(f"  [{classified.category.value}]")
        print(f"    Hint: {hint}\n")

    # ── Part 3: Structured error metadata ────────────────────────────────
    print("[Part 3] ClassifiedError as dict:\n")

    err = json.JSONDecodeError("Expecting value", '{"bad": json}', 8)
    classified = ErrorClassifier.classify(err, context={"node_id": "extract_node", "attempt": 1})
    d = classified.to_dict()

    for key, value in d.items():
        print(f"  {key}: {value}")

    # ── Part 4: Non-recoverable error set ────────────────────────────────
    print(f"\n[Part 4] Non-recoverable error categories:\n")
    for cat in ErrorClassifier.NON_RECOVERABLE:
        severity = ErrorClassifier.SEVERITY_MAP[cat]
        strategy = ErrorClassifier.STRATEGY_MAP[cat]
        print(f"  {cat.value:<20} severity={severity.value:<10} strategy={strategy}")

    # ── Part 5: All severity levels ──────────────────────────────────────
    print(f"\n[Part 5] All error categories by severity:\n")
    by_severity = {}
    for cat in ErrorCategory:
        sev = ErrorClassifier.SEVERITY_MAP.get(cat, ErrorSeverity.MEDIUM)
        by_severity.setdefault(sev.value, []).append(cat.value)

    for sev in ["low", "medium", "high", "critical"]:
        cats = by_severity.get(sev, [])
        print(f"  {sev.upper()}: {', '.join(cats)}")

    print("\nDone.")


if __name__ == "__main__":
    main()

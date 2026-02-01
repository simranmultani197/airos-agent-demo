"""
Example 1: Basic Agent — Schema validation + loop detection only.

No LLM required. Shows Fuse + Sentinel working with pure functions.

Run:
    python examples/basic_agent.py
"""
from pydantic import BaseModel
from airos import reliable, LoopError


class WeatherOutput(BaseModel):
    city: str
    temperature: float
    unit: str


@reliable(sentinel_schema=WeatherOutput, fuse_limit=3)
def weather_node(state: dict) -> dict:
    """Simulates a weather lookup. Always returns valid data."""
    return {
        "city": state["city"],
        "temperature": 72.5,
        "unit": "fahrenheit",
    }


@reliable(sentinel_schema=WeatherOutput, fuse_limit=3)
def bad_weather_node(state: dict) -> dict:
    """Returns invalid data — Sentinel will catch this."""
    return {"city": state["city"]}  # missing temperature and unit


def main():
    print("=" * 50)
    print("AirOS Demo: Basic Agent (Fuse + Sentinel)")
    print("=" * 50)

    # Test 1: Valid output passes through
    print("\n[Test 1] Valid output:")
    result = weather_node({"city": "San Francisco"})
    print(f"  Result: {result}")

    # Test 2: Invalid output caught by Sentinel (no Medic to repair)
    print("\n[Test 2] Invalid output (no Medic, so it raises):")
    try:
        result = bad_weather_node({"city": "New York"})
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Caught: {type(e).__name__}: {e}")

    # Test 3: Loop detection
    print("\n[Test 3] Loop detection (same input 3 times):")
    try:
        for i in range(5):
            result = weather_node({"city": "Chicago"})
            print(f"  Call {i+1}: OK")
    except LoopError as e:
        print(f"  Caught: LoopError — fuse tripped!")

    print("\nDone.")


if __name__ == "__main__":
    main()

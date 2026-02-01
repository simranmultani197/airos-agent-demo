"""
Example 2: Self-Healing Agent — Medic auto-repairs bad outputs.

Requires: GROQ_API_KEY environment variable.

Run:
    export GROQ_API_KEY=your_key_here
    python examples/self_healing_agent.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pydantic import BaseModel
from airos import reliable


class ExtractedData(BaseModel):
    name: str
    age: int
    email: str


def get_llm():
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

    def call(prompt: str) -> str:
        return client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        ).choices[0].message.content

    return call


llm = get_llm()


@reliable(sentinel_schema=ExtractedData, llm_callable=llm, fuse_limit=5)
def extract_node(state: dict) -> dict:
    """
    Extracts structured data from text.
    Deliberately returns malformed JSON to trigger Medic repair.
    """
    text = state["text"]
    # Simulate a bad LLM response — missing fields, wrong format
    return {
        "name": "John",
        # age and email are missing — Sentinel will reject,
        # Medic will call the LLM to produce a valid output
    }


@reliable(sentinel_schema=ExtractedData, llm_callable=llm, fuse_limit=5)
def good_extract_node(state: dict) -> dict:
    """Uses the LLM to actually extract data from text."""
    prompt = (
        f"Extract the person's name, age, and email from this text. "
        f"Return JSON with keys: name (str), age (int), email (str).\n\n"
        f"Text: {state['text']}"
    )
    raw = llm(prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}  # Medic will fix this


def main():
    print("=" * 50)
    print("AirOS Demo: Self-Healing Agent (Medic)")
    print("=" * 50)

    # Test 1: Medic repairs missing fields
    print("\n[Test 1] Bad output → Medic repairs:")
    try:
        result = extract_node({"text": "John Doe, 30 years old, john@example.com"})
        print(f"  Repaired result: {result}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

    # Test 2: LLM extraction with auto-repair fallback
    print("\n[Test 2] LLM extraction with Medic fallback:")
    result = good_extract_node({
        "text": "My name is Alice Smith, I'm 28 years old. Reach me at alice@tech.io"
    })
    print(f"  Result: {result}")

    print("\nDone.")


if __name__ == "__main__":
    main()

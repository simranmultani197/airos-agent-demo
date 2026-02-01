"""
Configuration for the demo agent.

Set your API key as an environment variable:
    export GROQ_API_KEY=your_key_here

You can swap to any provider (OpenAI, Anthropic) by changing this file.
"""
import os

MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def get_llm():
    """Return an LLM callable using Groq (free tier friendly)."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "groq package required. Install with: pip install groq"
        )

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Get a free key at https://console.groq.com"
        )

    client = Groq(api_key=GROQ_API_KEY)

    def call_llm(prompt: str) -> str:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
        )
        return response.choices[0].message.content

    return call_llm


# Shared budget for multi-node examples
from airos import GlobalBudget

BUDGET = GlobalBudget(max_cost_usd=1.0, max_seconds=60)

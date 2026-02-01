"""
Agent nodes — each decorated with @reliable to demonstrate AirOS features.
"""
import json
from pydantic import BaseModel
from airos import reliable
from .config import BUDGET, MODEL


# ── Schemas ──────────────────────────────────────────────────────────────

class ResearchOutput(BaseModel):
    topic: str
    findings: list[str]
    confidence: float


class SummaryOutput(BaseModel):
    summary: str
    key_points: list[str]


class FormattedOutput(BaseModel):
    title: str
    body: str
    word_count: int


# ── Lazy LLM loader ─────────────────────────────────────────────────────

_llm_instance = None


def _get_llm():
    """Lazy-load LLM so importing this module doesn't require GROQ_API_KEY."""
    global _llm_instance
    if _llm_instance is None:
        from .config import get_llm
        _llm_instance = get_llm()
    return _llm_instance


def _llm_proxy(prompt: str) -> str:
    """Proxy that forwards to the lazy-loaded LLM."""
    return _get_llm()(prompt)


# ── Nodes ────────────────────────────────────────────────────────────────

@reliable(
    sentinel_schema=ResearchOutput,
    llm_callable=_llm_proxy,
    fuse_limit=3,
    budget=BUDGET,
    model=MODEL,
)
def research_node(state: dict) -> dict:
    """Research a topic using an LLM. Returns structured findings."""
    llm = _get_llm()
    prompt = (
        f"Research the topic: {state['topic']}. "
        f"Return a JSON object with keys: topic (string), "
        f"findings (list of 3 strings), confidence (float 0-1)."
    )
    raw = llm(prompt)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Medic will auto-repair this via the LLM
        return {"raw_response": raw}


@reliable(
    sentinel_schema=SummaryOutput,
    llm_callable=_llm_proxy,
    fuse_limit=3,
    budget=BUDGET,
    model=MODEL,
)
def summarize_node(state: dict) -> dict:
    """Summarize research findings into key points."""
    llm = _get_llm()
    findings = state.get("findings", [])
    prompt = (
        f"Summarize these findings into a short summary and 3 key points. "
        f"Findings: {findings}. "
        f"Return JSON with keys: summary (string), key_points (list of 3 strings)."
    )
    raw = llm(prompt)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_response": raw}


@reliable(
    sentinel_schema=FormattedOutput,
    llm_callable=_llm_proxy,
    fuse_limit=3,
    budget=BUDGET,
    model=MODEL,
)
def format_node(state: dict) -> dict:
    """Format a summary into a presentable output with word count."""
    summary = state.get("summary", "")
    key_points = state.get("key_points", [])
    body = f"{summary}\n\n" + "\n".join(f"- {p}" for p in key_points)
    return {
        "title": f"Report: {state.get('topic', 'Unknown')}",
        "body": body,
        "word_count": len(body.split()),
    }

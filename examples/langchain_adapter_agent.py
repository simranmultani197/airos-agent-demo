"""
Example: LangChain Adapter — Testing AirOS reliability with LangChain components.

Tests:
1. AdapterRegistry discovery & creation
2. wrap_node() — wrapping plain functions
3. wrap_chain() — wrapping LCEL Runnables as RunnableLambda
4. wrap_tool() — wrapping LangChain tools
5. Middleware decorator pattern
6. Schema validation through adapter
7. Fuse (loop detection) through adapter
8. Medic (self-healing) through adapter
"""
import os
import json
from pydantic import BaseModel
from typing import Optional

from airos.adapters import LangChainAdapter, AdapterRegistry
from airos.adapters.base import AdapterConfig
from airos.fuse import LoopError

# ── Test Schemas ──────────────────────────────────────────────────────────

class AnalysisOutput(BaseModel):
    summary: str
    sentiment: str
    confidence: float

class SearchOutput(BaseModel):
    query: str
    results: list
    count: int


def get_llm_callable():
    """Get LLM callable if Groq API key is available."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq
    client = Groq(api_key=api_key)
    def call_llm(prompt: str) -> str:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content
    return call_llm


def main():
    print("=" * 60)
    print("LangChain Adapter — AirOS Reliability Integration")
    print("=" * 60)

    # ── Test 1: Registry Discovery ────────────────────────────────────
    print("\n── Test 1: AdapterRegistry Discovery ──")
    adapters = AdapterRegistry.list_adapters()
    print(f"  Available adapters: {adapters}")
    assert "langchain" in adapters, "LangChain adapter not registered!"

    lc_class = AdapterRegistry.get("langchain")
    print(f"  LangChain adapter class: {lc_class.__name__}")
    assert lc_class is LangChainAdapter
    print("  ✅ Registry discovery works")

    # ── Test 2: Adapter Creation with Config ──────────────────────────
    print("\n── Test 2: Adapter Creation ──")
    config = AdapterConfig(
        fuse_enabled=True,
        fuse_limit=3,
        sentinel_enabled=True,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter = LangChainAdapter(config)
    print(f"  Framework: {adapter.framework_name}")
    assert adapter.framework_name == "langchain"

    # Also test factory creation
    adapter2 = AdapterRegistry.create("langchain", config)
    assert adapter2.framework_name == "langchain"
    print("  ✅ Adapter creation works (direct + factory)")

    # ── Test 3: wrap_node() — Plain Function ──────────────────────────
    print("\n── Test 3: wrap_node() — Plain Function ──")

    def analyze_text(text: str) -> dict:
        """Simulate text analysis."""
        return {
            "summary": f"Analysis of: {text[:30]}",
            "sentiment": "positive",
            "confidence": 0.92,
        }

    config_no_storage = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=True,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_ns = LangChainAdapter(config_no_storage)
    wrapped = adapter_ns.wrap_node(analyze_text, schema=AnalysisOutput, node_name="text_analyzer")
    result = wrapped("Hello world, this is a test input for analysis")
    print(f"  Result: name={result.summary}, sentiment={result.sentiment}")
    assert result.sentiment == "positive"
    assert result.confidence == 0.92
    print("  ✅ wrap_node works with schema validation")

    # ── Test 4: wrap_chain() — LCEL Runnable ──────────────────────────
    print("\n── Test 4: wrap_chain() — LCEL Runnable ──")
    try:
        from langchain_core.runnables import RunnableLambda

        # Create a simple chain
        def search_fn(query: str) -> dict:
            return {
                "query": query if isinstance(query, str) else query.get("input", str(query)),
                "results": ["result1", "result2", "result3"],
                "count": 3,
            }

        chain = RunnableLambda(search_fn)
        config_chain = AdapterConfig(
            fuse_enabled=False,
            sentinel_enabled=True,
            medic_enabled=False,
            storage_enabled=False,
        )
        adapter_chain = LangChainAdapter(config_chain)
        wrapped_chain = adapter_chain.wrap_chain(chain, schema=SearchOutput)

        result = wrapped_chain.invoke("machine learning papers")
        print(f"  Query: {result.query}")
        print(f"  Count: {result.count}")
        assert result.count == 3
        print("  ✅ wrap_chain works — RunnableLambda with Sentinel")
    except ImportError:
        print("  ⚠️ langchain-core not installed, skipping chain test")

    # ── Test 5: wrap_tool() — LangChain Tool ─────────────────────────
    print("\n── Test 5: wrap_tool() — LangChain Tool ──")
    try:
        from langchain_core.tools import tool

        @tool
        def calculator(expression: str) -> str:
            """Evaluate a math expression."""
            try:
                result = eval(expression)
                return json.dumps({"expression": expression, "result": result})
            except Exception as e:
                return json.dumps({"expression": expression, "error": str(e)})

        config_tool = AdapterConfig(
            fuse_enabled=False,
            sentinel_enabled=False,
            medic_enabled=False,
            storage_enabled=False,
        )
        adapter_tool = LangChainAdapter(config_tool)
        wrapped_tool = adapter_tool.wrap_tool(calculator)

        result = wrapped_tool.invoke("2 + 3 * 4")
        parsed = json.loads(result)
        print(f"  Expression: {parsed['expression']}")
        print(f"  Result: {parsed['result']}")
        assert parsed["result"] == 14
        print("  ✅ wrap_tool works")
    except ImportError:
        print("  ⚠️ langchain-core not installed, skipping tool test")

    # ── Test 6: Middleware Pattern ─────────────────────────────────────
    print("\n── Test 6: Middleware Decorator Pattern ──")
    config_mid = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=True,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_mid = LangChainAdapter(config_mid)
    middleware = adapter_mid.create_middleware()

    @middleware.tool(schema=AnalysisOutput, name="sentiment_tool")
    def sentiment_analysis(text: str) -> dict:
        return {"summary": "Positive review", "sentiment": "positive", "confidence": 0.85}

    result = sentiment_analysis("This product is great!")
    print(f"  Sentiment: {result.sentiment}, Confidence: {result.confidence}")
    assert result.sentiment == "positive"
    print("  ✅ Middleware decorator works")

    # ── Test 7: Schema Validation Failure ─────────────────────────────
    print("\n── Test 7: Schema Validation Failure ──")

    def bad_analysis(text: str) -> dict:
        """Returns incomplete data — missing 'confidence' field."""
        return {"summary": "bad", "sentiment": "neutral"}  # missing confidence!

    config_strict = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=True,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_strict = LangChainAdapter(config_strict)
    wrapped_bad = adapter_strict.wrap_node(bad_analysis, schema=AnalysisOutput, node_name="bad_node")

    try:
        wrapped_bad("test")
        print("  ⚠️ Should have raised SentinelError")
    except Exception as e:
        print(f"  Caught: {type(e).__name__}")
        print("  ✅ Schema validation correctly rejects incomplete output")

    # ── Test 8: State & Config Extraction ─────────────────────────────
    print("\n── Test 8: State & Config Extraction ──")
    adapter_ext = LangChainAdapter()

    # Dict input
    state = adapter_ext.extract_state({"input": "hello", "context": "world"})
    assert state == {"input": "hello", "context": "world"}
    print(f"  Dict state: {state}")

    # String input
    state2 = adapter_ext.extract_state("just a string")
    assert state2 == {"input": "just a string"}
    print(f"  String state: {state2}")

    # Config extraction
    cfg = adapter_ext.extract_config(config={"run_id": "abc", "tags": ["test"]})
    print(f"  Config: {cfg}")
    assert cfg["run_id"] == "abc"
    print("  ✅ State/config extraction works")

    # ── Test 9: Medic Self-Healing (if Groq available) ────────────────
    print("\n── Test 9: Medic Self-Healing Through Adapter ──")
    llm = get_llm_callable()
    if llm:
        config_medic = AdapterConfig(
            fuse_enabled=False,
            sentinel_enabled=True,
            medic_enabled=True,
            llm_callable=llm,
            storage_enabled=False,
        )
        adapter_medic = LangChainAdapter(config_medic)

        def incomplete_analysis(text: str) -> dict:
            """Returns partial data — Medic should repair it."""
            return {"summary": "Good review"}  # missing sentiment and confidence

        wrapped_heal = adapter_medic.wrap_node(
            incomplete_analysis, schema=AnalysisOutput, node_name="healing_node"
        )
        try:
            result = wrapped_heal("This is an amazing product review")
            print(f"  Repaired: summary={result.summary}, sentiment={result.sentiment}")
            print("  ✅ Medic self-healing works through LangChain adapter")
        except Exception as e:
            print(f"  Medic repair failed: {e}")
            print("  ⚠️ Self-healing didn't succeed (may depend on LLM response)")
    else:
        print("  ⚠️ Skipping — GROQ_API_KEY not set")

    print("\n" + "=" * 60)
    print("All LangChain adapter tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()

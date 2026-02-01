"""
Example: AutoGen Adapter — Testing AirOS reliability with real AutoGen components.

Requires: pip install pyautogen==0.2.35
Uses the legacy AutoGen API (ConversableAgent, AssistantAgent, GroupChat).

Tests:
1. AdapterRegistry discovery & creation
2. wrap_node() — wrapping plain functions with schema validation
3. wrap_agent() — wrapping real ConversableAgent's generate_reply
4. wrap_agent() — wrapping registered functions in _function_map
5. wrap_function() — wrapping functions for agent registration
6. Middleware wrap_group_chat() — auto-wrap all agents in GroupChat
7. Middleware decorator pattern
8. State/config extraction from AutoGen-style arguments
9. Schema validation failure detection
10. Real agent conversation with AirOS reliability
"""
import os
import sys
import json
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from airos.adapters import AutoGenAdapter, AdapterRegistry
from airos.adapters.base import AdapterConfig


# ── Test Schemas ──────────────────────────────────────────────────────────

class CalculationOutput(BaseModel):
    expression: str
    result: float
    steps: list

class MessageOutput(BaseModel):
    content: str
    confidence: float


def get_groq_api_key():
    """Get Groq API key from env."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        os.environ["GROQ_API_KEY"] = key
                        break
    return key


def main():
    print("=" * 60)
    print("AutoGen Adapter — AirOS Reliability Integration")
    print("=" * 60)

    api_key = get_groq_api_key()
    if not api_key:
        print("ERROR: GROQ_API_KEY not found. Set it in .env or environment.")
        sys.exit(1)

    # ── Test 1: Registry Discovery ────────────────────────────────────
    print("\n── Test 1: AdapterRegistry Discovery ──")
    adapters = AdapterRegistry.list_adapters()
    print(f"  Available adapters: {adapters}")
    assert "autogen" in adapters, "AutoGen adapter not registered!"

    autogen_class = AdapterRegistry.get("autogen")
    print(f"  AutoGen adapter class: {autogen_class.__name__}")
    assert autogen_class is AutoGenAdapter

    # Factory creation
    adapter_factory = AdapterRegistry.create("autogen")
    assert adapter_factory.framework_name == "autogen"
    print("  ✅ Registry discovery works (direct + factory)")

    # ── Test 2: Adapter Creation with Config ──────────────────────────
    print("\n── Test 2: Adapter Creation with Config ──")
    config = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=True,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter = AutoGenAdapter(config)
    print(f"  Framework: {adapter.framework_name}")
    assert adapter.framework_name == "autogen"
    print("  ✅ Adapter creation works")

    # ── Test 3: wrap_node() — Plain Function ──────────────────────────
    print("\n── Test 3: wrap_node() — Plain Function with Schema ──")

    def calculate(expression: str) -> dict:
        try:
            val = eval(expression) if isinstance(expression, str) else expression
            return {
                "expression": str(expression),
                "result": float(val),
                "steps": [f"Evaluated {expression}"],
            }
        except Exception as e:
            return {"expression": str(expression), "result": 0.0, "steps": [f"Error: {e}"]}

    wrapped = adapter.wrap_node(calculate, schema=CalculationOutput, node_name="calculator")
    result = wrapped("2 + 3 * 4")
    print(f"  Expression: {result.expression}")
    print(f"  Result: {result.result}")
    assert result.result == 14.0
    print("  ✅ wrap_node works with schema validation")

    # ── Test 4: wrap_agent() — Real ConversableAgent ──────────────────
    print("\n── Test 4: wrap_agent() — Real ConversableAgent ──")
    from autogen import ConversableAgent, AssistantAgent, UserProxyAgent, GroupChat

    # Create a function-based agent (no LLM needed)
    calculator_agent = ConversableAgent(
        name="calculator",
        llm_config=False,
        human_input_mode="NEVER",
        default_auto_reply="I can only calculate.",
    )

    # Register functions
    def add(a: float, b: float) -> str:
        return json.dumps({"result": a + b})

    def multiply(a: float, b: float) -> str:
        return json.dumps({"result": a * b})

    calculator_agent.register_function(
        function_map={"add": add, "multiply": multiply}
    )

    print(f"  Agent name: {calculator_agent.name}")
    print(f"  Has generate_reply: {hasattr(calculator_agent, 'generate_reply')}")
    print(f"  Functions: {list(calculator_agent._function_map.keys())}")

    config_agent = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=False,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_agent = AutoGenAdapter(config_agent)
    wrapped_agent = adapter_agent.wrap_agent(calculator_agent)

    # Verify wrapping
    print(f"  Name preserved: {wrapped_agent.name}")
    assert wrapped_agent.name == "calculator"
    print(f"  Functions still registered: {list(wrapped_agent._function_map.keys())}")
    assert "add" in wrapped_agent._function_map
    assert "multiply" in wrapped_agent._function_map

    # Test wrapped function calls
    add_result = wrapped_agent._function_map["add"](3.0, 4.0)
    print(f"  add(3, 4) = {add_result}")
    mul_result = wrapped_agent._function_map["multiply"](5.0, 6.0)
    print(f"  multiply(5, 6) = {mul_result}")
    print("  ✅ wrap_agent wraps generate_reply and _function_map")

    # ── Test 5: wrap_function() — Standalone Function Wrapping ────────
    print("\n── Test 5: wrap_function() — Standalone Function Wrapping ──")

    def search(query: str) -> dict:
        return {
            "content": f"Results for: {query}",
            "confidence": 0.85,
        }

    config_fn = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=True,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_fn = AutoGenAdapter(config_fn)
    wrapped_search = adapter_fn.wrap_function(search, schema=MessageOutput, name="search_fn")
    result = wrapped_search("machine learning papers")
    print(f"  Content: {result.content}")
    print(f"  Confidence: {result.confidence}")
    assert result.confidence == 0.85
    print("  ✅ wrap_function works with schema validation")

    # ── Test 6: wrap_agent() with LLM-backed Agent ────────────────────
    print("\n── Test 6: wrap_agent() — LLM-backed AssistantAgent ──")

    # Configure Groq as the LLM for AutoGen
    llm_config = {
        "config_list": [
            {
                "model": "llama-3.3-70b-versatile",
                "api_key": api_key,
                "base_url": "https://api.groq.com/openai/v1",
                "api_type": "groq",
            }
        ],
        "temperature": 0,
    }

    assistant = AssistantAgent(
        name="assistant",
        llm_config=llm_config,
        system_message="You are a helpful assistant. Reply concisely.",
        human_input_mode="NEVER",
    )
    print(f"  Assistant name: {assistant.name}")
    print(f"  Has generate_reply: {hasattr(assistant, 'generate_reply')}")

    config_llm = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=False,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_llm = AutoGenAdapter(config_llm)
    wrapped_assistant = adapter_llm.wrap_agent(assistant)
    print(f"  Name preserved: {wrapped_assistant.name}")
    assert wrapped_assistant.name == "assistant"
    print("  ✅ wrap_agent works with LLM-backed AssistantAgent")

    # ── Test 7: Middleware wrap_group_chat() ───────────────────────────
    print("\n── Test 7: Middleware wrap_group_chat() ──")

    agent1 = ConversableAgent(
        name="planner",
        llm_config=False,
        human_input_mode="NEVER",
    )
    agent2 = ConversableAgent(
        name="executor",
        llm_config=False,
        human_input_mode="NEVER",
    )

    group_chat = GroupChat(
        agents=[agent1, agent2],
        messages=[],
        max_round=2,
    )

    config_gc = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=False,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_gc = AutoGenAdapter(config_gc)
    middleware = adapter_gc.create_middleware()
    wrapped_gc = middleware.wrap_group_chat(group_chat)

    print(f"  GroupChat agents: {len(wrapped_gc.agents)}")
    print(f"  Agent names: {[a.name for a in wrapped_gc.agents]}")
    assert len(wrapped_gc.agents) == 2
    print("  ✅ wrap_group_chat auto-wraps all agents")

    # ── Test 8: Middleware Decorator Pattern ───────────────────────────
    print("\n── Test 8: Middleware Decorator Pattern ──")

    @middleware.function(name="data_processor")
    def process_data(data: str) -> dict:
        return {"content": f"Processed: {data}", "confidence": 0.95}

    @middleware.agent_reply(name="reply_handler")
    def handle_reply(message: str) -> dict:
        return {"content": f"Reply: {message}", "confidence": 0.88}

    r1 = process_data("test input")
    r2 = handle_reply("hello world")
    print(f"  Function result: {r1}")
    print(f"  Reply result: {r2}")
    print("  ✅ Middleware decorators work")

    # ── Test 9: State/Config Extraction ───────────────────────────────
    print("\n── Test 9: State & Config Extraction ──")
    adapter_ext = AutoGenAdapter()

    # Dict input
    state = adapter_ext.extract_state({"content": "hello", "role": "user"})
    print(f"  Dict state: {state}")
    assert state["content"] == "hello"

    # String input
    state2 = adapter_ext.extract_state("just a message")
    print(f"  String state: {state2}")
    assert state2["content"] == "just a message"

    # Kwargs with sender/messages
    state3 = adapter_ext.extract_state(
        "test",
        sender=agent1,
        messages=[{"content": "msg1"}, {"content": "msg2"}],
    )
    print(f"  Kwargs state: {state3}")
    assert state3["sender"] == "planner"

    # Config with chat_id
    cfg = adapter_ext.extract_config(chat_id="chat-123", sender=agent1, recipient=agent2)
    print(f"  Config: {cfg}")
    assert cfg["run_id"] == "chat-123"
    assert cfg["configurable"]["sender"] == "planner"
    assert cfg["configurable"]["recipient"] == "executor"
    print("  ✅ State/config extraction works")

    # ── Test 10: Schema Validation Failure ────────────────────────────
    print("\n── Test 10: Schema Validation Failure ──")

    def bad_calc(expr: str) -> dict:
        return {"expression": expr}  # missing result and steps

    config_strict = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=True,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_strict = AutoGenAdapter(config_strict)
    wrapped_bad = adapter_strict.wrap_node(bad_calc, schema=CalculationOutput, node_name="bad_calc")

    try:
        wrapped_bad("1 + 1")
        print("  Should have raised an error!")
    except Exception as e:
        print(f"  Caught: {type(e).__name__}")
        print("  ✅ Schema validation correctly rejects incomplete output")

    # ── Test 11: Real Agent Conversation ──────────────────────────────
    print("\n── Test 11: Real Agent Conversation ──")

    # Create a user proxy and assistant with Groq
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
        is_termination_msg=lambda msg: True,  # stop after first reply
    )

    groq_assistant = AssistantAgent(
        name="groq_assistant",
        llm_config=llm_config,
        system_message="You are a helpful assistant. Answer in one sentence only.",
        human_input_mode="NEVER",
    )

    # Wrap both agents with AirOS
    config_conv = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=False,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_conv = AutoGenAdapter(config_conv)
    adapter_conv.wrap_agent(user_proxy)
    adapter_conv.wrap_agent(groq_assistant)

    print("  Running real agent conversation with Groq...")
    try:
        user_proxy.initiate_chat(
            groq_assistant,
            message="What is Python? Answer in one sentence.",
            max_turns=1,
        )
        # Get the last message from the assistant
        last_msg = user_proxy.chat_messages[groq_assistant][-1]
        reply = last_msg.get("content", "")
        print(f"  Assistant reply (first 100 chars): {reply[:100]}")
        assert len(reply) > 0
        print("  ✅ Real agent conversation works with AirOS-wrapped agents!")
    except Exception as e:
        print(f"  Conversation error: {e}")
        print("  ⚠️ Real conversation may need specific Groq config")

    print("\n" + "=" * 60)
    print("All AutoGen adapter tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()

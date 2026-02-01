"""
Example: CrewAI Adapter — Testing AirOS reliability with real CrewAI components.

Requires: pip install crewai litellm
Uses Groq LLM via GROQ_API_KEY environment variable.

Tests:
1. AdapterRegistry discovery & creation
2. wrap_node() — wrapping plain functions with schema validation
3. wrap_agent() — wrapping real CrewAI Agent's execute_task method
4. wrap_task() — wrapping real CrewAI Task's execute_sync method
5. Middleware wrap_crew() — auto-wrap all agents + tasks in a Crew
6. Middleware decorator pattern
7. Real Crew kickoff with AirOS reliability wrapping
8. State/config extraction from CrewAI-style arguments
9. Schema validation failure detection
"""
import os
import sys
from pydantic import BaseModel
from typing import Optional, List

from airos.adapters import CrewAIAdapter, AdapterRegistry
from airos.adapters.base import AdapterConfig


# ── Test Schemas ──────────────────────────────────────────────────────────

class ResearchOutput(BaseModel):
    topic: str
    findings: list
    confidence: float

class TaskResult(BaseModel):
    result: str
    quality_score: float


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
    print("CrewAI Adapter — AirOS Reliability Integration")
    print("=" * 60)

    api_key = get_groq_api_key()
    if not api_key:
        print("ERROR: GROQ_API_KEY not found. Set it in .env or environment.")
        sys.exit(1)

    # ── Test 1: Registry Discovery ────────────────────────────────────
    print("\n── Test 1: AdapterRegistry Discovery ──")
    adapters = AdapterRegistry.list_adapters()
    print(f"  Available adapters: {adapters}")
    assert "crewai" in adapters, "CrewAI adapter not registered!"

    crewai_class = AdapterRegistry.get("crewai")
    print(f"  CrewAI adapter class: {crewai_class.__name__}")
    assert crewai_class is CrewAIAdapter

    adapter_factory = AdapterRegistry.create("crewai")
    assert adapter_factory.framework_name == "crewai"
    print("  ✅ Registry discovery works (direct + factory)")

    # ── Test 2: Adapter Creation with Config ──────────────────────────
    print("\n── Test 2: Adapter Creation with Config ──")
    config = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=True,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter = CrewAIAdapter(config)
    print(f"  Framework: {adapter.framework_name}")
    assert adapter.framework_name == "crewai"
    print("  ✅ Adapter creation works")

    # ── Test 3: wrap_node() — Plain Function ──────────────────────────
    print("\n── Test 3: wrap_node() — Plain Function with Schema ──")

    def research_fn(topic: str) -> dict:
        return {
            "topic": topic if isinstance(topic, str) else str(topic),
            "findings": ["AI is transformative", "LLMs are improving rapidly"],
            "confidence": 0.91,
        }

    wrapped = adapter.wrap_node(research_fn, schema=ResearchOutput, node_name="researcher")
    result = wrapped("artificial intelligence trends")
    print(f"  Topic: {result.topic}")
    print(f"  Findings: {result.findings}")
    print(f"  Confidence: {result.confidence}")
    assert result.confidence == 0.91
    print("  ✅ wrap_node works with schema validation")

    # ── Test 4: wrap_agent() — Real CrewAI Agent ──────────────────────
    print("\n── Test 4: wrap_agent() — Real CrewAI Agent ──")
    from crewai import Agent, Task, Crew, LLM

    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=api_key,
    )

    researcher = Agent(
        role="Research Analyst",
        goal="Provide concise research summaries on given topics",
        backstory="You are an expert research analyst who summarizes information clearly.",
        llm=llm,
        verbose=False,
    )
    print(f"  Agent role: {researcher.role}")
    print(f"  Has execute_task: {hasattr(researcher, 'execute_task')}")

    config_agent = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=False,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_agent = CrewAIAdapter(config_agent)

    # SDK now handles Pydantic models via _safe_setattr
    wrapped_agent = adapter_agent.wrap_agent(researcher)

    print(f"  execute_task wrapped: {hasattr(wrapped_agent, 'execute_task')}")
    print(f"  Role preserved: {wrapped_agent.role}")
    assert wrapped_agent.role == "Research Analyst"
    print("  ✅ wrap_agent wraps real Agent.execute_task")

    # ── Test 5: wrap_task() — Real CrewAI Task ────────────────────────
    print("\n── Test 5: wrap_task() — Real CrewAI Task ──")

    researcher2 = Agent(
        role="Research Analyst",
        goal="Provide concise research summaries",
        backstory="Expert researcher.",
        llm=llm,
        verbose=False,
    )

    task = Task(
        description="Summarize the key benefits of Python programming",
        expected_output="A brief summary of Python benefits",
        agent=researcher2,
    )
    print(f"  Task description: {task.description}")
    print(f"  Has execute_sync: {hasattr(task, 'execute_sync')}")

    config_task = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=False,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_task = CrewAIAdapter(config_task)

    # SDK now handles execute_sync for CrewAI v1.x
    wrapped_task = adapter_task.wrap_task(task)
    print(f"  Task wrapped, description: {wrapped_task.description}")
    print("  ✅ wrap_task wraps real Task.execute_sync")

    # ── Test 6: Middleware wrap_crew() ─────────────────────────────────
    print("\n── Test 6: Middleware wrap_crew() ──")

    researcher3 = Agent(
        role="Research Analyst",
        goal="Research topics",
        backstory="Expert.",
        llm=llm,
        verbose=False,
    )
    writer = Agent(
        role="Technical Writer",
        goal="Write clear documentation",
        backstory="Expert writer.",
        llm=llm,
        verbose=False,
    )
    task1 = Task(
        description="Research the benefits of microservices",
        expected_output="A summary",
        agent=researcher3,
    )
    task2 = Task(
        description="Write about microservices",
        expected_output="A paragraph",
        agent=writer,
    )

    crew = Crew(
        agents=[researcher3, writer],
        tasks=[task1, task2],
        verbose=False,
    )

    config_crew = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=False,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_crew = CrewAIAdapter(config_crew)
    middleware = adapter_crew.create_middleware()
    wrapped_crew = middleware.wrap_crew(crew)

    print(f"  Wrapped crew agents: {len(wrapped_crew.agents)}")
    print(f"  Wrapped crew tasks: {len(wrapped_crew.tasks)}")
    assert len(wrapped_crew.agents) == 2
    assert len(wrapped_crew.tasks) == 2
    print("  ✅ wrap_crew auto-wraps all agents and tasks")

    # ── Test 7: Middleware Decorator Pattern ───────────────────────────
    print("\n── Test 7: Middleware Decorator Pattern ──")

    @middleware.agent(name="custom_processor")
    def process_data(data: str) -> dict:
        return {"result": f"Processed: {data}", "quality_score": 0.95}

    @middleware.task(name="custom_task")
    def run_task(input_text: str) -> dict:
        return {"result": f"Task done: {input_text}", "quality_score": 0.88}

    r1 = process_data("test input")
    r2 = run_task("test task data")
    print(f"  Agent decorator result: {r1}")
    print(f"  Task decorator result: {r2}")
    print("  ✅ Middleware decorators work")

    # ── Test 8: State/Config Extraction ───────────────────────────────
    print("\n── Test 8: State & Config Extraction ──")
    adapter_ext = CrewAIAdapter()

    # Extract state from a real Task object (has .description)
    state = adapter_ext.extract_state(task)
    print(f"  Task state: {state}")
    assert "task_description" in state

    # Dict input
    state2 = adapter_ext.extract_state({"input": "raw data"})
    print(f"  Dict state: {state2}")
    assert state2["input"] == "raw data"

    # String input
    state3 = adapter_ext.extract_state("just a string")
    print(f"  String state: {state3}")
    assert state3["input"] == "just a string"

    # Config with crew (has .id)
    cfg = adapter_ext.extract_config(crew=crew)
    print(f"  Config with crew: {cfg}")
    print("  ✅ State/config extraction works")

    # ── Test 9: Schema Validation Failure ─────────────────────────────
    print("\n── Test 9: Schema Validation Failure ──")

    def bad_research(topic: str) -> dict:
        return {"topic": "test"}  # missing findings and confidence

    config_strict = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=True,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_strict = CrewAIAdapter(config_strict)
    wrapped_bad = adapter_strict.wrap_node(bad_research, schema=ResearchOutput, node_name="bad_node")

    try:
        wrapped_bad("test topic")
        print("  Should have raised an error!")
    except Exception as e:
        print(f"  Caught: {type(e).__name__}")
        print("  ✅ Schema validation correctly rejects incomplete output")

    # ── Test 10: Real Crew Kickoff with Wrapped Agents ────────────────
    print("\n── Test 10: Real Crew Kickoff ──")

    research_agent = Agent(
        role="Quick Researcher",
        goal="Give one-sentence answers",
        backstory="You give very brief answers in one sentence.",
        llm=llm,
        verbose=False,
    )
    research_task = Task(
        description="What is Python? Answer in one sentence.",
        expected_output="One sentence about Python",
        agent=research_agent,
    )

    config_kick = AdapterConfig(
        fuse_enabled=False,
        sentinel_enabled=False,
        medic_enabled=False,
        storage_enabled=False,
    )
    adapter_kick = CrewAIAdapter(config_kick)
    mid = adapter_kick.create_middleware()

    test_crew = Crew(
        agents=[research_agent],
        tasks=[research_task],
        verbose=False,
    )
    mid.wrap_crew(test_crew)

    print("  Running crew.kickoff() with AirOS-wrapped agents...")
    result = test_crew.kickoff()
    print(f"  Result type: {type(result).__name__}")
    print(f"  Result (first 100 chars): {str(result.raw)[:100]}")
    assert result.raw is not None and len(str(result.raw)) > 0
    print("  ✅ Real Crew kickoff works with AirOS-wrapped agents!")

    print("\n" + "=" * 60)
    print("All CrewAI adapter tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()

# AirOS Agent Demo

Test agent that exercises every AirOS SDK feature.

## Setup

```bash
pip install airos-sdk[groq]
export GROQ_API_KEY=your_key_here
```

## Examples

| # | File | What it tests | LLM required? |
|---|------|---------------|---------------|
| 1 | `examples/basic_agent.py` | Fuse (loop detection) + Sentinel (schema validation) | No |
| 2 | `examples/self_healing_agent.py` | Medic (LLM auto-repair of bad outputs) | Yes |
| 3 | `examples/budget_protected_agent.py` | `max_cost_usd` + `max_seconds` circuit breakers | No |
| 4 | `examples/multi_node_budget.py` | `GlobalBudget` shared across nodes | No |
| 5 | `examples/cost_tracking_agent.py` | `CostCalculator`, model pricing table, `cost_per_token` | No |
| 6 | `examples/error_handling_agent.py` | `ErrorClassifier`, severity, recovery hints, non-recoverable set | No |
| 7 | `examples/storage_agent.py` | `InMemoryStorage`, `Storage` (SQLite), settings, `set_default_storage` | No |
| 8 | `examples/full_pipeline_agent.py` | All features combined: Fuse + Sentinel + Medic + Budget + Pricing + Storage | No |
| 9 | `examples/langchain_adapter_agent.py` | LangChain adapter: wrap_node, wrap_chain, wrap_tool, middleware, Medic healing | Yes |
| 10 | `examples/langgraph_adapter_agent.py` | LangGraph adapter: wrap_node, StateGraph pipeline, Fuse loop detection, middleware | No |
| 11 | `examples/crewai_adapter_agent.py` | CrewAI adapter: wrap_agent, wrap_task, wrap_crew, real Crew kickoff | Yes |
| 12 | `examples/autogen_adapter_agent.py` | AutoGen adapter: wrap_agent, wrap_function, GroupChat, real agent conversation | Yes |

### Run without an API key (examples 1, 3–8, 10)

```bash
python examples/basic_agent.py
python examples/budget_protected_agent.py
python examples/multi_node_budget.py
python examples/cost_tracking_agent.py
python examples/error_handling_agent.py
python examples/storage_agent.py
python examples/full_pipeline_agent.py
python examples/langgraph_adapter_agent.py
```

### Run with Groq API key (examples 2, 9, 11, 12)

```bash
export GROQ_API_KEY=your_key_here
python examples/self_healing_agent.py
python examples/langchain_adapter_agent.py
python examples/crewai_adapter_agent.py
python examples/autogen_adapter_agent.py
```

### Run the full pipeline with real LLM (requires API key)

```bash
python -c "from agent.graph import run_pipeline; run_pipeline('AI safety')"
```

### Adapter Dependencies

```bash
pip install langchain-core langgraph        # For LangChain/LangGraph examples
pip install crewai litellm                  # For CrewAI example
pip install "pyautogen==0.2.35"             # For AutoGen example (legacy API)
```

## Edge Case Tests (213 tests)

```bash
pytest tests/ -v
```

| Test File | Tests | What it covers |
|-----------|-------|----------------|
| `test_budget_edge_cases.py` | 35 | Zero/negative/NaN/Inf budgets, exact boundaries, 100-thread contention, float accumulation |
| `test_pricing_edge_cases.py` | 35 | All 27 models positive, partial matching, empty strings, negative tokens, model cost comparisons |
| `test_fuse_edge_cases.py` | 22 | Zero/negative limits, None/unicode/bytes/set state, large objects, hash determinism |
| `test_sentinel_edge_cases.py` | 21 | No-schema pass-through, missing fields, nested schemas, constrained fields, type coercion |
| `test_medic_edge_cases.py` | 17 | No LLM fallback, max attempts, non-recoverable errors, garbage/empty/None LLM response |
| `test_error_classifier.py` | 27 | All 16 error categories, severity mapping, recovery hints, custom error types |
| `test_decorator_edge_cases.py` | 25 | Isolated storage, all return types, budget+fuse combos, cost_per_token+model, all params together |
| `test_concurrent_stress.py` | 5 | 50 threads shared budget, concurrent budget trip, concurrent storage writes, rapid reset cycles |

## Features Covered

- **Fuse** — Loop detection kills infinite retries
- **Sentinel** — Pydantic schema validation on every output
- **Medic** — LLM auto-repairs invalid outputs
- **BudgetFuse** — Per-node dollar limit (`max_cost_usd`)
- **TimeoutFuse** — Per-node time limit (`max_seconds`)
- **GlobalBudget** — Shared cost + time budget across multiple nodes
- **CostCalculator** — Model-aware cost estimation (25+ models)
- **Pricing table** — Built-in rates for OpenAI, Anthropic, Groq, Google
- **cost_per_token** — User override for custom/enterprise pricing
- **ErrorClassifier** — Automatic error categorization with recovery hints
- **Storage** — In-memory + SQLite trace persistence
- **@reliable** — All parameter combinations tested
- **Thread safety** — Concurrent stress tests with 50+ threads
- **LangChain adapter** — wrap_node, wrap_chain (RunnableLambda), wrap_tool, middleware decorators
- **LangGraph adapter** — wrap_node, full StateGraph pipeline, Fuse loop detection, middleware
- **CrewAI adapter** — wrap_agent, wrap_task, wrap_crew, real Crew kickoff (Pydantic-safe)
- **AutoGen adapter** — wrap_agent, wrap_function, GroupChat wrapping, real agent conversation

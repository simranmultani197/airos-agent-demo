"""
Example 7: Storage — Trace persistence with SQLite.

Shows how AirOS logs every node execution and lets you query traces.
No LLM required.

Run:
    python examples/storage_agent.py
"""
import os
import tempfile
from pydantic import BaseModel
from airos import reliable, set_default_storage
from airos.storage import Storage, InMemoryStorage


class TaskOutput(BaseModel):
    task: str
    status: str


def main():
    print("=" * 50)
    print("AirOS Demo: Storage & Trace Persistence")
    print("=" * 50)

    # ── Part 1: In-memory storage (default) ──────────────────────────────
    print("\n[Part 1] In-memory storage:\n")

    mem_store = InMemoryStorage()

    @reliable(sentinel_schema=TaskOutput, storage=mem_store)
    def task_node(state):
        return {"task": state["name"], "status": "completed"}

    task_node({"name": "task_1"})
    task_node({"name": "task_2"})
    task_node({"name": "task_3"})

    history = mem_store.get_run_history("local_dev_run")
    print(f"  Traces logged: {len(history)}")
    for h in history:
        print(f"    node={h['node_id']}, status={h['status']}")

    cost = mem_store.get_run_cost("local_dev_run")
    print(f"  Total run cost: ${cost:.6f}")

    # ── Part 2: SQLite persistence ───────────────────────────────────────
    print("\n[Part 2] SQLite persistent storage:\n")

    db_path = os.path.join(tempfile.gettempdir(), "airos_demo_traces.db")
    db_store = Storage(db_path=db_path)

    @reliable(sentinel_schema=TaskOutput, storage=db_store)
    def persistent_node(state):
        return {"task": state["name"], "status": "done"}

    persistent_node({"name": "persist_1"})
    persistent_node({"name": "persist_2"})

    history = db_store.get_run_history("local_dev_run")
    print(f"  Traces in DB: {len(history)}")
    print(f"  DB file: {db_path}")
    print(f"  DB exists: {os.path.exists(db_path)}")

    run_cost = db_store.get_run_cost("local_dev_run")
    print(f"  Run cost: ${run_cost:.6f}")

    # ── Part 3: Settings storage ─────────────────────────────────────────
    print("\n[Part 3] Settings storage:\n")

    db_store.set_setting("cost_per_token", "0.00005")
    db_store.set_setting("default_model", "gpt-4o-mini")

    cpt = db_store.get_setting("cost_per_token")
    model = db_store.get_setting("default_model")
    missing = db_store.get_setting("nonexistent_key")

    print(f"  cost_per_token: {cpt}")
    print(f"  default_model: {model}")
    print(f"  nonexistent: {missing}")

    # ── Part 4: set_default_storage ──────────────────────────────────────
    print("\n[Part 4] Global default storage:\n")

    global_store = InMemoryStorage()
    set_default_storage(global_store)

    @reliable(sentinel_schema=TaskOutput)
    def auto_stored_node(state):
        return {"task": state["name"], "status": "auto-stored"}

    auto_stored_node({"name": "global_1"})
    auto_stored_node({"name": "global_2"})

    global_history = global_store.get_run_history("local_dev_run")
    print(f"  Traces via global storage: {len(global_history)}")

    # Reset default storage to avoid affecting other tests
    set_default_storage(InMemoryStorage())

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"\n  Cleaned up: {db_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

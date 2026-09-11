from __future__ import annotations

import asyncio
from types import SimpleNamespace

from pawn_agent.core.agent_runner import run_agent_turn


class FakeRecorder:
    instances: list["FakeRecorder"] = []

    def __init__(self, dsn, *, run_id, thread_id):
        self.dsn = dsn
        self.run_id = run_id
        self.thread_id = thread_id
        self.events: list[tuple[str, dict]] = []
        self.instances.append(self)

    def record(self, event_type: str, **fields) -> None:
        self.events.append((event_type, fields))


class FakeRegistry:
    def __init__(self) -> None:
        self.graph_recorder = None

    async def handle_turn(self, session_id, prompt, cfg, db_dsn, *, graph_recorder=None):
        self.graph_recorder = graph_recorder
        return f"{session_id}:{prompt}:{cfg.pydantic_model}:{db_dsn}"


def test_run_agent_turn_reuses_agent_run_id_for_graph_recorder(monkeypatch) -> None:
    FakeRecorder.instances.clear()
    updates: list[tuple[str, str, dict]] = []

    monkeypatch.setattr(
        "pawn_agent.core.agent_runner.create_agent_run",
        lambda *args, **kwargs: "run-123",
    )
    monkeypatch.setattr(
        "pawn_agent.core.agent_runner.update_agent_run",
        lambda dsn, run_id, status, **kwargs: updates.append((run_id, status, kwargs)),
    )
    monkeypatch.setattr("pawn_agent.core.agent_runner.GraphEventRecorder", FakeRecorder)

    cfg = SimpleNamespace(db_dsn="postgresql://db", pydantic_model="openai:test")
    registry = FakeRegistry()

    result = asyncio.run(
        run_agent_turn(
            cfg=cfg,
            registry=registry,
            prompt="hello",
            session_id="thread-1",
            source="api",
        )
    )

    assert result.run_id == "run-123"
    assert registry.graph_recorder is FakeRecorder.instances[0]
    assert FakeRecorder.instances[0].run_id == "run-123"
    assert FakeRecorder.instances[0].thread_id == "thread-1"
    assert [event[0] for event in FakeRecorder.instances[0].events] == [
        "run_start",
        "run_end",
    ]
    assert updates[0] == ("run-123", "running", {})
    assert updates[-1][0:2] == ("run-123", "completed")

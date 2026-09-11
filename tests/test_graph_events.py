from __future__ import annotations

import asyncio

from pawn_agent.core.graph_events import instrument_node, instrument_router


class FakeRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def record(self, event_type: str, **fields) -> None:
        self.events.append((event_type, fields))


def test_instrument_node_records_sync_start_end() -> None:
    recorder = FakeRecorder()

    def node(state):
        return {"value": state["value"] + 1}

    wrapped = instrument_node("example", node, recorder)

    assert wrapped({"value": 1}) == {"value": 2}
    assert [event[0] for event in recorder.events] == ["node_start", "node_end"]
    assert recorder.events[0][1]["node_name"] == "example"
    assert recorder.events[1][1]["status"] == "completed"
    assert isinstance(recorder.events[1][1]["duration_ms"], int)


def test_instrument_node_records_async_error() -> None:
    recorder = FakeRecorder()

    async def node(_state):
        raise RuntimeError("boom")

    wrapped = instrument_node("bad_node", node, recorder)

    try:
        asyncio.run(wrapped({}))
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")

    assert [event[0] for event in recorder.events] == ["node_start", "error"]
    assert recorder.events[1][1]["status"] == "failed"
    assert recorder.events[1][1]["payload"] == {"error": "boom"}


def test_instrument_router_records_choice_and_edge() -> None:
    recorder = FakeRecorder()

    def router(_state):
        return "respond_fast"

    wrapped = instrument_router("dispatch", router, recorder)

    assert wrapped({}) == "respond_fast"
    assert [event[0] for event in recorder.events] == ["router_decision", "edge_taken"]
    assert recorder.events[0][1]["router_choice"] == "respond_fast"
    assert recorder.events[1][1]["from_node"] == "dispatch"
    assert recorder.events[1][1]["to_node"] == "respond_fast"

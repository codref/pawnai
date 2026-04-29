from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from pawn_agent.core.langgraph_chat import (
    LangGraphRouterChatAgent,
    _next_node_from_dispatch,
)
from pawn_agent.tools.push_queue_message import push_queue_message_impl
from pawn_agent.utils.config import AgentConfig, QueueProducerConfig, load_config

# ── Config parsing ────────────────────────────────────────────────────────────


def test_load_config_parses_queue_producers(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "queue_producers:\n"
        "  matrix:\n"
        "    topic: matrix-jobs\n"
        "    bucket_name: my-bucket\n"
        "    producer_name: test-producer\n"
        "  downstream:\n"
        "    topic: downstream-jobs\n"
        "    bucket_name: shared-bucket\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    assert cfg.queue_producers is not None
    assert set(cfg.queue_producers) == {"matrix", "downstream"}
    assert cfg.queue_producers["matrix"].topic == "matrix-jobs"
    assert cfg.queue_producers["matrix"].bucket_name == "my-bucket"
    assert cfg.queue_producers["matrix"].producer_name == "test-producer"


# ── Tool implementation ───────────────────────────────────────────────────────


def _fake_pawn_queue_builder(message_id: str = "msg-123"):
    """Return a mock PawnQueueBuilder class."""

    class FakeProducer:
        async def publish(self, topic, payload):
            return message_id

    class FakeQueue:
        async def create_topic(self, topic):
            pass

        async def register_producer(self, name):
            return FakeProducer()

    class FakeQueueContext:
        async def __aenter__(self):
            return FakeQueue()

        async def __aexit__(self, *args):
            pass

    class FakeBuilder:
        def s3(self, **kwargs):
            self._s3 = kwargs
            return self

        def polling(self, **kwargs):
            return self

        def concurrency(self, **kwargs):
            return self

        async def build(self):
            return FakeQueueContext()

    return FakeBuilder


def test_push_queue_message_impl_publishes_envelope() -> None:
    cfg = AgentConfig(
        queue_producers={
            "matrix": QueueProducerConfig(topic="matrix-jobs", bucket_name="my-bucket"),
        },
        s3={
            "bucket": "my-bucket",
            "endpoint_url": "http://localhost:9000",
            "access_key": "ak",
            "secret_key": "sk",
        },
    )
    FakeBuilder = _fake_pawn_queue_builder("msg-abc")

    with patch("pawn_queue.PawnQueueBuilder", FakeBuilder):
        result = asyncio.run(
            push_queue_message_impl(
                cfg,
                target="matrix",
                command="run",
                payload={"session_id": "abc123", "prompt": "Summarise"},
            )
        )

    assert "target=matrix" in result
    assert "topic=matrix-jobs" in result
    assert "message_id=msg-abc" in result


def test_push_queue_message_impl_rejects_unknown_target() -> None:
    cfg = AgentConfig(
        queue_producers={
            "matrix": QueueProducerConfig(topic="matrix-jobs", bucket_name="my-bucket"),
        },
        s3={
            "bucket": "my-bucket",
            "endpoint_url": "http://localhost:9000",
            "access_key": "ak",
            "secret_key": "sk",
        },
    )
    result = asyncio.run(
        push_queue_message_impl(
            cfg,
            target="unknown",
            command="run",
            payload={},
        )
    )
    assert "unknown queue target" in result


def test_push_queue_message_impl_rejects_missing_s3() -> None:
    cfg = AgentConfig(
        queue_producers={
            "matrix": QueueProducerConfig(topic="matrix-jobs", bucket_name="my-bucket"),
        },
        s3=None,
    )
    result = asyncio.run(
        push_queue_message_impl(
            cfg,
            target="matrix",
            command="run",
            payload={},
        )
    )
    assert "no s3: section found" in result


def test_push_queue_message_impl_rejects_payload_with_command() -> None:
    cfg = AgentConfig(
        queue_producers={
            "matrix": QueueProducerConfig(topic="matrix-jobs", bucket_name="my-bucket"),
        },
        s3={
            "bucket": "my-bucket",
            "endpoint_url": "http://localhost:9000",
            "access_key": "ak",
            "secret_key": "sk",
        },
    )
    result = asyncio.run(
        push_queue_message_impl(
            cfg,
            target="matrix",
            command="run",
            payload={"command": "bad"},
        )
    )
    assert "must not contain a 'command' key" in result


def test_push_queue_message_impl_rejects_no_queue_producers() -> None:
    cfg = AgentConfig(s3={"bucket": "b"})
    result = asyncio.run(
        push_queue_message_impl(
            cfg,
            target="matrix",
            command="run",
            payload={},
        )
    )
    assert "no queue_producers configured" in result


# ── LangGraph wiring ──────────────────────────────────────────────────────────


def test_langgraph_router_valid_actions_includes_push_queue_message() -> None:
    assert "tool_push_queue_message" in LangGraphRouterChatAgent.VALID_ACTIONS


def test_next_node_from_dispatch_routes_push_queue_message() -> None:
    class FakeState:
        def __init__(self, route_kind: str):
            self._route_kind = route_kind

    def fake_get_state_field(state, key):
        if key == "route_kind":
            return state._route_kind
        return ""

    with patch("pawn_agent.core.langgraph_chat.get_state_field", fake_get_state_field):
        assert (
            _next_node_from_dispatch(FakeState("tool_push_queue_message"))
            == "tool_push_queue_message"
        )

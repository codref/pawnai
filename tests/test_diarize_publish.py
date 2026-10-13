"""diarize_publish: queue_producers target or diarize_queue fallback."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

from pawn_agent.utils.config import AgentQueueConfig, ApiSection, QueueProducerConfig
from pawn_server.core import diarize_publish


def test_publish_prefers_queue_producers(monkeypatch) -> None:
    called: dict = {}

    async def fake_push(cfg, *, target, command, payload):
        called["target"] = target
        called["command"] = command
        called["payload"] = payload
        return "Published to target=diarize"

    monkeypatch.setattr(
        "pawn_agent.tools.push_queue_message.push_queue_message_impl",
        fake_push,
    )
    cfg = SimpleNamespace(
        api=ApiSection(upload_audio_target="diarize"),
        queue_producers={
            "diarize": QueueProducerConfig(topic="audio-chunks", bucket_name="b"),
            "matrix": QueueProducerConfig(topic="m", bucket_name="b"),
        },
        diarize_queue=AgentQueueConfig(topic="other", bucket_name="b"),
    )
    receipt = asyncio.run(
        diarize_publish.publish_to_diarize_queue(
            cfg, command="retranscribe", payload={"session": "s1"}
        )
    )
    assert "Published" in receipt
    assert called["target"] == "diarize"
    assert called["command"] == "retranscribe"


def test_publish_falls_back_to_diarize_queue(monkeypatch) -> None:
    published: dict = {}

    class FakeProducer:
        async def publish(self, topic, envelope):
            published["topic"] = topic
            published["envelope"] = envelope
            return "msg-1"

    class FakeQueue:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def create_topic(self, topic):
            published["created"] = topic

        async def register_producer(self, name):
            published["producer"] = name
            return FakeProducer()

    class FakeBuilder:
        def s3(self, **kwargs):
            published["s3"] = kwargs
            return self

        def polling(self, **kwargs):
            return self

        def concurrency(self, **kwargs):
            return self

        async def build(self):
            return FakeQueue()

    mod = ModuleType("pawn_queue")
    mod.PawnQueueBuilder = FakeBuilder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pawn_queue", mod)

    cfg = SimpleNamespace(
        api=ApiSection(upload_audio_target="diarize"),
        queue_producers={
            "matrix": QueueProducerConfig(topic="m", bucket_name="b"),
        },
        diarize_queue=AgentQueueConfig(
            topic="pawn-diarize-jobs",
            bucket_name="codref01",
        ),
        s3_config={
            "endpoint_url": "http://localhost:9000",
            "access_key": "a",
            "secret_key": "s",
            "region": "us-east-1",
            "verify_ssl": False,
        },
    )
    receipt = asyncio.run(
        diarize_publish.publish_to_diarize_queue(
            cfg, command="rediarize", payload={"session": "meet-1", "push_vault": True}
        )
    )
    assert "diarize_queue" in receipt
    assert published["topic"] == "pawn-diarize-jobs"
    assert published["envelope"]["command"] == "rediarize"
    assert published["envelope"]["session"] == "meet-1"
    assert published["s3"]["bucket_name"] == "codref01"


def test_publish_errors_when_neither_configured() -> None:
    cfg = SimpleNamespace(
        api=ApiSection(upload_audio_target="diarize"),
        queue_producers={"matrix": QueueProducerConfig(topic="m", bucket_name="b")},
        diarize_queue=None,
    )
    receipt = asyncio.run(
        diarize_publish.publish_to_diarize_queue(
            cfg, command="retranscribe", payload={"session": "s"}
        )
    )
    assert receipt.startswith("Error:")
    assert "diarize" in receipt
    assert "matrix" in receipt


def test_resolve_source() -> None:
    cfg = SimpleNamespace(
        api=ApiSection(upload_audio_target="diarize"),
        queue_producers={"matrix": QueueProducerConfig(topic="m", bucket_name="b")},
        diarize_queue=AgentQueueConfig(topic="t", bucket_name="b"),
    )
    assert diarize_publish.resolve_diarize_publish_source(cfg) == "diarize_queue"

    cfg.queue_producers["diarize"] = QueueProducerConfig(topic="t", bucket_name="b")
    assert (
        diarize_publish.resolve_diarize_publish_source(cfg) == "queue_producers:diarize"
    )

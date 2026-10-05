"""Tests for pawn-server queue admin (multi-target empty / stats / pause)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from pawn_agent.utils.config import AgentConfig, AgentQueueConfig, QueueProducerConfig
from pawn_core.queue_control import listen_respecting_pause
from pawn_server.cli.commands import app
from pawn_server.core.queue_admin import (
    discover_queue_targets,
    empty_queue,
    queue_stats,
    resolve_queue_targets,
    set_queue_paused,
)

runner = CliRunner()


def _cfg(**kwargs: Any) -> AgentConfig:
    defaults: dict[str, Any] = {
        "s3": {
            "endpoint_url": "http://localhost:9000",
            "access_key": "ak",
            "secret_key": "sk",
            "bucket": "unused",
            "region": "us-east-1",
        },
        "agent_queue": AgentQueueConfig(
            topic="pawn-agent-jobs",
            bucket_name="agent-bucket",
            consumer_name="pawn-agent-listener",
        ),
        "diarize_queue": AgentQueueConfig(
            topic="audio-chunks",
            bucket_name="agent-bucket",
            consumer_name="pawn-diarize-listener",
        ),
        "queue_producers": {
            "matrix": QueueProducerConfig(
                topic="matrix-bot-notifications",
                bucket_name="agent-bucket",
            )
        },
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


class _FakeQueue:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def __aenter__(self) -> "_FakeQueue":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


def test_discover_queue_targets_includes_agent_diarize_producers() -> None:
    targets = discover_queue_targets(_cfg())
    by_name = {t.name: t for t in targets}
    assert set(by_name) == {"agent", "diarize", "matrix"}
    assert by_name["diarize"].topic == "audio-chunks"
    assert by_name["diarize"].source == "diarize_queue"
    assert by_name["matrix"].source == "queue_producers"


def test_resolve_requires_selector_when_multiple() -> None:
    with pytest.raises(RuntimeError, match="--name"):
        resolve_queue_targets(_cfg())


def test_resolve_by_name_and_topic() -> None:
    by_name = resolve_queue_targets(_cfg(), name="diarize")
    assert [t.topic for t in by_name] == ["audio-chunks"]
    by_topic = resolve_queue_targets(_cfg(), topic="pawn-agent-jobs")
    assert [t.name for t in by_topic] == ["agent"]


def test_empty_queue_deletes_messages_and_leases() -> None:
    client = MagicMock()
    client.list_objects = AsyncMock(
        side_effect=[
            ["pawn-agent-jobs/messages/a.json", "pawn-agent-jobs/messages/b.json"],
            ["pawn-agent-jobs/leases/a.lease"],
        ]
    )
    client.delete_objects = AsyncMock()

    with patch(
        "pawn_server.core.queue_admin._build_queue_for_bucket",
        new=AsyncMock(return_value=_FakeQueue(client)),
    ):
        results = asyncio.run(empty_queue(_cfg(), name="agent"))

    assert len(results) == 1
    result = results[0]
    assert result.name == "agent"
    assert result.messages == 2
    assert result.leases == 1
    assert result.total == 3
    assert client.delete_objects.await_count == 2


def test_empty_queue_dry_run_does_not_delete() -> None:
    client = MagicMock()
    client.list_objects = AsyncMock(side_effect=[["audio-chunks/messages/a.json"], []])
    client.delete_objects = AsyncMock()

    with patch(
        "pawn_server.core.queue_admin._build_queue_for_bucket",
        new=AsyncMock(return_value=_FakeQueue(client)),
    ):
        results = asyncio.run(empty_queue(_cfg(), name="diarize", dry_run=True))

    assert results[0].messages == 1
    assert results[0].dry_run is True
    client.delete_objects.assert_not_awaited()


def test_queue_stats_defaults_to_all_targets() -> None:
    client = MagicMock()
    client.list_objects = AsyncMock(return_value=[])
    client.object_exists = AsyncMock(return_value=False)

    with patch(
        "pawn_server.core.queue_admin._build_queue_for_bucket",
        new=AsyncMock(return_value=_FakeQueue(client)),
    ):
        rows = asyncio.run(queue_stats(_cfg()))

    assert [r.name for r in rows] == ["agent", "diarize", "matrix"]


def test_set_queue_paused_writes_marker_for_named_target() -> None:
    client = MagicMock()
    client.object_exists = AsyncMock(return_value=False)
    client.put_object = AsyncMock()

    with patch(
        "pawn_server.core.queue_admin._build_queue_for_bucket",
        new=AsyncMock(return_value=_FakeQueue(client)),
    ):
        results = asyncio.run(set_queue_paused(_cfg(), paused=True, name="diarize"))

    assert results[0].changed is True
    assert results[0].name == "diarize"
    assert results[0].topic == "audio-chunks"
    client.put_object.assert_awaited()


def test_listen_respecting_pause_skips_poll_while_paused() -> None:
    consumer = MagicMock()
    consumer._config.polling.interval_seconds = 0.01
    consumer.poll = AsyncMock(return_value=[])

    async def _refresher() -> None:
        await asyncio.sleep(10)

    consumer._lease_refresher = _refresher

    async def _pause_state(_client: Any, _topic: str):
        return True, "t0"

    async def _run() -> None:
        with patch("pawn_core.queue_control.read_pause_state", new=_pause_state):
            task = asyncio.create_task(
                listen_respecting_pause(consumer, AsyncMock(), MagicMock(), "audio-chunks")
            )
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())
    assert consumer.poll.await_count == 0


def test_cli_queue_stats() -> None:
    from pawn_server.core.queue_admin import QueueStats

    fake = [
        QueueStats(
            name="agent",
            source="agent_queue",
            topic="pawn-agent-jobs",
            bucket="agent-bucket",
            messages=2,
            leases=0,
            dead_letters=0,
            paused=False,
        ),
        QueueStats(
            name="diarize",
            source="diarize_queue",
            topic="audio-chunks",
            bucket="agent-bucket",
            messages=5,
            leases=1,
            dead_letters=0,
            paused=False,
        ),
    ]
    with (
        patch("pawn_agent.utils.config.load_config", return_value=_cfg()),
        patch(
            "pawn_server.core.queue_admin.queue_stats",
            new=AsyncMock(return_value=fake),
        ),
    ):
        result = runner.invoke(app, ["queue", "stats"])

    assert result.exit_code == 0
    assert "diarize" in result.stdout
    assert "agent" in result.stdout
    assert "diarize_queue" in result.stdout


def test_cli_queue_empty_aborts_without_yes() -> None:
    with (
        patch("pawn_agent.utils.config.load_config", return_value=_cfg()),
        patch(
            "pawn_server.core.queue_admin.resolve_queue_targets",
            return_value=discover_queue_targets(_cfg())[:1],
        ),
    ):
        result = runner.invoke(app, ["queue", "empty", "--name", "agent"], input="n\n")

    assert result.exit_code == 0
    assert "Aborted" in result.stdout


def test_cli_queue_pause_requires_selector() -> None:
    with patch("pawn_agent.utils.config.load_config", return_value=_cfg()):
        result = runner.invoke(app, ["queue", "pause"])

    assert result.exit_code == 1
    assert "--name" in result.stdout or "Multiple queues" in result.stdout


def test_cli_queue_pause_named() -> None:
    from pawn_server.core.queue_admin import PauseResult

    paused = PauseResult(
        name="diarize",
        topic="audio-chunks",
        bucket="agent-bucket",
        paused=True,
        changed=True,
    )
    with (
        patch("pawn_agent.utils.config.load_config", return_value=_cfg()),
        patch(
            "pawn_server.core.queue_admin.set_queue_paused",
            new=AsyncMock(return_value=[paused]),
        ),
    ):
        result = runner.invoke(app, ["queue", "pause", "--name", "diarize"])

    assert result.exit_code == 0
    assert "Paused diarize" in result.stdout

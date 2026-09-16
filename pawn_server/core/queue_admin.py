"""Admin helpers for pawn-queue topics configured in pawnai.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pawn_core.queue_control import pause_marker_key, read_pause_state
from pawn_queue.utils import canonical_json, utcnow_iso

#: Max keys per S3 DeleteObjects call.
_DELETE_BATCH = 1000


@dataclass(frozen=True)
class QueueTarget:
    """One configured queue endpoint (listener or producer topic)."""

    name: str
    source: str
    topic: str
    bucket_name: str


@dataclass
class EmptyQueueResult:
    """Counts of objects removed (or that would be removed) when emptying a topic."""

    name: str
    topic: str
    bucket: str
    messages: int = 0
    leases: int = 0
    dead_letters: int = 0
    dry_run: bool = False
    keys: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.messages + self.leases + self.dead_letters


@dataclass
class QueueStats:
    """Snapshot of pending work and control state for one topic."""

    name: str
    source: str
    topic: str
    bucket: str
    messages: int
    leases: int
    dead_letters: int
    paused: bool
    paused_at: Optional[str] = None


@dataclass
class PauseResult:
    """Outcome of pausing or resuming a topic."""

    name: str
    topic: str
    bucket: str
    paused: bool
    changed: bool
    paused_at: Optional[str] = None


def _require_s3(cfg: Any) -> dict[str, Any]:
    s3_cfg: Optional[dict[str, Any]] = cfg.s3_config
    if not s3_cfg:
        raise RuntimeError(
            "No 's3:' section found in pawnai.yaml. "
            "Queue admin commands require S3 credentials in the "
            "top-level 's3:' section."
        )
    return s3_cfg


def discover_queue_targets(cfg: Any) -> list[QueueTarget]:
    """Return configured queue endpoints from agent / diarize / producers."""
    targets: list[QueueTarget] = []
    seen_names: set[str] = set()

    def _add(name: str, source: str, topic: str, bucket_name: str) -> None:
        if name in seen_names:
            raise RuntimeError(
                f"Duplicate queue name {name!r} in config "
                f"(source={source}). Rename the producer or section."
            )
        seen_names.add(name)
        targets.append(
            QueueTarget(
                name=name,
                source=source,
                topic=topic,
                bucket_name=bucket_name,
            )
        )

    agent_queue = getattr(cfg, "agent_queue", None)
    if agent_queue is not None:
        data = agent_queue.model_dump() if hasattr(agent_queue, "model_dump") else dict(agent_queue)
        _add(
            "agent",
            "agent_queue",
            data.get("topic") or "pawn-agent-jobs",
            data.get("bucket_name") or "pawn-agent-queue",
        )

    diarize_queue = getattr(cfg, "diarize_queue", None)
    if diarize_queue is not None:
        data = (
            diarize_queue.model_dump()
            if hasattr(diarize_queue, "model_dump")
            else dict(diarize_queue)
        )
        _add(
            "diarize",
            "diarize_queue",
            data.get("topic") or "audio-chunks",
            data.get("bucket_name") or "pawn-diarize-queue",
        )

    producers = getattr(cfg, "queue_producers", None) or {}
    for prod_name, producer in producers.items():
        data = producer.model_dump() if hasattr(producer, "model_dump") else dict(producer)
        _add(
            str(prod_name),
            "queue_producers",
            data["topic"],
            data["bucket_name"],
        )

    return targets


def resolve_queue_targets(
    cfg: Any,
    *,
    name: Optional[str] = None,
    topic: Optional[str] = None,
    all_targets: bool = False,
    default_all: bool = False,
) -> list[QueueTarget]:
    """Select one or more configured targets.

    * ``all_targets`` / ``default_all`` → every configured queue
    * ``name`` → exact configured name (``agent``, ``diarize``, producer key)
    * ``topic`` → all targets whose topic matches
    * otherwise, if exactly one target exists, return it; else raise
    """
    targets = discover_queue_targets(cfg)
    if not targets:
        raise RuntimeError(
            "No queues configured in pawnai.yaml. "
            "Add agent_queue:, diarize_queue:, and/or queue_producers:."
        )

    if all_targets or default_all:
        return targets

    if name and topic:
        raise RuntimeError("Pass either --name or --topic, not both.")

    if name:
        matched = [t for t in targets if t.name == name]
        if not matched:
            known = ", ".join(t.name for t in targets)
            raise RuntimeError(f"Unknown queue name {name!r}. Known: {known}")
        return matched

    if topic:
        matched = [t for t in targets if t.topic == topic]
        if not matched:
            known = ", ".join(f"{t.name}={t.topic}" for t in targets)
            raise RuntimeError(f"No configured queue uses topic {topic!r}. Known: {known}")
        return matched

    if len(targets) == 1:
        return targets

    known = ", ".join(f"{t.name} ({t.topic})" for t in targets)
    raise RuntimeError(
        "Multiple queues are configured; pass --name, --topic, or --all. "
        f"Known: {known}"
    )


async def _build_queue_for_bucket(cfg: Any, bucket_name: str):
    """Build a started :class:`pawn_queue.PawnQueue` for *bucket_name*."""
    try:
        from pawn_queue import PawnQueueBuilder
    except ImportError as exc:
        raise ImportError(
            "pawn-queue is not installed. Run: uv pip install pawn-queue"
        ) from exc

    s3_cfg = _require_s3(cfg)
    endpoint_url: str = s3_cfg.get("endpoint_url", "http://localhost:9000")
    use_ssl: bool = bool(s3_cfg.get("verify_ssl", s3_cfg.get("use_ssl", False)))

    builder = PawnQueueBuilder()
    builder = builder.s3(
        endpoint_url=endpoint_url,
        bucket_name=bucket_name,
        access_key=s3_cfg.get("access_key", s3_cfg.get("aws_access_key_id", "")),
        secret_key=s3_cfg.get("secret_key", s3_cfg.get("aws_secret_access_key", "")),
        region_name=s3_cfg.get("region", s3_cfg.get("region_name", "us-east-1")),
        use_ssl=use_ssl,
    )
    return await builder.build()


async def _delete_prefix_keys(client: Any, prefix: str, *, dry_run: bool) -> list[str]:
    keys: list[str] = await client.list_objects(prefix)
    if not keys or dry_run:
        return keys
    for i in range(0, len(keys), _DELETE_BATCH):
        batch = keys[i : i + _DELETE_BATCH]  # noqa: E203
        await client.delete_objects(batch)
    return keys


async def empty_queue(
    cfg: Any,
    *,
    name: Optional[str] = None,
    topic: Optional[str] = None,
    all_targets: bool = False,
    include_dead_letter: bool = False,
    dry_run: bool = False,
) -> list[EmptyQueueResult]:
    """Delete pending messages and leases for one or more configured topics."""
    targets = resolve_queue_targets(
        cfg, name=name, topic=topic, all_targets=all_targets
    )
    results: list[EmptyQueueResult] = []
    for target in targets:
        async with await _build_queue_for_bucket(cfg, target.bucket_name) as pq:
            client = pq._client  # noqa: SLF001
            message_keys = await _delete_prefix_keys(
                client, f"{target.topic}/messages/", dry_run=dry_run
            )
            lease_keys = await _delete_prefix_keys(
                client, f"{target.topic}/leases/", dry_run=dry_run
            )
            dead_keys: list[str] = []
            if include_dead_letter:
                dead_keys = await _delete_prefix_keys(
                    client, f"{target.topic}/dead-letter/", dry_run=dry_run
                )
        results.append(
            EmptyQueueResult(
                name=target.name,
                topic=target.topic,
                bucket=target.bucket_name,
                messages=len(message_keys),
                leases=len(lease_keys),
                dead_letters=len(dead_keys),
                dry_run=dry_run,
                keys=message_keys + lease_keys + dead_keys,
            )
        )
    return results


async def queue_stats(
    cfg: Any,
    *,
    name: Optional[str] = None,
    topic: Optional[str] = None,
    all_targets: bool = False,
) -> list[QueueStats]:
    """Count pending messages, leases, and dead-letters for configured topics.

    With no selector, returns stats for every discovered queue.
    """
    targets = resolve_queue_targets(
        cfg,
        name=name,
        topic=topic,
        all_targets=all_targets,
        default_all=not name and not topic and not all_targets,
    )
    results: list[QueueStats] = []
    for target in targets:
        async with await _build_queue_for_bucket(cfg, target.bucket_name) as pq:
            client = pq._client  # noqa: SLF001
            message_keys = await client.list_objects(f"{target.topic}/messages/")
            lease_keys = await client.list_objects(f"{target.topic}/leases/")
            dead_keys = await client.list_objects(f"{target.topic}/dead-letter/")
            paused, paused_at = await read_pause_state(client, target.topic)
        results.append(
            QueueStats(
                name=target.name,
                source=target.source,
                topic=target.topic,
                bucket=target.bucket_name,
                messages=len(message_keys),
                leases=len(lease_keys),
                dead_letters=len(dead_keys),
                paused=paused,
                paused_at=paused_at,
            )
        )
    return results


async def set_queue_paused(
    cfg: Any,
    *,
    paused: bool,
    name: Optional[str] = None,
    topic: Optional[str] = None,
    all_targets: bool = False,
) -> list[PauseResult]:
    """Write or clear S3 pause markers so listeners stop claiming work."""
    targets = resolve_queue_targets(
        cfg, name=name, topic=topic, all_targets=all_targets
    )
    results: list[PauseResult] = []
    for target in targets:
        key = pause_marker_key(target.topic)
        async with await _build_queue_for_bucket(cfg, target.bucket_name) as pq:
            client = pq._client  # noqa: SLF001
            currently_paused, paused_at = await read_pause_state(client, target.topic)
            if paused:
                if currently_paused:
                    results.append(
                        PauseResult(
                            name=target.name,
                            topic=target.topic,
                            bucket=target.bucket_name,
                            paused=True,
                            changed=False,
                            paused_at=paused_at,
                        )
                    )
                    continue
                paused_at = utcnow_iso()
                body = canonical_json(
                    {
                        "paused": True,
                        "paused_at": paused_at,
                        "topic": target.topic,
                    }
                )
                await client.put_object(key, body)
                results.append(
                    PauseResult(
                        name=target.name,
                        topic=target.topic,
                        bucket=target.bucket_name,
                        paused=True,
                        changed=True,
                        paused_at=paused_at,
                    )
                )
                continue

            if not currently_paused:
                results.append(
                    PauseResult(
                        name=target.name,
                        topic=target.topic,
                        bucket=target.bucket_name,
                        paused=False,
                        changed=False,
                        paused_at=None,
                    )
                )
                continue
            await client.delete_object(key)
            results.append(
                PauseResult(
                    name=target.name,
                    topic=target.topic,
                    bucket=target.bucket_name,
                    paused=False,
                    changed=True,
                    paused_at=None,
                )
            )
    return results

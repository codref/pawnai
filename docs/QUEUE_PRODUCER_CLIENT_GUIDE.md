# Queue Producer Client Guide

The `push_queue_message` tool lets the LangGraph agent send **progress notifications** and **alerts** to an external queue (e.g. a Matrix bot).  When a user says *"keep me posted"*, the planner inserts `tool_push_queue_message` at key points in the action plan so downstream consumers can surface live status updates.

This guide covers the producer configuration, the notification message contract, and how to implement a client-side consumer.

---

## Configuration

Add a `queue_producers:` section to `pawnai.yaml`.  Each key is a **target name**; the agent prefers a target named `matrix` for notifications, otherwise it falls back to the first target.

```yaml
s3:
  bucket: my-bucket
  endpoint_url: https://s3.amazonaws.com
  access_key: AKIAIOSFODNN7EXAMPLE
  secret_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
  region: us-east-1
  verify_ssl: true

queue_producers:
  matrix:
    topic: matrix-jobs
    bucket_name: my-bucket
  downstream:
    topic: downstream-jobs
    bucket_name: shared-bucket
```

S3 credentials are always read from the shared top-level `s3:` section.

### Producer target fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `topic` | yes | — | Queue topic to publish to |
| `bucket_name` | yes | — | S3 bucket for queue storage |
| `producer_name` | no | `pawn-agent-producer-<target>` | Producer registration name |
| `polling` | no | — | Dict with `interval_seconds`, `max_messages_per_poll`, `visibility_timeout_seconds`, `lease_refresh_interval_seconds`, `jitter_max_ms` |
| `concurrency` | no | — | Dict with `strategy` (`auto`, `conditional_write`, `csprng_verify`) |

---

## Tool behaviour

### Notification mode (default)

When the planner routes to `tool_push_queue_message` as part of a normal workflow, the node **auto-generates** a progress message from the agent's current state:

```json
{
  "command": "notify",
  "message": "Completed: query_conversation | Next: reply_deep | Session: meeting-20260321",
  "completed_step": "query_conversation",
  "remaining_steps": ["reply_deep"],
  "session_id": "meeting-20260321",
  "original_prompt": "provide a deep analysis of meeting-20260321 and keep me posted"
}
```

The agent can include `tool_push_queue_message` **multiple times** in a single action plan (e.g. before and after a long-running step) so the user receives step-by-step updates.

### Explicit mode

If the user's message explicitly names a target and payload (e.g. *"publish a run command to downstream with session_id abc123"*), the fast model extracts the parameters and the tool publishes the exact envelope supplied by the user.

### Receipt

On success the tool returns:

```
Published to target=matrix topic=matrix-jobs message_id=0192a3f4-...
```

On error it returns a descriptive string (no exception thrown to the graph).

---

## Prompts that trigger notifications

| Prompt | Planner behaviour |
|--------|-------------------|
| *"Provide a deep analysis of xyz and keep me posted"* | Inserts `tool_push_queue_message` before/after key steps |
| *"Analyze this session and alert me via Matrix when done"* | Final step sends completion notification |
| *"Keep me updated on your progress"* | Multiple progress updates during the plan |
| *"Send a notification to matrix that I'm waiting"* | Explicit notification with custom message |

---

## Client consumer contract

A consumer is any process that registers a **consumer** on the same topic and handles the messages.

### Notification envelope

Notification-mode messages always have `command: "notify"` and a `message` field:

```json
{
  "command": "notify",
  "message": "Completed: query_conversation | Next: reply_deep | Session: meeting-20260321",
  "completed_step": "query_conversation",
  "remaining_steps": ["reply_deep"],
  "session_id": "meeting-20260321",
  "original_prompt": "provide a deep analysis of meeting-20260321 and keep me posted"
}
```

Explicit-mode messages have whatever `command` the user or agent specified:

```json
{
  "command": "run",
  "session_id": "abc123",
  "prompt": "Summarise and save to SiYuan"
}
```

Consumers **should** validate `command` and reject unknown commands (nack / dead-letter).

### Ack / nack expectations

- **Ack** the message after successful processing.
- **Nack** the message on any failure so it moves to the dead-letter queue for inspection.
- Keep processing idempotent where possible — `pawn-queue` delivers at-least-once.

---

## Minimal consumer example (Matrix notifier)

```python
import asyncio
from pawn_queue import PawnQueueBuilder


async def consume_matrix_notifications():
    builder = (
        PawnQueueBuilder()
        .s3(
            endpoint_url="https://s3.amazonaws.com",
            bucket_name="my-bucket",
            access_key="AKIA...",
            secret_key="wJal...",
        )
    )

    async with await builder.build() as pq:
        await pq.create_topic("matrix-jobs")
        consumer = await pq.register_consumer("matrix-bot", topics=["matrix-jobs"])

        async def handler(msg):
            payload = dict(msg.payload)
            command = payload.pop("command", None)

            if command == "notify":
                text = payload.get("message", "Agent update")
                session_id = payload.get("session_id")
                print(f"[NOTIFY] {session_id}: {text}")
                # Forward to Matrix room here ...
                await msg.ack()
                return

            if command == "run":
                prompt = payload.get("prompt", "")
                session_id = payload.get("session_id")
                print(f"[RUN] {session_id}: {prompt}")
                # Execute job ...
                await msg.ack()
                return

            print(f"Unknown command {command!r} — nack")
            await msg.nack()

        await consumer.listen(handler)


if __name__ == "__main__":
    asyncio.run(consume_matrix_notifications())
```

---

## Dead-letter queue

Failed messages are moved to:

```
<bucket>/<topic>/dead-letter/<message-id>.json
```

Inspect and replay by re-publishing the payload after fixing the root cause.

---

## Idempotency notes

- The producer creates the topic idempotently on every call.
- The producer registers itself with a stable name (`producer_name`) so redundant registrations are harmless.
- Message IDs are generated by `pawn-queue` and are unique.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `no queue_producers configured` | Missing `queue_producers:` section in `pawnai.yaml` |
| `no s3: section found` | Missing top-level `s3:` credentials in `pawnai.yaml` |
| `payload must not contain a 'command' key` | The LLM included `"command"` inside the payload dict (explicit mode only) |
| `pawn-queue is not installed` | `pip install pawn-queue` (or `uv pip install pawn-queue`) |

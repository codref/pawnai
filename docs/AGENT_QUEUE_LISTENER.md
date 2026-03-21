# Pawn Agent Queue Listener

Pawn Agent can operate as a background worker that consumes jobs from a message queue instead of being driven interactively.  The underlying queue is [pawn-queue](https://github.com/codref/pawn-queue) — a lightweight, S3-backed, at-least-once queue that works with any S3-compatible object store (AWS S3, MinIO, Hetzner Object Storage, Cloudflare R2, …).

Because the listener runs headless (no terminal), all results are persisted to the `agent_runs` PostgreSQL table where they can be queried later.

---

## Architecture

```
Producer (script / CI / API)
        │  publishes JSON message
        ▼
   pawn-queue topic  (stored as objects in an S3 bucket)
        │
        │  poll every N seconds
        ▼
  pawn-agent listen  (consumer worker)
        │  reads payload, runs PydanticAgent with all tools
        ▼
  PostgreSQL agent_runs table  (prompt, response, status)
```

Each message lives as a small JSON object in S3.  When `pawn-agent listen` picks up a message it writes a **lease** file to claim it; if processing succeeds the message is **acked** (deleted); if it fails the message is moved to a **dead-letter prefix** for manual inspection.

---

## Configuration

The queue listener reuses the top-level `s3:` section for credentials and has its own dedicated `agent_queue:` section — separate from the `queue:` section used by pawn-diarize, so both listeners can run independently with different topics:

```yaml
# Shared S3 connection — used by both pawn-diarize and pawn-agent
s3:
  bucket: my-bucket
  endpoint_url: https://s3.amazonaws.com
  access_key: AKIAIOSFODNN7EXAMPLE
  secret_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
  region: us-east-1
  verify_ssl: true

# pawn-diarize queue (separate section)
diarize_queue:
  topic: audio-chunks
  consumer_name: pawn-diarize-listener
  bucket_name: my-bucket

# pawn-agent queue
agent_queue:
  topic: pawn-agent-jobs              # default: pawn-agent-jobs
  consumer_name: pawn-agent-listener  # default: pawn-agent-listener
  bucket_name: my-bucket              # S3 bucket for queue storage
  polling:
    interval_seconds: 5
    max_messages_per_poll: 1
    visibility_timeout_seconds: 300
    lease_refresh_interval_seconds: 60
    jitter_max_ms: 200
  concurrency:
    strategy: auto                    # auto | conditional_write | csprng_verify

# Agent settings (model, tools, etc.)
agent:
  name: Bob
  openai:
    model: gpt-4o
    base_url: http://localhost:11434/v1
    api_key: ollama
```

---

## Running the Listener

```bash
# Basic — uses pawnai.yaml in current directory
pawn-agent listen

# Custom config file
pawn-agent listen --config /etc/pawnai/pawnai.yaml

# Override topic and consumer name
pawn-agent listen --topic my-agent-topic --consumer-name worker-02
```

The listener blocks until interrupted with Ctrl-C.

### CLI Options

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to YAML config file |
| `--topic` | `-T` | Topic name (overrides config) |
| `--consumer-name` | `-n` | Consumer name (overrides config) |

---

## Message Format

Messages are JSON objects with a required `command` key.  Currently one command is supported: `run`.

### `run` Command

Execute a natural-language prompt through the PydanticAgent with all tools available.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `command` | yes | string | Must be `"run"` |
| `prompt` | yes | string | Natural-language instruction for the agent |
| `session_id` | no | string | Session ID hint (prepended to prompt as `[Session ID: ...]`) |
| `model` | no | string | Model override (e.g. `"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-5-20251001"`) |

### Example Messages

**Basic prompt:**
```json
{
  "command": "run",
  "prompt": "What sessions are available in the database?"
}
```

**Analyze a session:**
```json
{
  "command": "run",
  "prompt": "Analyze this session and save the results to SiYuan",
  "session_id": "meeting-20260321"
}
```

**With model override:**
```json
{
  "command": "run",
  "prompt": "Extract knowledge graph triples from this conversation",
  "session_id": "interview-042",
  "model": "anthropic:claude-sonnet-4-5-20251001"
}
```

**Vectorize a session for RAG:**
```json
{
  "command": "run",
  "prompt": "Vectorize session meeting-20260321 into the RAG index",
  "session_id": "meeting-20260321"
}
```

---

## Agent Run History

Every message processed by the listener creates a row in the `agent_runs` table:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `message_id` | string | pawn-queue message ID for traceability |
| `command` | string | `"run"` |
| `prompt` | text | The prompt sent to the agent |
| `session_id` | string | Session ID hint (if provided) |
| `model` | string | Model used for this run |
| `status` | string | `pending` → `running` → `completed` / `failed` |
| `response` | text | Agent output (on success) |
| `error` | text | Error message (on failure) |
| `created_at` | timestamp | When the message was received |
| `started_at` | timestamp | When processing began |
| `completed_at` | timestamp | When processing finished |

### Status Lifecycle

```
pending  →  running  →  completed
                     →  failed
```

### Querying Results

```sql
-- Recent completed runs
SELECT id, session_id, model, prompt, LEFT(response, 200), completed_at
FROM agent_runs
WHERE status = 'completed'
ORDER BY completed_at DESC
LIMIT 10;

-- Failed runs
SELECT id, prompt, error, created_at
FROM agent_runs
WHERE status = 'failed'
ORDER BY created_at DESC;

-- Runs for a specific session
SELECT * FROM agent_runs
WHERE session_id = 'meeting-20260321'
ORDER BY created_at DESC;
```

### Migration

Apply the migration to create the `agent_runs` table:

```bash
alembic upgrade head
```

---

## Publishing Messages

Use pawn-queue's producer API to publish messages:

```python
import asyncio
from pawn_queue import PawnQueueBuilder


async def publish_agent_job():
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
        await pq.create_topic("pawn-agent-jobs")
        producer = await pq.register_producer("my-app")

        message_id = await producer.publish("pawn-agent-jobs", {
            "command": "run",
            "prompt": "Analyze this session and push to SiYuan",
            "session_id": "meeting-20260321",
        })
        print(f"Published: {message_id}")


asyncio.run(publish_agent_job())
```

---

## Dead-Letter Queue

Messages that fail processing are moved to:

```
<bucket>/<topic>/dead-letter/<message-id>.json
```

You can inspect them with any S3 client:

```bash
aws s3 ls s3://my-bucket/pawn-agent-jobs/dead-letter/
aws s3 cp s3://my-bucket/pawn-agent-jobs/dead-letter/<id>.json -
```

After fixing the root cause, replay by re-publishing the message payload.

---

## systemd Service

Example unit file for running the listener as a daemon:

```ini
[Unit]
Description=Pawn Agent Queue Listener
After=network.target postgresql.service

[Service]
Type=simple
User=pawnai
WorkingDirectory=/opt/pawnai
ExecStart=/opt/pawnai/.venv/bin/pawn-agent listen --config /etc/pawnai/pawnai.yaml
Restart=on-failure
RestartSec=10
Environment=DATABASE_URL=postgresql+psycopg://pawnai:secret@localhost:5432/pawnai

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now pawn-agent-listener
sudo journalctl -u pawn-agent-listener -f
```

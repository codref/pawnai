# Pawn Diarize Queue Listener

Pawn Diarize can operate as a background worker that consumes jobs from a message queue instead of being driven by the command line.  The underlying queue is [pawn-queue](https://github.com/codref/pawn-queue) — a lightweight, S3-backed, at-least-once queue that works with any S3-compatible object store (AWS S3, MinIO, Hetzner Object Storage, Cloudflare R2, …).

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
  pawn-diarize listen  (consumer worker)
        │  reads payload, dispatches to core function
        ▼
  PostgreSQL / audio processing output
```

Each message lives as a small JSON object in S3.  When `pawn-diarize listen` picks up a message it writes a **lease** file to claim it; if processing succeeds the message is **acked** (deleted); if it fails the message is moved to a **dead-letter prefix** for manual inspection.  The lease is automatically refreshed in the background so long-running jobs (multi-minute CPU diarization) are never re-queued mid-flight.

---

## Installation

```bash
uv pip install pawn-queue
# aioboto3 is pulled in automatically by pawn-queue
```

---

## Configuration

The queue listener reuses the top-level `s3:` section already configured for
audio downloads — no duplicate credentials needed.  Add only a `diarize_queue:` section
with the queue-specific settings:

```yaml
# Shared S3 connection — used by both audio downloads and the queue.
s3:
  bucket: my-audio-bucket
  endpoint_url: https://s3.amazonaws.com   # or http://localhost:9000 for MinIO
  access_key: AKIAIOSFODNN7EXAMPLE
  secret_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
  region: us-east-1
  verify_ssl: true

diarize_queue:
  # Topic the listener subscribes to (created automatically if absent).
  topic: pawn-diarize-jobs

  # Unique name for this consumer instance.
  # Use a different name when running multiple parallel workers.
  consumer_name: pawn-diarize-listener

  # Dedicated bucket for queue message objects (separate from audio files).
  # Credentials and endpoint are taken from the s3: section above.
  bucket_name: pawn-diarize-queue

  polling:
    interval_seconds: 5
    max_messages_per_poll: 1        # process one job at a time
    # Must be longer than the slowest expected job to prevent re-delivery.
    visibility_timeout_seconds: 300
    lease_refresh_interval_seconds: 60
    jitter_max_ms: 200

  concurrency:
    # auto  → probe at startup (works everywhere)
    # conditional_write → faster, requires AWS S3 / R2
    # csprng_verify     → works on MinIO / Ceph / Hetzner
    strategy: auto
```

> **`visibility_timeout_seconds`** is the lease duration.  While a worker holds a lease no other consumer will touch the message.  The background refresher extends the lease every `lease_refresh_interval_seconds`.  If the worker crashes, the lease expires after `visibility_timeout_seconds` and another worker can reclaim the message.

---

## Running the listener

```bash
# Uses .pawn-diarize.yml in the current directory
pawn-diarize listen

# Explicit config file
pawn-diarize listen --config /etc/pawn-diarize/.pawn-diarize.yml

# Override topic and consumer name without changing the config file
pawn-diarize listen --topic my-custom-topic --consumer-name worker-02

# As a Python module
python -m pawn-diarize listen
```

Stop with **Ctrl-C** — the listener drains the current message before exiting.

---

## Message format

Every message sent to the queue must be a JSON object with a `command` key.  All other keys are the parameters for that command; any omitted parameter falls back to the same default that the CLI uses.

```json
{
  "command": "<command-name>",
  "<param1>": "<value1>",
  ...
}
```

### Supported commands

| `command` | Required fields | Notable optional fields |
|---|---|---|
| `transcribe-diarize` | `audio_paths` | `session`, `threshold`, `cross_file_threshold`, `device`, `backend`, `store_new`, `no_timestamps`, `chunk_duration`, `output`, `db_dsn` |
| `transcribe` | `audio_paths` | `session`, `device`, `backend`, `timestamps`, `chunk_duration`, `output`, `db_dsn` |
| `diarize` | `audio_paths` | `threshold`, `store_new`, `output`, `db_dsn` |
| `embed` | `audio_paths`, `speaker_id` | `db_dsn` |
| `analyze` | `session` or `input_path` | `mode` (`summary`\|`graph`), `model`, `output`, `db_dsn` |
| `sync-siyuan` | `session` or `all_sessions: true` | `notebook`, `token`, `url`, `path_template`, `daily_note`, `daily_path_template`, `db_dsn` |

---

## Sample messages

### transcribe-diarize (typical production job)

```json
{
  "command": "transcribe-diarize",
  "audio_paths": ["s3://260305173832/260305173832_*.flac"],
  "threshold": 0.2,
  "cross_file_threshold": 0.2,
  "session": "tom-20260305",
  "device": "cpu"
}
```

### transcribe-diarize (multiple explicit files, GPU, whisper backend)

```json
{
  "command": "transcribe-diarize",
  "audio_paths": [
    "s3://recordings/meeting_part1.flac",
    "s3://recordings/meeting_part2.flac"
  ],
  "session": "engineering-standup-20260306",
  "threshold": 0.65,
  "cross_file_threshold": 0.80,
  "device": "cuda",
  "backend": "whisper",
  "output": "/output/standup-20260306.txt"
}
```

### transcribe only

```json
{
  "command": "transcribe",
  "audio_paths": ["s3://mybucket/lecture.wav"],
  "device": "cpu",
  "backend": "nemo",
  "timestamps": false,
  "output": "/output/lecture.txt"
}
```

### diarize only

```json
{
  "command": "diarize",
  "audio_paths": ["s3://mybucket/interview.flac"],
  "threshold": 0.75,
  "store_new": true
}
```

### embed — register a known speaker

```json
{
  "command": "embed",
  "audio_paths": ["s3://mybucket/tom_reference.flac"],
  "speaker_id": "tom"
}
```

### sync-siyuan — push analysis to SiYuan Note

```json
{
  "command": "sync-siyuan",
  "session": "tom-20260305"
}
```

Override SiYuan connection details when needed (otherwise the `siyuan:` section in `.pawn-diarize.yml` is used):

```json
{
  "command": "sync-siyuan",
  "session": "tom-20260305",
  "notebook": "20210817205410-2kvfpfn",
  "token": "your_api_token",
  "daily_note": true
}
```

Sync every session that has a completed analysis:

```json
{
  "command": "sync-siyuan",
  "all_sessions": true
}
```

### analyze a recorded session

```json
{
  "command": "analyze",
  "session": "tom-20260305",
  "mode": "summary",
  "model": "gpt-4o",
  "output": "/output/tom-20260305-analysis.txt"
}
```

---

## Publishing messages from a producer

```python
import asyncio
from pawn_queue import PawnQueueBuilder

async def main():
    async with await (
        PawnQueueBuilder()
        .s3(
            endpoint_url="http://localhost:9000",
            bucket_name="pawn-diarize-queue",
            access_key="minioadmin",
            secret_key="minioadmin",
        )
        .build()
    ) as pq:
        await pq.create_topic("pawn-diarize-jobs")   # idempotent
        producer = await pq.register_producer("my-script")

        message_id = await producer.publish("pawn-diarize-jobs", {
            "command": "transcribe-diarize",
            "audio_paths": ["s3://260305173832/260305173832_*.flac"],
            "threshold": 0.2,
            "cross_file_threshold": 0.2,
            "session": "tom-20260305",
            "device": "cpu",
        })
        print(f"Published: {message_id}")

asyncio.run(main())
```

---

## Dead-letter queue

Messages that fail (exception in processing, missing `command` key, unsupported command) are automatically moved to the dead-letter prefix inside the same S3 bucket:

```
pawn-diarize-queue/<topic>/dead-letter/<message-id>.json
```

You can inspect them with `pawn-diarize s3-ls` or any S3 browser and re-publish them after investigating the root cause.

### Replaying dead-letter messages

```python
"""Replay all messages from the dead-letter prefix of a topic.

Usage:
    python replay_dead_letter.py
"""
import asyncio
import json
import yaml
from pawn_queue import PawnQueueBuilder
from pawn_queue.client import S3Client

CONFIG_FILE = ".pawn-diarize.yml"
TOPIC = "audio-chunks"   # topic whose dead-letter queue to drain


async def main() -> None:
    cfg = yaml.safe_load(open(CONFIG_FILE))
    s3_cfg = cfg["s3"]
    q_cfg = cfg["queue"]

    builder = (
        PawnQueueBuilder()
        .s3(
            endpoint_url=s3_cfg["endpoint_url"],
            bucket_name=q_cfg["bucket_name"],
            access_key=s3_cfg["access_key"],
            secret_key=s3_cfg["secret_key"],
            region_name=s3_cfg.get("region", "us-east-1"),
            use_ssl=s3_cfg.get("verify_ssl", False),
        )
    )

    async with await builder.build() as pq:
        producer = await pq.register_producer("dead-letter-replayer")
        client = pq._client  # internal S3 client

        prefix = f"{TOPIC}/dead-letter/"
        objects = await client.list_objects(prefix=prefix)

        if not objects:
            print(f"No dead-letter messages found under {prefix}")
            return

        for obj in objects:
            key = obj["Key"]
            raw = await client.get_object(key)
            msg = json.loads(raw)
            payload = msg.get("payload", {})
            print(f"Replaying {key}  command={payload.get('command')}")
            new_id = await producer.publish(TOPIC, payload)
            print(f"  → re-published as {new_id}")
            await client.delete_object(key)   # remove from dead-letter

        print("Done.")


asyncio.run(main())
```

---

## Running as a system service (systemd)

```ini
[Unit]
Description=Pawn Diarize queue listener
After=network.target

[Service]
User=pawn-diarize
WorkingDirectory=/opt/pawn-diarize
ExecStart=pawn-diarize listen --config /etc/pawn-diarize/.pawn-diarize.yml
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

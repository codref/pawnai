# PawnAI

A Python monolith for speaker diarization, audio transcription, and LLM-powered conversation analysis. Provides two CLI applications: **pawn-diarize** for audio processing and embedding management, and **pawn-agent** for conversational analysis of recorded sessions.

## Features

- **Speaker Diarization**: Identify and separate multiple speakers using pyannote.audio
- **Audio Transcription**: Convert speech to text via NeMo Parakeet or faster-whisper backends
- **Combined Transcription & Diarization**: Transcribe and label speakers in one command
- **Multi-file Processing**: Process multiple files as one conversation with cross-file speaker alignment
- **Session Accumulation**: Process long conversations in parts across multiple invocations
- **Speaker Embeddings**: Extract and store 512-dim speaker vectors in PostgreSQL with pgvector
- **Conversation Analysis**: Generate titles, summaries, key topics, sentiment, and per-speaker highlights
- **Knowledge Graph Extraction**: Extract semantic triples (subject, relation, object) from transcripts
- **RAG Index**: Embed transcripts and SiYuan notes for semantic similarity search
- **S3 Storage Management**: List, filter, and delete objects in S3-compatible storage
- **SiYuan Notes Integration**: Push analysis documents back to a SiYuan Notes instance
- **Background Queue Worker**: S3-backed job queue with lease-based concurrency
- **GPU Support**: Accelerated processing on CUDA-enabled devices

## Installation

### From Source

```bash
git clone <repository-url>
cd parakeet
pip install -e ".[dev]"
```

### System Requirements

- Python 3.10+
- PostgreSQL with the `pgvector` extension
- CUDA 11.0+ (optional, for GPU acceleration)
- 8 GB+ RAM recommended for ML models

### Database Setup

```bash
# Apply all migrations
alembic upgrade head
```

## Configuration

All settings live in `pawnai.yaml` (auto-discovered in the current directory, or passed via `--config`).
Precedence: **CLI flags → `pawnai.yaml` → environment variables → defaults**.

```yaml
models:
  hf_token: hf_your_token_here        # HuggingFace token for gated models
  # asr_model: nvidia/parakeet-tdt-0.6b-v3
  # diarization_model: pyannote/speaker-diarization-3.1

db_dsn: postgresql+psycopg://user:pass@localhost:5433/pawnai

paths:
  db_path: speakers_db
  audio_dir: audio/

device: auto    # auto | cuda | cpu

s3:
  bucket: my-audio-bucket
  endpoint_url: https://s3.amazonaws.com
  access_key: AKIAIOSFODNN7EXAMPLE
  secret_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
  region: us-east-1
  prefix: ""
  verify_ssl: true
  path_style: true

siyuan:
  url: http://localhost:6806
  token: your-siyuan-token
  notebook: Meetings
  path_template: "/conversations/{date}/{session_id}/{title}"  # child page per analysis

agent:
  name: Bob
  anima: anima.md                     # optional personality/system-prompt file
  openai:
    model: qwen3:8b
    base_url: http://localhost:11434/v1
    api_key: ollama
  copilot:
    model: gpt-4o

rag:
  embed_model: Qwen/Qwen3-Embedding-0.6B
  embed_dim: 1024
  embed_device: cpu

queue:
  topic: pawn-jobs
  consumer_name: worker-1
  bucket_name: my-audio-bucket
```

---

## pawn-diarize

### Quick Start

```bash
# Check system status
pawn-diarize status

# Transcribe and diarize a meeting
pawn-diarize transcribe-diarize meeting.wav -o transcript.txt

# Label a speaker for future recognition
pawn-diarize label -f audio.wav -s SPEAKER_00 -n "Alice Smith"
```

### Commands Reference

#### `diarize`

Perform speaker diarization on one or more audio files. Automatically recognizes known speakers from the database and replaces generic labels with names.

```bash
pawn-diarize diarize <audio_path>... [OPTIONS]

Options:
  --output, -o TEXT        Output file (.txt or .json)
  --config TEXT            Path to pawnai.yaml
  --db-dsn TEXT            PostgreSQL DSN
  --threshold, -t FLOAT    Similarity threshold for speaker matching (default: 0.7)
  --store-new/--no-store   Store embeddings for unknown speakers (default: enabled)
```

```bash
pawn-diarize diarize meeting.wav -o speakers.json
pawn-diarize diarize part1.wav part2.wav part3.wav -o full_diarization.json
pawn-diarize diarize meeting.wav --no-store
```

#### `transcribe`

Transcribe audio to text. Supports two backends: `nemo` (Nvidia Parakeet) and `whisper` (faster-whisper).

```bash
pawn-diarize transcribe <audio_path> [OPTIONS]

Options:
  --output, -o TEXT          Output file (.txt or .json)
  --config TEXT              Path to pawnai.yaml
  --db-dsn TEXT              PostgreSQL DSN
  --timestamps/--no-timestamps  Include word-level timestamps (default: enabled)
  --device, -d TEXT          cuda or cpu (default: auto)
  --chunk-duration, -c FLOAT Split audio into N-second chunks (helps avoid OOM)
  --backend TEXT             nemo or whisper (default: nemo)
```

```bash
pawn-diarize transcribe speech.wav -o transcript.txt
pawn-diarize transcribe large.mp3 --backend whisper --device cpu -c 300
```

#### `transcribe-diarize`

Combined transcription + diarization with speaker labels in a single pass. When multiple files are given they are treated as ordered chunks of the same conversation, with speaker labels aligned across files.

```bash
pawn-diarize transcribe-diarize <audio_path>... [OPTIONS]

Options:
  --output, -o TEXT             Output file (.txt or .json)
  --config TEXT                 Path to pawnai.yaml
  --session, -s TEXT            Session JSON file for accumulative processing
  --db-dsn TEXT                 PostgreSQL DSN
  --threshold, -t FLOAT         Speaker matching threshold (default: 0.7)
  --store-new/--no-store        Store unknown speaker embeddings (default: enabled)
  --device, -d TEXT             cuda or cpu
  --chunk-duration, -c FLOAT    Split audio into chunks
  --cross-threshold, -x FLOAT   Cross-file speaker alignment threshold (default: 0.85)
  --no-timestamps               Hide timestamps in output
  --backend TEXT                nemo or whisper (default: nemo)
```

```bash
# Single file
pawn-diarize transcribe-diarize meeting.wav -o transcript.txt

# Multiple files (ordered chunks of same conversation)
pawn-diarize transcribe-diarize part1.wav part2.wav part3.wav -o full.txt

# Accumulate across invocations
pawn-diarize transcribe-diarize part1.wav --session conv.json
pawn-diarize transcribe-diarize part2.wav --session conv.json
pawn-diarize transcribe-diarize part3.wav --session conv.json -o final.txt
```

#### `embed`

Extract and store speaker embeddings (512-dim) in PostgreSQL for future recognition.

```bash
pawn-diarize embed <audio_path> --speaker-id SPEAKER_ID [OPTIONS]

Options:
  --speaker-id, -s TEXT   Unique speaker identifier (required)
  --config TEXT           Path to pawnai.yaml
  --db-dsn TEXT           PostgreSQL DSN
```

```bash
pawn-diarize embed person.wav --speaker-id alice_smith
```

#### `search`

Find speakers with similar voice characteristics via cosine similarity on stored embeddings.

```bash
pawn-diarize search <speaker_id> [OPTIONS]

Options:
  --limit INT    Maximum results (default: 5)
  --db-dsn TEXT  PostgreSQL DSN
```

#### `label`

Assign a human-readable name to a speaker for automatic recognition in future diarizations.

```bash
pawn-diarize label [OPTIONS]

Options:
  --file, -f TEXT      Audio file containing the speaker
  --speaker, -s TEXT   Speaker label (e.g. SPEAKER_00)
  --name, -n TEXT      Human-readable name
  --session TEXT       Session ID to scope the labeling
  --list, -l           List all speaker name mappings
  --config TEXT        Path to pawnai.yaml
  --db-dsn TEXT        PostgreSQL DSN
```

```bash
pawn-diarize label -f audio.wav -s SPEAKER_00 -n "Alice Smith"
pawn-diarize label --list
```

**Typical workflow:**
1. `transcribe-diarize` → transcript with SPEAKER_00, SPEAKER_01, …
2. `label` → assign names to those labels
3. Future `transcribe-diarize` runs → names appear automatically

#### `session-relabel`

Bulk-rename a mis-identified speaker across an entire session.

```bash
pawn-diarize session-relabel --session SESSION_ID --from OLD_NAME --to NEW_NAME [OPTIONS]

Options:
  --yes, -y    Skip confirmation prompt
  --db-dsn TEXT
  --config TEXT
```

#### `session-info`

Show speakers and their embedding source files for a session.

```bash
pawn-diarize session-info <session_id> [OPTIONS]
```

#### `sessions`

List all sessions or inspect a specific one.

```bash
pawn-diarize sessions [OPTIONS]

Options:
  --session TEXT    Inspect a specific session
  --head INT        Show first N segments
  --tail INT        Show last N segments
  --output TEXT     Save output to file
  --config TEXT     Path to pawnai.yaml
```

#### `sync-siyuan`

Push a session's analysis to a SiYuan Notes instance as a new document.

```bash
pawn-diarize sync-siyuan [OPTIONS]

Options:
  --session TEXT          Session ID to sync
  --all                   Sync all sessions with stored analysis
  --notebook TEXT         Target SiYuan notebook
  --token TEXT            SiYuan API token
  --url TEXT              SiYuan instance URL
  --path-template TEXT    Document path template
  --daily-note            Insert into today's daily note
  --db-dsn TEXT
  --config TEXT
```

#### `s3 ls`

List objects in the configured S3 bucket with POSIX-style output (timestamp and size always shown).

```bash
pawn-diarize s3 ls [PREFIX] [OPTIONS]

Options:
  --config TEXT           Path to pawnai.yaml
  --recursive, -r         Flat listing (no directory grouping)
  --time, -t              Sort by modification time, newest first
  --reverse               Reverse sort order
  --older-than INTEGER    Show only objects older than N days
```

```bash
pawn-diarize s3 ls
pawn-diarize s3 ls recordings/ -t
pawn-diarize s3 ls recordings/ -r -t --reverse
pawn-diarize s3 ls --older-than 30
```

Output format:
```
2024-03-15 14:32:10    45.2 MB  recordings/2024/meeting.wav
                PRE             recordings/archive/
```

#### `s3 rm`

Delete objects from S3 by key or by age. Always prefer `--dry-run` first.

```bash
pawn-diarize s3 rm [PATH] [OPTIONS]

Options:
  --config TEXT           Path to pawnai.yaml
  --older-than INTEGER    Delete all objects older than N days
  --prefix TEXT           Limit --older-than to this key prefix
  --dry-run, -n           Show what would be deleted without making changes
  --yes, -y               Skip confirmation prompt for bulk deletions
```

```bash
# Delete a specific file
pawn-diarize s3 rm recordings/2024/old.wav

# Preview bulk deletion
pawn-diarize s3 rm --older-than 30 --dry-run

# Delete all objects older than 90 days under a prefix
pawn-diarize s3 rm --older-than 90 --prefix recordings/ --yes
```

#### `listen`

Background worker that polls a pawn-queue topic and executes jobs (transcribe-diarize, transcribe, diarize, embed, analyze, sync-siyuan).

```bash
pawn-diarize listen [OPTIONS]

Options:
  --config TEXT          Path to pawnai.yaml
  --topic TEXT           Queue topic name
  --consumer-name TEXT   Worker identity for lease-based concurrency
```

#### `status`

Show system status: device, CUDA availability, GPU info, configured models.

```bash
pawn-diarize status [--config TEXT]
```

### S3 Audio Paths

All audio processing commands accept `s3://` URIs directly — files are downloaded to a temp location, processed, then cleaned up automatically.

```bash
pawn-diarize transcribe s3://my-bucket/recordings/meeting.wav -o transcript.txt
pawn-diarize transcribe-diarize s3://my-bucket/recordings/*.wav
```

---

## pawn-agent

LLM-powered conversational agent for analyzing pawn-diarize sessions. Backed by PydanticAI and supports multiple providers (OpenAI-compatible, GitHub Copilot, Anthropic, Google, Groq, Mistral, and others).

Tools are **auto-discovered** from `pawn_agent/tools/` — any module that exports `NAME`, `DESCRIPTION`, and `build(cfg)` is registered automatically.

### Commands Reference

#### `chat`

Start an interactive multi-turn conversation session.

```bash
pawn-agent chat [OPTIONS]

Options:
  --config TEXT    Path to pawnai.yaml
  --model TEXT     Override the configured LLM model (e.g. openai:gpt-4o)
  --session TEXT   Session ID to load and continue a stored conversation
  --db-dsn TEXT    PostgreSQL DSN override
  --langgraph      Run the LangGraph evaluation chat path
  --langgraph-trace-state
                   Attach the full LangGraph state as JSON to Phoenix spans
```

**Terminal features:**

- **Multi-line paste**: pasted text is buffered as a single message (no spurious multi-turn splits)
- **Context size indicator**: the prompt shows serialized history size and a token estimate after every turn:
  ```
  You [18.3 KB · ~4k tok]:
  ```
  On session load the same figures are printed next to the resume banner:
  ```
  Resuming session 'warren-20260325' (42 stored message(s) · 18.3 KB · ~4k tok)
  ```
- **Slash commands:**

  | Command | Effect |
  |---------|--------|
  | `/exit` or `/quit` | End the session |
  | `/reset` | Clear in-memory history and wipe stored turns from the database |

  `Ctrl-D` and `Ctrl-C` also exit cleanly.

**Session persistence**: when `--session` is given, every turn is appended to `agent_session_turns` in PostgreSQL. Resuming the same session replays the full stored history so the model retains context across invocations. `/reset` deletes all stored turns so the next message starts fresh.

**LangGraph evaluation mode**: `pawn-agent chat --langgraph` runs the explicit orchestration path used to evaluate routed multi-step chat behavior. It still has a minimal REPL surface, but internally it uses structured state, fast/deep response routing, session-aware tool nodes, artifact carry-forward for later save steps, and optional Phoenix tracing via `--langgraph-trace-state`. It does not yet support `--session`, DB-backed chat history, or the richer slash commands from the default chat path.

```bash
pawn-agent chat --langgraph
```

#### `run`

Execute a single prompt and exit (useful for scripting).

```bash
pawn-agent run "<prompt>" [OPTIONS]

Options:
  --config TEXT    Path to pawnai.yaml
  --model TEXT     Override the configured LLM model
  --session TEXT   Session hint
```

```bash
pawn-agent run "Summarize session abc123 and save to SiYuan"
```

#### `tools`

List all registered agent tools with their descriptions.

```bash
pawn-agent tools [--config TEXT]
```

#### `models`

List available Copilot models with billing and reasoning effort details.

```bash
pawn-agent models [--config TEXT]
```

### Available Tools

| Tool | Description |
|------|-------------|
| `query_conversation` | Fetch the full transcript for a session from the database |
| `analyze_summary` | Run structured analysis (title, summary, topics, sentiment, tags) and persist to DB |
| `get_analysis` | Retrieve the most recent stored analysis for a session |
| `extract_graph` | Extract knowledge-graph triples from a transcript and persist to DB |
| `vectorize` | Embed session transcripts or SiYuan pages into the RAG index |
| `search_knowledge` | Semantic similarity search over transcript chunks and SiYuan notes |
| `save_to_siyuan` | Save Markdown content to SiYuan Notes as a child page under the session node; title inferred from the first `# Heading` |
| `fetch_siyuan_page` | Fetch the text content of a SiYuan page by path |
| `rag_stats` | Show a summary of the RAG vector index (sources and chunk counts) |

---

## Project Structure

```
parakeet/
├── pawn_diarize/
│   ├── __main__.py           # Entry point → pawn-diarize
│   ├── cli/
│   │   ├── commands.py       # All CLI command definitions
│   │   └── utils.py          # Console, helpers
│   ├── core/
│   │   ├── config.py         # pawnai.yaml loader
│   │   ├── diarization.py    # pyannote.audio + DBSCAN clustering
│   │   ├── transcription.py  # NeMo Parakeet / faster-whisper backends
│   │   ├── embeddings.py     # Speaker embeddings (PostgreSQL + pgvector)
│   │   ├── combined.py       # Multi-file transcription+diarization
│   │   ├── database.py       # SQLAlchemy ORM (embeddings, segments, analysis, RAG, graph)
│   │   ├── analysis.py       # Session analysis via Copilot
│   │   ├── s3.py             # S3/MinIO client and transparent download helpers
│   │   ├── siyuan.py         # SiYuan Notes API client
│   │   └── queue_listener.py # Background job worker
│   └── utils/
├── pawn_agent/
│   ├── __main__.py           # Entry point → pawn-agent
│   ├── cli/commands.py       # run, chat, tools, models
│   ├── core/
│   │   ├── agent.py          # ConversationAgent (Copilot SDK)
│   │   └── pydantic_agent.py # PydanticAI multi-provider agent
│   ├── tools/                # Auto-discovered tool modules
│   └── utils/
│       ├── config.py         # AgentConfig loader
│       ├── db.py             # DB session factory + RAG tables
│       ├── transcript.py     # Fetch/format session transcripts
│       ├── siyuan.py         # SiYuan helpers
│       ├── analysis.py       # Analysis runner
│       └── vectorize.py      # RAG vectorization
├── migrations/               # Alembic schema versions
├── pawnai.yaml               # Project configuration
└── alembic.ini
```

## Technologies

| Category | Technology |
|----------|-----------|
| Speaker diarization | pyannote.audio 4.0+ |
| Transcription | NeMo Parakeet (nvidia/parakeet-tdt-0.6b-v3), faster-whisper |
| Speaker embeddings | PostgreSQL + pgvector (512-dim cosine similarity) |
| RAG embeddings | sentence-transformers / Qwen3-Embedding-0.6B (1024-dim) |
| Database ORM | SQLAlchemy 2.0 + Alembic migrations |
| LLM agent | PydanticAI (multi-provider) + GitHub Copilot SDK |
| S3 storage | boto3 / aioboto3 |
| Job queue | pawn-queue (S3-backed) |
| Notes integration | SiYuan Notes API |
| CLI | Typer + Rich |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Format
black pawn_diarize pawn_agent tests
isort pawn_diarize pawn_agent tests

# Lint
flake8 pawn_diarize pawn_agent tests
mypy pawn_diarize pawn_agent

# Test
pytest
pytest tests/test_cli.py::test_status
pytest --cov=pawn_diarize --cov-report=html
```

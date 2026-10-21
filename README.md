# PawnAI

A Python monolith for speaker diarization, audio transcription, and an LLM coworker that lives in your Obsidian vault. Three console scripts from `pyproject.toml`:

- **pawn-diarize** — transcription, diarization, speaker gallery, S3/queue workers
- **pawn-agent** — sallm ReAct agent (skills + CliTools) for chat and domain tools
- **pawn-server** — FastAPI OpenAI-compatible API, queue listener, durable scheduler, vault watcher, optional Matrix bot

Companion UIs (separate packages in this repo):

- **Obsidian plugin** — chat pane, Inbox, Sessions, background jobs (`obsidian-plugin/pawn/`)
- **Browser capture** — MV3 side panel for text/image snippets (`browser-extension/pawn/`)
- **Thunderbird capture** — MailExtension companion tray (`thunderbird-extension/pawn/`)

## Features

- **Speaker diarization** — pyannote (default) or Nemotron/Sortformer backends
- **Transcription** — NeMo Parakeet or faster-whisper
- **Combined transcribe + diarize** — multi-file chunks, session accumulation, S3 URIs
- **Curated Speakers gallery** — manual enrollments / reidentify (runtime diarization never auto-enrolls)
- **People bios** — vault notes under `People/` linked from transcript Speakers tables
- **Conversation analysis** — titles, summaries, topics, sentiment; optional vault save under `Pawn/Analyses/`
- **Knowledge / RAG index** — transcript and vault-note embeddings for related search
- **Obsidian vault** — S3 Markdown vault (Sync Engine); transcripts, tasks, captures, coworker notes
- **Coworker loop** — after sessions, extract/score items against `Goals.md`, Inbox triage, morning/weekly crons
- **Background queues** — S3-backed pawn-queue with lease-based concurrency
- **Durable agent scheduler** — PostgreSQL schedules (`once` / `interval` / `cron`)
- **Capture extensions** — browser and Thunderbird → `/v1/captures` (notes, session Annotations, or research inbox)
- **GPU support** — CUDA when available

## Installation

Requires **Python 3.12+**, PostgreSQL with **pgvector**, and (for local sallm embeddings) Ollama.

```bash
git clone <repository-url>
cd parakeet

# Prefer uv (matches uv.lock)
uv sync --extra dev

# Or pip
pip install -e ".[dev]"

# Matrix bot extras
uv sync --extra matrix
```

`sallm-agent` is an editable path dependency (`../sallm-agent` via `[tool.uv.sources]`). Alternative: `uv add git+https://github.com/codref/sallm-agent.git`.

### Database

```bash
docker compose -f docker/docker-compose.yml up -d postgres
alembic upgrade head
```

Default DSN uses host port **5433**.

### Local embeddings

```bash
ollama pull qwen3-embedding:0.6b
```

## Configuration

Settings live in `pawnai.yaml` (auto-discovered in the current directory, or `--config`). Precedence: **CLI flags → YAML → environment variables → defaults**. Env vars use `PAWN_` with `__` nesting (e.g. `PAWN_DB_DSN`, `PAWN_MODELS__HF_TOKEN`). Copy `pawnai.example.yaml` for a full commented template — do not commit a real `pawnai.yaml` (it may contain secrets).

Minimal sketch of current shape:

```yaml
db_dsn: postgresql+psycopg://postgres:postgres@localhost:5433/pawnai

models:
  hf_token: ""
  transcription_backend: nemo
  diarization_backend: pyannote
  diarization_model: pyannote/speaker-diarization-community-1
  embedding_model: nvidia/speakerverification_en_titanet_large

device:
  type: auto    # auto | cuda | cpu

s3:
  bucket: my-audio-bucket
  endpoint_url: https://s3.amazonaws.com
  access_key: ""
  secret_key: ""
  path_style: true

# Obsidian Sync Engine bucket (separate credentials from audio/queue s3:)
vault:
  s3:
    bucket: my-obsidian-bucket
    endpoint_url: https://fsn1.your-objectstorage.com
    access_key: ""
    secret_key: ""
  agent_root: Pawn
  auto_push_transcript: false

agent:
  name: Bob
  anima: anima.md
  # Catalog id = provider@model (interactive + background default)
  default: ollama@gemma4:4b
  providers:
    ollama:
      base_url: http://localhost:11434/v1
      api_key: ollama
      models:
        - model: gemma4:4b
          profile: large.yaml
          # vision: true   # Matrix / Obsidian image attachments
  # Legacy agent.openai block still works if providers is omitted

agent_queue:
  topic: pawn-agent-jobs
  consumer_name: pawn-agent-listener
  bucket_name: my-audio-bucket

agent_scheduler:
  enabled: true
  poll_interval_seconds: 30

coworker:
  enabled: false
  goals_path: Goals.md
```

Chat models are a catalog under `agent.providers`. Selectable ids look like `ollama@gemma4:4b`. `/model <id>` changes the background default at runtime; autonomous jobs (scheduler, coworker, Matrix, vault watcher) use that default when no model is passed.

---

## pawn-diarize

### Quick start

```bash
pawn-diarize status
pawn-diarize transcribe-diarize meeting.wav -o transcript.txt
pawn-diarize speakers create Alice
pawn-diarize speakers enroll -s Alice --audio alice-hello.wav
```

### Main commands

| Command | Purpose |
|---------|---------|
| `diarize` | Speaker labels only |
| `transcribe` | ASR (`nemo` or `whisper`) |
| `transcribe-diarize` | Combined pass; multi-file / `--session` accumulation; optional annotations/screenshots |
| `embed` / `search` / `label` | Legacy embedding helpers and display-name labels |
| `speakers …` | Curated gallery: `list`, `show`, `create`, `enroll`, `update`, … — see [docs/SPEAKERS.md](docs/SPEAKERS.md) |
| `session-relabel` | Rename a speaker across one session |
| `reidentify` | Re-match session speakers against the gallery |
| `rediarize` / `retranscribe` | Repair labels only, or full ASR+diarize overwrite (`--confirm`) |
| `push-vault` | Write/update vault Speakers+Transcript+Screenshots note |
| `sessions` / `session-info` | Inspect stored sessions |
| `s3 ls` / `s3 rm` | Bucket listing and age-based cleanup |
| `listen` | Diarize queue worker |
| `status` | Device / CUDA / model summary |

Audio commands accept `s3://` URIs. Opt-in auto vault push: `vault.auto_push_transcript: true`.

```bash
pawn-diarize transcribe-diarize meeting.wav -o transcript.txt
pawn-diarize transcribe-diarize part1.wav part2.wav --session conv.json
pawn-diarize session-relabel --session my-session --from SPEAKER_00 --to Davide --yes
pawn-diarize push-vault --session myconv
pawn-diarize rediarize --session X --confirm
pawn-diarize retranscribe --session X --confirm
```

Queue / chain-agent details: [docs/QUEUE_LISTENER.md](docs/QUEUE_LISTENER.md). Vault diary notes: [docs/OBSIDIAN_AGENT.md](docs/OBSIDIAN_AGENT.md).

---

## pawn-agent

Interactive and scripted access to the embedded **sallm** ReAct agent (skills + CliTools). Conversation memory lives under `agent.sallm.state_dir` (SQLite + Lance). PostgreSQL still holds transcripts, analyses, `agent_runs`, and schedules.

```bash
pawn-agent chat
pawn-agent chat --model ollama@gemma4:4b
pawn-agent tools
pawn-agent models
```

**Slash commands (chat):** `/exit`, `/quit`, `/stats`, `/reset`, `/model <id>`, `/model reset`, `/idea …`, `/goal` / `/park` / `/goal apply`.

### CliTools

Registered in `pawn_agent/core/sallm_tools.py` (domain logic in `pawn_agent/tools/*_impl` + `tools/cli/`):

| CliTool | Description |
|---------|-------------|
| `sessions_list` / `session_transcript` / `session_analyze` | List sessions, fetch transcript, structured analysis (`--save` → `Pawn/Analyses/`) |
| `session_screenshots` | List/summarize session screenshots |
| `session_delete` / `session_relabel` | Wipe a session (`--confirm`) / rename a speaker |
| `speakers_*` / `speaker_enroll` / `session_reidentify` | Speakers gallery |
| `people_show` / `people_ensure` / `people_append` | People bios |
| `note_read` / `note_search` / `note_write` / `note_append` | Vault notes (write guards under `Pawn/`) |
| `capture_update` / `task_update` | Research captures and vault jobs |
| `knowledge_search` | Related / RAG search |
| `schedule_propose` / `queue_push` / `goal_propose` | Schedules (proposals only), notifications, goal drafts |
| `tasknotes_*` | TaskNotes-compatible notes under `{agent_root}/TaskNotes/` |

Skills (modes): `converse`, `sessions`, `notes`, `scheduling`, `ops`, `vault_tasks`, `tasknotes`, `coworker`, `research_capture`. Full tool authoring notes: [docs/TOOLS.md](docs/TOOLS.md), [docs/AGENT_TOOL_AUTHORING.md](docs/AGENT_TOOL_AUTHORING.md).

---

## pawn-server

HTTP API, agent queue listener, scheduler, vault watcher, and optional Matrix bot.

```bash
pawn-server serve
pawn-server serve --no-queue --disable-scheduler --no-matrix --no-vault-watcher
pawn-server serve --scheduler-only
pawn-server serve --matrix-only
pawn-server serve --vault-watcher-only
```

Useful flags: `--host` / `--port`, `--ssl-certfile` / `--ssl-keyfile`, `--no-queue`, `--disable-scheduler`, `--no-matrix`, `--no-vault-watcher`, and the `*-only` single-worker modes.

Chat completions always go through the embedded sallm agent. Surfaces:

| Surface | Docs |
|---------|------|
| OpenAI-compatible `/v1/chat/completions` | [docs/OPENAI_API.md](docs/OPENAI_API.md) |
| Obsidian plugin chat + jobs (`/v1/pawn/chat`, `/v1/jobs`) | [docs/OBSIDIAN_AGENT.md](docs/OBSIDIAN_AGENT.md) |
| Browser / Thunderbird captures | [docs/BROWSER_CAPTURE.md](docs/BROWSER_CAPTURE.md), [docs/THUNDERBIRD_CAPTURE.md](docs/THUNDERBIRD_CAPTURE.md) |
| Matrix bot | [docs/MATRIX_BOT.md](docs/MATRIX_BOT.md) (needs `uv sync --extra matrix`) |
| Coworker / Inbox | [docs/COWORKER.md](docs/COWORKER.md) |
| People bios | [docs/PEOPLE.md](docs/PEOPLE.md) |
| Scheduler | [docs/AGENT_SCHEDULER.md](docs/AGENT_SCHEDULER.md) |

### Admin CLIs

```bash
# Schedules (agent tools only propose; you approve here)
pawn-server schedules list
pawn-server schedules proposals
pawn-server schedules approve <proposal-id>

# Queues (agent_queue, diarize_queue, queue_producers)
pawn-server queue stats
pawn-server queue pause --name agent
pawn-server queue resume --all

# API IP blacklist
pawn-server blacklist list

# Coworker one-offs
pawn-server coworker process --session ID
pawn-server coworker people-refresh --session ID
pawn-server coworker status
```

When exposing the API, set `api.token`, turn `api.enable_docs` off, and prefer a reverse proxy (or `make ssl-cert` + `api.ssl_*` for direct TLS). See `deploy/nginx-pawn-location.conf` and the API security notes in [docs/OPENAI_API.md](docs/OPENAI_API.md).

---

## Obsidian plugin & capture extensions

```bash
make obsidian-plugin          # build
make obsidian-plugin-dist     # pawn.zip → <vault>/.obsidian/plugins/
make browser-extension-dist   # Chrome/Firefox zips
make thunderbird-extension-dist
```

The Obsidian plugin is MIT and written from scratch — do not copy AGPL `obsidian-copilot` code into it. Plugin chat uses `POST /v1/pawn/chat` (SSE); background work uses `POST /v1/jobs`. Stock Copilot can still target `/v1/chat/completions`.

---

## Project structure

```
parakeet/
├── pawn_core/              # Shared config, DB base, transcription, TTS, vault store
├── pawn_diarize/           # CLI + audio business logic
├── pawn_agent/             # CLI, sallm harness, scheduler, CliTools, coworker/people
├── pawn_server/            # FastAPI, queue listener, Matrix, vault watcher, jobs/captures
├── obsidian-plugin/pawn/   # Obsidian desktop/mobile plugin
├── browser-extension/pawn/ # MV3 capture side panel
├── thunderbird-extension/pawn/
├── migrations/             # Alembic
├── docs/                   # Feature docs (see index below)
├── docker/                 # Postgres compose, litellm image
├── pawnai.example.yaml
└── pyproject.toml
```

Transcription engine lives in `pawn_core/transcription.py`; `pawn_diarize/core/transcription.py` is a compatibility re-export.

## Technologies

| Category | Technology |
|----------|------------|
| Speaker diarization | pyannote.audio 4.x (optional Nemotron/Sortformer) |
| Transcription | NeMo Parakeet, faster-whisper |
| Speaker embeddings | PostgreSQL + pgvector; TitaNet gallery enrollments |
| RAG / related | sentence-transformers / Qwen3-Embedding (and Ollama for sallm) |
| Database | SQLAlchemy 2 + Alembic; PostgreSQL + pgvector |
| LLM agent | sallm ReAct + LiteLLM providers catalog |
| HTTP | FastAPI + uvicorn |
| S3 / queue | boto3 / aioboto3, pawn-queue |
| Scheduler | PostgreSQL + croniter |
| Notes | Obsidian Sync Engine + plugin |
| CLI | Typer + Rich |

## Development

```bash
uv sync --extra dev

black pawn_diarize pawn_agent tests
isort pawn_diarize pawn_agent tests
flake8 pawn_diarize pawn_agent tests
mypy pawn_diarize pawn_agent

pytest --no-cov          # faster
pytest                   # default includes --cov=pawn_diarize
```

Mypy checks `pawn_diarize` and `pawn_agent` unless paths are added. Agent operating notes for coding agents: [AGENTS.md](AGENTS.md).

## Documentation

| Doc | Topic |
|-----|--------|
| [docs/OBSIDIAN_AGENT.md](docs/OBSIDIAN_AGENT.md) | Vault, plugin chat/jobs, task loop |
| [docs/COWORKER.md](docs/COWORKER.md) | Items, Goals, Inbox, crons |
| [docs/SPEAKERS.md](docs/SPEAKERS.md) | Speakers gallery |
| [docs/PEOPLE.md](docs/PEOPLE.md) | People bios |
| [docs/BROWSER_CAPTURE.md](docs/BROWSER_CAPTURE.md) | Browser extension |
| [docs/THUNDERBIRD_CAPTURE.md](docs/THUNDERBIRD_CAPTURE.md) | Thunderbird add-on |
| [docs/MATRIX_BOT.md](docs/MATRIX_BOT.md) | Inbound Matrix chatbot |
| [docs/OPENAI_API.md](docs/OPENAI_API.md) | HTTP API / Copilot compat |
| [docs/AGENT_SCHEDULER.md](docs/AGENT_SCHEDULER.md) | Durable schedules |
| [docs/TOOLS.md](docs/TOOLS.md) | Agent tools |
| [docs/TTS.md](docs/TTS.md) | Speech endpoint |
| [docs/COMBINED_FEATURE.md](docs/COMBINED_FEATURE.md) | Multi-file transcribe-diarize |

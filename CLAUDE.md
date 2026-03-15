# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PawnAI** is a Python monolith with two CLI applications:
- **pawn-diarize**: Speaker diarization, transcription, and embedding management
- **pawn-agent**: LLM-powered conversational agent (GitHub Copilot SDK) for analyzing pawn-diarize sessions

## Common Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run CLIs
pawn-diarize --help
pawn-agent --help

# Code quality
black pawn_diarize pawn_agent tests
isort pawn_diarize pawn_agent tests
flake8 pawn_diarize pawn_agent tests
mypy pawn_diarize pawn_agent

# Tests
pytest
pytest tests/test_cli.py::test_status   # single test
pytest --cov=pawn_diarize --cov-report=html
```

## Architecture

### pawn-diarize

Three-layer architecture under `pawn_diarize/`:
- **`cli/`** — Typer command definitions and Rich output; no business logic
- **`core/`** — Pure business logic (diarization, transcription, embeddings, analysis, S3, SiYuan, database, queue listener)
- **`utils/`** — Shared helper functions

All commands route through `pawn_diarize/__main__.py → cli/commands.py`. The `core/__init__.py` uses `__getattr__` for lazy-loading heavy ML models (pyannote, NeMo) to keep startup fast.

Key core modules:
- `diarization.py` — Speaker identification with pyannote.audio + DBSCAN clustering
- `transcription.py` — Dual backend: NeMo Parakeet (`nemo`) or CTranslate2 faster-whisper (`whisper`)
- `embeddings.py` — 512-dim speaker embeddings stored in PostgreSQL with pgvector
- `combined.py` — Multi-file transcription+diarization with cross-file speaker alignment
- `database.py` — SQLAlchemy ORM; tables: `embeddings`, `speaker_names`, `transcription_segments`, `session_analysis`
- `s3.py` — S3/MinIO integration with transparent temp-file downloading and glob expansion
- `queue_listener.py` — Background job worker with lease-based concurrency

### pawn-agent

Under `pawn_agent/`, wraps GitHub Copilot SDK (`CopilotClient`) via `core/agent.py::ConversationAgent`. Tools are **auto-discovered** from `pawn_agent/tools/` — any non-private module that exports `NAME`, `DESCRIPTION`, and `build(cfg, client)` is registered automatically at session start. No registration code is needed; just add a module to `tools/`.

Current tools: `query_conversation`, `analyze_custom`, `save_to_siyuan`.

Utils under `pawn_agent/utils/`: `config.py` (loads `pawnai.yaml`), `db.py` (ORM session factory), `transcript.py` (fetch/format transcript), `siyuan.py` (SiYuan API helpers).

### Configuration

Both apps read `pawnai.yaml` (config precedence: CLI flags → `pawnai.yaml` → environment variables → defaults). Key sections: `models`, `db_dsn`, `s3`, `siyuan`, `agent`, `queue`. Database migrations managed with Alembic (`alembic.ini`).

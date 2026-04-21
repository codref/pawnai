# AGENTS.md

Compact instruction file for future OpenCode sessions working in the PawnAI repository.

## Project Structure

Python monolith with **three CLI entry points**:
- `pawn-diarize` — audio diarization, transcription, embedding management
- `pawn-agent` — LLM-powered conversational agent for analyzing sessions
- `pawn-server` — HTTP API server (`POST /v1/chat/completions`) + queue listener

Package boundaries (all in `pyproject.toml` `[tool.setuptools] packages`):
- `pawn_core/` — shared primitives: config, database ORM (`Base`), transcription engine, TTS, SiYuan client
- `pawn_diarize/` — CLI (`cli/commands.py`) and core business logic (`core/`)
- `pawn_agent/` — CLI, PydanticAI agent, auto-discovered tools (`tools/`), utilities
- `pawn_server/` — FastAPI server (`core/api_server.py`) and queue listener

**Important**: `pawn_core/transcription.py` is the real transcription engine. `pawn_diarize/core/transcription.py` is only a backward-compatible re-export.

## Setup

The repo has a `uv.lock` file and supports both traditional pip and uv workflows.

**With uv (recommended):**
```bash
uv venv
uv sync --extra dev          # installs from uv.lock
# or, to also update the lockfile from pyproject.toml:
# uv sync --extra dev --upgrade
```

**With pip:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Database required before first run:**
```bash
alembic upgrade head
```

Docker Compose provides dependencies (postgres on **5433**, siyuan, litellm proxy, phoenix):
```bash
docker compose -f docker/docker-compose.yml up -d postgres
```

## Developer Commands

Format:
```bash
black pawn_diarize pawn_agent tests       # line-length 100
isort pawn_diarize pawn_agent tests       # black profile
```

Lint / typecheck:
```bash
flake8 pawn_diarize pawn_agent tests
mypy pawn_diarize pawn_agent              # NOTE: does NOT include pawn_core, pawn_server, or tests by default
```

Test:
```bash
pytest                                    # default adds --cov=pawn_diarize --cov-report=term-missing
pytest tests/test_cli.py::test_status     # single test
pytest --cov=pawn_diarize --cov-report=html
```

## Architecture Quirks

- **Lazy-loaded ML models**: `pawn_diarize/core/__init__.py` uses `__getattr__` to defer importing heavy modules (pyannote, NeMo). Lightweight commands (`s3 ls`, `status`) stay fast because they never touch the lazy imports.
- **Auto-discovered tools**: `pawn_agent/tools/` modules are registered automatically at runtime. Any non-private module exporting `NAME`, `DESCRIPTION`, and `build(cfg)` (or `build(cfg, session_vars=…)`) becomes a tool. No registration boilerplate required.
- **Shared ORM base**: `pawn_core.database.Base` is the single declarative base for all packages. Package-specific models (e.g., `Embedding` in pawn_diarize, `AgentRun` in pawn_agent) inherit from it in their own modules.
- **Alembic DSN resolution**: `alembic.ini` intentionally omits `sqlalchemy.url`. The DSN is read at runtime from `PawnConfig().db_dsn` inside `migrations/env.py`.

## Configuration

`pawnai.yaml` is **gitignored** (do not commit tokens). It is auto-discovered in cwd or passed via `--config`.

Precedence: CLI flags → `pawnai.yaml` → environment variables → defaults.

Env-var convention: prefix `PAWN_` with `__` for nesting:
- `PAWN_DB_DSN` → `db_dsn`
- `PAWN_MODELS__HF_TOKEN` → `models.hf_token`
- `DATABASE_URL` → `db_dsn` (legacy alias)
- `HF_TOKEN` → `models.hf_token` (legacy alias)

## Testing Notes

- `tests/conftest.py` provides minimal fixtures (`temp_audio_file`, `temp_db_path`).
- `pyproject.toml` `[tool.pytest.ini_options]` sets `--cov=pawn_diarize --cov-report=term-missing` as default. To run without coverage, pass `--no-cov`.
- No CI workflows are present in `.github/workflows/`.

## Important Constraints

- PostgreSQL **pgvector** extension is required.
- Default DB DSN in config uses port **5433** (matching docker-compose), not 5432.
- `pawnai.yaml` contains real tokens in this working copy — never stage or commit it.
- If adding a new tool to `pawn_agent/tools/`, ensure it exports `NAME`, `DESCRIPTION`, and `build()`. The `__init__.py` loader inspects signatures and will pass `session_vars` only if the `build` function declares that parameter.

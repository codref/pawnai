# AGENTS.md

Compact operating notes for future Codex/OpenCode sessions in PawnAI.

## Repo Shape

Python monolith with three console scripts from `pyproject.toml`:
- `pawn-diarize`: diarization/transcription/embeddings and queue publishing.
- `pawn-agent`: LangGraph/PydanticAI conversational agent and tools.
- `pawn-server`: FastAPI OpenAI-compatible API, S3 queue listener, durable scheduler.

Packages:
- `pawn_core/`: shared config, DB base, transcription, TTS, SiYuan client.
- `pawn_diarize/`: CLI and audio business logic. Real transcription engine is `pawn_core/transcription.py`; `pawn_diarize/core/transcription.py` is only a compatibility re-export.
- `pawn_agent/`: CLI, LangGraph router/chat, scheduler, auto-discovered tools, agent DB models.
- `pawn_server/`: HTTP API, queue listener, scheduler CLI/server runner.

## Setup / Commands

Use `uv sync --extra dev` from `uv.lock` when possible. Pip fallback: `pip install -e ".[dev]"`.

DB/deps:
```bash
docker compose -f docker/docker-compose.yml up -d postgres
alembic upgrade head
```

Quality checks:
```bash
black pawn_diarize pawn_agent tests
isort pawn_diarize pawn_agent tests
flake8 pawn_diarize pawn_agent tests
mypy pawn_diarize pawn_agent
pytest --no-cov
pytest
```

Notes: pytest defaults to `--cov=pawn_diarize --cov-report=term-missing`; pass `--no-cov` for faster targeted runs. Mypy config intentionally checks only `pawn_diarize pawn_agent` unless paths are added explicitly.

## Runtime Architecture

- `pawn-server` always routes chat completions through LangGraph; the OpenAI `model` field is accepted for compatibility but ignored by the API server. Queue/scheduler runs can still pass model overrides through `run_agent_turn`.
- LangGraph state lives in `pawn_agent/core/langgraph_state.py` as bucketed `session_state`, `durable_facts`, `artifacts`, plus trimmed `recent_messages` (`RECENT_MESSAGE_LIMIT = 12`). Use helpers like `ensure_langgraph_state`, `get_state_field`, `set_state_fields`, `get_recent_messages`.
- LangGraph tool adapters and routing helpers live in `pawn_agent/core/langgraph_tools.py`; they call tool `*_impl` functions directly for graph nodes.
- Session discovery uses `pawn_agent/core/session_candidates.py`. `list_sessions` now has both text output (`list_sessions_impl`) and structured candidates (`list_session_candidates_impl`); LangGraph uses candidates to resolve vague prompts like "latest session".
- Queue listener (`pawn_server/core/queue_listener.py`) expects `{"command": "run", "prompt": ..., "session_id": ..., "model": ...}`. `session_id` is the diarization session name and also keys the LangGraph registry conversation.
- Agent run persistence is centralized in `pawn_agent/core/agent_runner.py`; it creates/updates `agent_runs` for queue and scheduled runs.

## Agent Tools

`pawn_agent/tools/` is auto-discovered by `pawn_agent/tools/__init__.py`: every non-private module with `build(cfg)` is imported. Tool modules should export `NAME`, `DESCRIPTION`, and `build()`.

If a tool needs request/session metadata, declare `build(cfg, session_vars=None)`. The loader passes `session_vars` only when that parameter exists. Keep reusable logic in an importable `*_impl` function so LangGraph nodes and tests can call it without PydanticAI wrapping.

Current scheduling tool: `propose_schedule_change` only creates proposals. It never directly mutates schedules; proposals must be approved by the application/CLI.

## Scheduler

Durable schedules are in `pawn_agent/core/scheduler.py` and DB models in `pawn_agent/utils/db.py`:
- Tables include `agent_schedules`, `agent_schedule_proposals`, `agent_schedule_fires`, and `agent_runs`.
- Supported kinds: `once`, `interval`, `cron` (`croniter` dependency).
- `AgentSchedulerConfig` defaults: enabled, 30s poll, max 5 due per tick, timezone `UTC`, stale fire 3600s.
- `pawn-server run` starts API/queue/scheduler according to config and flags; `--scheduler-only` runs only the scheduler.
- CLI management is under `pawn-server schedules`: `list`, `show`, `proposals`, `approve`, `reject`, `pause`, `resume`, `cancel`.

## Config

`pawnai.yaml` / `pawnai.yml` are auto-discovered and gitignored. Do not stage them; this working copy may contain real tokens.

Precedence is CLI/explicit overrides, YAML, env vars, defaults. Env vars use `PAWN_` plus `__` nesting:
- `PAWN_DB_DSN`, legacy `DATABASE_URL`
- `PAWN_MODELS__HF_TOKEN`, legacy `HF_TOKEN`
- `PAWN_AGENT__OPENAI__API_KEY`, `PAWN_AGENT__OPENAI__FAST_MODEL`, etc.

Default DB uses PostgreSQL on port `5433` and requires `pgvector`.

## Testing Pointers

- `tests/conftest.py` has minimal audio/DB fixtures.
- Scheduler coverage: `tests/test_agent_scheduler.py`.
- LangGraph/router coverage: `tests/test_langgraph_chat.py`, `tests/test_pawn_agent_cli.py`, `tests/test_push_queue_message.py`.
- Queue listener coverage: `tests/test_agent_queue_listener.py`.
- No CI workflows are present in `.github/workflows/`.

## Constraints / Gotchas

- `pawn_diarize/core/__init__.py` lazy-loads heavy ML modules via `__getattr__`; keep lightweight commands from importing pyannote/NeMo accidentally.
- `pawn_core.database.Base` is the shared SQLAlchemy declarative base. Package-specific models must inherit from it.
- `alembic.ini` intentionally omits `sqlalchemy.url`; `migrations/env.py` reads `PawnConfig().db_dsn`.
- Prefer structured DB/API helpers over parsing human-readable tool output. Only parse rendered output where compatibility requires it.
- Keep secrets out of commits, especially `pawnai.yaml`.

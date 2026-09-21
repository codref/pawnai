# AGENTS.md

Compact operating notes for future Codex/OpenCode sessions in PawnAI.

## Repo Shape

Python monolith with three console scripts from `pyproject.toml`:
- `pawn-diarize`: diarization/transcription/embeddings and queue publishing.
- `pawn-agent`: sallm conversational agent (ReAct + skills + CliTools) and domain tools.
- `pawn-server`: FastAPI OpenAI-compatible API, S3 queue listener, durable scheduler, optional Matrix bot.

Packages:
- `pawn_core/`: shared config, DB base, transcription, TTS, SiYuan client.
- `pawn_diarize/`: CLI and audio business logic. Real transcription engine is `pawn_core/transcription.py`; `pawn_diarize/core/transcription.py` is only a compatibility re-export.
- `pawn_agent/`: CLI, sallm harness, scheduler, domain tool impls + CliTool CLIs, agent DB models.
- `pawn_server/`: HTTP API, queue listener, Matrix bot, scheduler CLI/server runner.

## Setup / Commands

Use `uv sync --extra dev` from `uv.lock` when possible. Pip fallback: `pip install -e ".[dev]"`.
For the Matrix bot: `uv sync --extra matrix` (pulls `matrix-nio[e2e]`).

`sallm-agent` is pulled as an editable path dependency (`../sallm-agent` via `[tool.uv.sources]`). Alternative: `uv add git+https://github.com/codref/sallm-agent.git`.

DB/deps:
```bash
docker compose -f docker/docker-compose.yml up -d postgres
alembic upgrade head
```

Local embeddings (sallm default) need Ollama:
```bash
ollama pull qwen3-embedding:0.6b
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

Notes: pytest defaults to `--cov=pawn_diarize --cov-report=term-missing`; pass `--no-cov` for faster targeted runs. Mypy config intentionally checks only `pawn_diarize pawn_agent` unless paths are added explicitly. Requires Python `>=3.12` (sallm).

## Runtime Architecture

- `pawn-server` always routes chat completions through the embedded **sallm** agent. The OpenAI `model` field is accepted for compatibility but ignored by the API server. Queue/scheduler runs can still pass model overrides through `run_agent_turn`.
- Harness modules: `sallm_factory.py` (build Agent), `sallm_session.py` (async façade), `sallm_registry.py` (session pool), `sallm_skills.py`, `sallm_tools.py`.
- Durable chat memory lives in SQLite + Lance under `agent.sallm.state_dir` (default `.sallm/`). PostgreSQL still holds transcripts, analyses, `agent_runs`, and schedules. `langgraph_session_state` is unused leftover schema (not dropped in v1).
- Dual `session_id` semantics: queue/scheduler conversation key **is** the diarization session name. API uses OpenAI `user` (or message hash) as the conversation key; Matrix bot uses `matrix:{room_id}`. Tools must discover diarization ids via `sessions_list`.
- Queue listener (`pawn_server/core/queue_listener.py`) expects `{"command": "run", "prompt": ..., "session_id": ..., "model": ...}`.
- Matrix bot (`pawn_server/core/matrix_bot.py`) is an optional `serve` worker: in-process `run_agent_turn` with `source="matrix"` (same tools/skills as CLI chat). See `docs/MATRIX_BOT.md`.
- Agent run persistence is centralized in `pawn_agent/core/agent_runner.py`.

## Agent Tools / Skills

Domain logic stays in `pawn_agent/tools/*_impl`. Production path uses **CliTools** under `pawn_agent/tools/cli/` registered in `sallm_tools.py`.

Migrated CliTools: `sessions_list`, `session_transcript`, `session_analyze`,
`session_delete`, `siyuan_save`, `schedule_propose`, `queue_push`.

`siyuan_save`: prefer `--from-analysis` or `session_analyze --save`. Free-form
Markdown uses `--content-file @note` plus a ```file note` block (sallm writes a
temp file). Do not paste long bodies into `--content`. Notes skill includes
`sessions_list` so analyze+save can resolve a diarization id; never dump tool
errors / fallback journaling into SiYuan.

`session_delete` permanently wipes diarization DB rows (segments, analyses,
`session_state`, graph triples) for one session name. It always requires
`--confirm` to exactly match `--session-id`; the agent must ask the user
in chat before calling. It does not clear sallm chat memory or SiYuan notes.

SiYuan diary transcripts (diarize, not agent): `pawn-diarize push-siyuan`
creates/updates a stable session doc (Speakers + Transcript managed;
Annotations preserved). Opt-in auto-push after each `transcribe-diarize`
chunk via `siyuan.auto_push_transcript`. Mapping table: `siyuan_session_docs`.
Legacy analysis upsert remains `pawn-diarize sync-siyuan`.

Skills (modes): `converse`, `sessions`, `notes`, `scheduling`, `ops` — see `sallm_skills.py`.

Session memory is owned by sallm (SQLite + Lance + `Agent.remember`). Old memorize/recall/vectorize tools were removed.

`pawn-agent tools` lists CliTools from `sallm_tools.py`.

Current scheduling tool: `schedule_propose` only creates proposals. Approvals stay on `pawn-server schedules`.

## Scheduler

Durable schedules are in `pawn_agent/core/scheduler.py` and DB models in `pawn_agent/utils/db.py`:
- Tables include `agent_schedules`, `agent_schedule_proposals`, `agent_schedule_fires`, and `agent_runs`.
- Supported kinds: `once`, `interval`, `cron` (`croniter` dependency).
- `AgentSchedulerConfig` defaults: enabled, 30s poll, max 5 due per tick, timezone `UTC`, stale fire 3600s.
- `pawn-server serve` starts API/queue/scheduler/Matrix according to config and flags; `--scheduler-only` / `--matrix-only` run a single worker.
- CLI management is under `pawn-server schedules`: `list`, `show`, `proposals`, `approve`, `reject`, `pause`, `resume`, `cancel`.
- Queue admin is under `pawn-server queue`: `stats`, `pause`, `resume`, `empty`.
  Discovers `agent_queue`, `diarize_queue`, and `queue_producers` from config.
  `stats` lists all by default; mutating commands take `--name` / `--topic` / `--all`.
  Pause writes `{topic}/.paused` in the queue bucket; agent and diarize listeners
  stop claiming new messages until `resume`.

## Matrix bot

Inbound chatbot under `pawn-server serve` (not a separate console script):
- Config: `matrix_bot:` (`enabled` default false). Env: `PAWN_MATRIX_BOT__*`.
- Flags: `--no-matrix`, `--matrix-only` (mutually exclusive with `--scheduler-only`).
- Rooms need `command_prefix` (default `!pawn`); DMs are free-speak; `/reset` clears the room conversation; `/stats` shows sallm session metrics (also on `pawn-agent chat`).
- Keep `device_id` + `store_path` stable for E2EE. Do not confuse with `queue_producers.matrix` (outbound notifications only).

## Config

`pawnai.yaml` / `pawnai.yml` are auto-discovered and gitignored. Do not stage them; this working copy may contain real tokens.

Precedence is CLI/explicit overrides, YAML, env vars, defaults. Env vars use `PAWN_` plus `__` nesting:
- `PAWN_DB_DSN`, legacy `DATABASE_URL`
- `PAWN_MODELS__HF_TOKEN`, legacy `HF_TOKEN`
- `PAWN_AGENT__OPENAI__API_KEY`, `PAWN_AGENT__OPENAI__FAST_MODEL`, etc.
- `PAWN_AGENT__SALLM__STATE_DIR`, `PAWN_AGENT__SALLM__MAX_STEPS`, `PAWN_AGENT__SALLM__PROFILE`, `PAWN_AGENT__SALLM__OTLP_ENDPOINT`
- `PAWN_MATRIX_BOT__ENABLED`, `PAWN_MATRIX_BOT__HOMESERVER_URL`, `PAWN_MATRIX_BOT__USER_TOKEN`, etc.

Chat model comes from `agent.openai` (etc.) and is mapped to LiteLLM via `cfg.litellm_model` (`openai:gpt-4o` → `openai/gpt-4o`). Optional Tempo: `agent.sallm.otlp_endpoint` / `metrics_port` (off by default for the server).
`agent.sallm.profile` is a sallm CompiledProfile YAML/JSON path (default `large.yaml` in `pawn_agent/profiles/`, 10× token budgets). Empty string uses stock sallm limits.

Default DB uses PostgreSQL on port `5433` and requires `pgvector`.

## Testing Pointers

- `tests/conftest.py` has minimal audio/DB fixtures.
- Scheduler coverage: `tests/test_agent_scheduler.py`.
- Sallm harness: `tests/test_sallm_registry.py`, `tests/test_sallm_cli_tools.py`, `tests/test_pawn_agent_cli.py`, `tests/test_push_queue_message.py`.
- Queue listener coverage: `tests/test_agent_queue_listener.py`.
- Matrix bot helpers: `tests/test_matrix_bot.py`.
- No CI workflows are present in `.github/workflows/`.

## Constraints / Gotchas

- `pawn_diarize/core/__init__.py` lazy-loads heavy ML modules via `__getattr__`; keep lightweight commands from importing pyannote/NeMo accidentally.
- `pawn_core.database.Base` is the shared SQLAlchemy declarative base. Package-specific models must inherit from it.
- `alembic.ini` intentionally omits `sqlalchemy.url`; `migrations/env.py` reads `PawnConfig().db_dsn`.
- Prefer structured DB/API helpers over parsing human-readable tool output. Only parse rendered output where compatibility requires it.
- Keep secrets out of commits, especially `pawnai.yaml`.
- CliTool subprocesses inherit the parent environment and discover `pawnai.yaml` / `PAWN_*` the same way as the host process.

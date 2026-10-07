# AGENTS.md

Compact operating notes for future Codex/OpenCode sessions in PawnAI.

## Repo Shape

Python monolith with three console scripts from `pyproject.toml`:
- `pawn-diarize`: diarization/transcription/embeddings and queue publishing.
- `pawn-agent`: sallm conversational agent (ReAct + skills + CliTools) and domain tools.
- `pawn-server`: FastAPI OpenAI-compatible API, S3 queue listener, durable scheduler, optional Matrix bot.

Packages:
- `pawn_core/`: shared config, DB base, transcription, TTS, S3 vault store.
- `pawn_diarize/`: CLI and audio business logic. Real transcription engine is `pawn_core/transcription.py`; `pawn_diarize/core/transcription.py` is only a compatibility re-export.
- `pawn_agent/`: CLI, sallm harness, scheduler, domain tool impls + CliTool CLIs, agent DB models.
- `pawn_server/`: HTTP API, queue listener, Matrix bot, vault watcher, scheduler CLI/server runner.
- `obsidian-plugin/pawn/`: Obsidian desktop/mobile plugin (Copilot-style chat pane, prompt commands, background jobs, uploads). MIT, written from scratch; do not copy obsidian-copilot (AGPL) code into it.
- `browser-extension/pawn/`: MV3 side panel that appends ordered text/image snippets to vault capture notes or session Annotations (no chat UI). See `docs/BROWSER_CAPTURE.md`.

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
- Conversation keys: queue/scheduler use the diarization session name; API uses OpenAI `user` (or message hash); Matrix bot uses `matrix:{room_id}`; vault tasks/chat use `note:{path}`. Tools must discover diarization ids via `sessions_list` when the chat key is not a session name.
- Queue listener (`pawn_server/core/queue_listener.py`) expects `{"command": "run"|"vault_run", "prompt": ..., "session_id": ..., "model": ...}` (vault_run is usually started by the vault watcher / HTTP API, not the queue).
- Matrix bot (`pawn_server/core/matrix_bot.py`) is an optional `serve` worker: in-process `run_agent_turn` with `source="matrix"` (same tools/skills as CLI chat). When `matrix_bot.notify_room_id` is set, it also consumes `queue_producers.matrix` for outbound alerts. See `docs/MATRIX_BOT.md`.
- Obsidian (`docs/OBSIDIAN_AGENT.md`): Sync Engine keeps devices ↔ S3 in sync; Pawn reads/writes the bucket via `pawn_core/vault.py`. Three modes:
  - Plugin chat: `POST /v1/pawn/chat` (SSE `progress`/`answer`/`job`/`error`/`done`; prompt built server-side in `pawn_server/core/chat_context.py`).
  - Background jobs: `POST /v1/jobs` (+ `/upload`, `/events` SSE, `/approve`, `/cancel`, `/dismiss`). Always 202, no sync/async race. Logic in `pawn_server/core/jobs.py`; rows in `vault_tasks` (`kind` = ask | push_note | upload, `payload`, `result_text`). `ask` jobs mirror to `Pawn/Tasks/{id}.md`. Offline plugin writes `todo` task notes for `vault_watcher`. Status events: `pawn_server/core/job_events.py` (in-process only).
  - Stock obsidian-copilot → `/v1/chat/completions` + `/v1/models` (list content parts, CORS, `X-Pawn-Conversation`, streamed keep-alives/`reasoning_content` progress via `pawn_server/core/progress.py`).
  - `/v1/vault/tasks*` are deprecated aliases. Approve indexes into sallm memory. Session keys `note:{path}` / `chat:{uuid}`.
- Browser capture (`docs/BROWSER_CAPTURE.md`): MV3 extension side panel (no chat) posts ordered text/image snippets via `GET/POST /v1/captures` and `GET /v1/sessions`. Default note is `Pawn/Captures/{date} {title}.md`; session attach splices into transcript `## Annotations` when mapped. Images under `Pawn/Captures/assets/`. Logic in `pawn_server/core/captures.py`.
- Agent run persistence is centralized in `pawn_agent/core/agent_runner.py`.
- Coworker loop (`coworker:` in config, off by default): after diarization, `chain_agent.command: session_completed` extracts items, scores them against `Goals.md`, writes `Pawn/Items` and `Pawn/Today.md`, and notifies only on an active-thread interrupt. Triage is `todo|file|task|delete|ignore` (plus Matrix `later`) from the plugin Inbox timeline, item-note frontmatter, `GET/POST /v1/items`, or chat. `/idea` writes one note under `Ideas/` (`status: inbox`) and does not start a model turn. The plugin Inbox is open Items (time-grouped; Add TODO / File / Delete; multi-select and delete-all); the Ideas chip lists idea notes (Keep / Goal / Task / Drop). Idea notes are not extracted. `/goal` and `/park` write `Goals.md`, and a normal prompt only drafts `Pawn/Reviews/goal-proposal.md` until `/goal apply`. Morning and weekly crons live in `pawn_server/core/coworker_worker.py`. See `docs/COWORKER.md`.
- People bios (`docs/PEOPLE.md`): vault notes `People/{speaker_id}.md` linked to the Speakers gallery. After `session_completed`, `speakers_refresh` appends Appearances and proposes or applies Facts under autonomy `people_refresh`. Transcript Speakers tables wikilink into People notes. Tools: `people_show` / `people_ensure` / `people_append`. One-off CLI: `pawn-server coworker people-refresh --session ID` (does not create Items; use `coworker process` for Items only).

## Agent Tools / Skills

Domain logic stays in `pawn_agent/tools/*_impl`. Production path uses **CliTools** under `pawn_agent/tools/cli/` registered in `sallm_tools.py`.

Migrated CliTools: `sessions_list`, `session_transcript`, `session_analyze`,
`session_screenshots`, `session_delete`, `session_relabel`, `speakers_list`,
`speakers_show`, `speakers_update`, `speaker_enroll`, `session_reidentify`,
`people_show`, `people_ensure`, `people_append`, `note_read`, `note_search`,
`note_write`, `note_append`, `task_update`, `schedule_propose`, `queue_push`,
`goal_propose`.

`session_analyze --save` / `note_write`: prefer `--save` for analysis Markdown
under `Pawn/Analyses/`. Free-form notes use `note_write --content-file @note`
plus a ```file note` block. Do not paste long bodies into `--content`. Notes
skill includes `sessions_list` so analyze+save can resolve a diarization id;
never dump tool errors into vault notes.

`note_read` / `note_append` / `task_update`: vault tools for the task loop
(skill `vault_tasks`). Write guards: free write under `Pawn/`; outside that
root only with `pawn: editable`; never `.obsidian/`.

`session_delete` permanently wipes diarization DB rows (segments, analyses,
`session_state`, graph triples, `session_captures`) for one session name. It always requires
`--confirm` to exactly match `--session-id`; the agent must ask the user
in chat before calling. It does not clear sallm chat memory or vault notes.

`session_relabel` renames a speaker across one session (`--from SPEAKER_00
--to Davide` or a wrong display name). Updates transcript labels,
`speaker_names`, `session_speaker_map`, and `session_state` prior-speaker
keys. Same core as `pawn-diarize session-relabel`. For lasting voice
identity use the curated Speakers gallery (`docs/SPEAKERS.md`):
`speakers create/enroll`, then `reidentify` / `session_reidentify`.
Runtime diarization never auto-enrolls voiceprints.

`pawn-diarize rediarize --session X --confirm` re-runs diarization labels
only (keeps ASR text). `pawn-diarize retranscribe --session X --confirm`
repairs/downloads the session's stored S3 audio, wipes segments/state, and
runs a full transcribe-diarize overwrite — use this when transcript text
is missing or wrong.

Vault diary transcripts (diarize, not agent): `pawn-diarize push-vault`
creates/updates a stable session note (Speakers, Transcript, and Screenshots
managed; Annotations preserved). Optional `annotations` and `screenshots` on
`transcribe-diarize` are stored in `session_captures` before transcription.
Opt-in auto-push after each chunk via `vault.auto_push_transcript`.
`vault.screenshot_vision` (default true) describes screenshot changes with the
background vision model; `session_screenshots --summarize` can do that later.
Mapping table: `vault_notes`.

Skills (modes): `converse`, `sessions`, `notes`, `scheduling`, `ops`,
`vault_tasks`, `tasknotes`, `coworker` — see `sallm_skills.py`.

`tasknotes` writes TaskNotes-compatible notes under `{agent_root}/TaskNotes/`
(tasks, project stubs, `.base` boards, and a checklist proposal). The user
picks from the checklist, or asks Pawn to create the list directly. Same
title and assignee are not duplicated. `task_update` remains the vault-job
tool and is not used for these notes.

`coworker` drafts goal threads (`goal_propose`). It does not write `Goals.md`
and it does not capture ideas. `/idea` writes the idea note directly. `/goal`,
`/park`, and `/goal apply` are the chat commands that write `Goals.md`.

Session memory is owned by sallm (SQLite + Lance + `Agent.remember`). Old memorize/recall/vectorize tools were removed.

`pawn-agent tools` lists CliTools from `sallm_tools.py`.

Current scheduling tool: `schedule_propose` only creates proposals. Approvals stay on `pawn-server schedules`.

## Scheduler

Durable schedules are in `pawn_agent/core/scheduler.py` and DB models in `pawn_agent/utils/db.py`:
- Tables include `agent_schedules`, `agent_schedule_proposals`, `agent_schedule_fires`, `agent_runs`, `vault_notes`, and `vault_tasks`.
- Supported kinds: `once`, `interval`, `cron` (`croniter` dependency).
- `AgentSchedulerConfig` defaults: enabled, 30s poll, max 5 due per tick, timezone `UTC`, stale fire 3600s.
- `pawn-server serve` starts API/queue/scheduler/Matrix/vault-watcher according to config and flags; `--scheduler-only` / `--matrix-only` / `--vault-watcher-only` run a single worker.
- CLI management is under `pawn-server schedules`: `list`, `show`, `proposals`, `approve`, `reject`, `pause`, `resume`, `cancel`.
- Queue admin is under `pawn-server queue`: `stats`, `pause`, `resume`, `empty`.
  Discovers `agent_queue`, `diarize_queue`, and `queue_producers` from config.
  `stats` lists all by default; mutating commands take `--name` / `--topic` / `--all`.
  Pause writes `{topic}/.paused` in the queue bucket; agent and diarize listeners
  stop claiming new messages until `resume`.
- API IP blacklist is under `pawn-server blacklist`: `list`, `add`, `remove`, `clear`.
  Auto-ban heuristics (auth 401 / path-scan 404) and `api.whitelist_ips` /
  `api.enable_docs` live on `ApiSection` — disable docs when exposing port 8000.
  Behind a reverse proxy set `api.trust_proxy` + `api.trusted_proxies` so
  X-Real-IP / X-Forwarded-For are used (off by default).
  Direct TLS: `api.ssl_certfile` + `api.ssl_keyfile` (or `--ssl-*` flags);
  `make ssl-cert` writes a self-signed pair under `certs/`.

## Matrix bot

Inbound chatbot under `pawn-server serve` (not a separate console script):
- Config: `matrix_bot:` (`enabled` default false). Env: `PAWN_MATRIX_BOT__*`.
- Flags: `--no-matrix`, `--matrix-only` (mutually exclusive with `--scheduler-only` / `--vault-watcher-only`).
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
- `PAWN_VAULT__S3__BUCKET`, `PAWN_VAULT__S3__ACCESS_KEY`, `PAWN_VAULT__S3__SECRET_KEY`, `PAWN_VAULT__S3__ENDPOINT_URL`, etc.
- `PAWN_MATRIX_BOT__NOTIFY_ROOM_ID` for outbound ready-for-review alerts
- `api.cors_origins`, `api.include_system_prompt`, `api.stream_progress`, `api.upload_audio_target`
  (queue_producers name for audio uploads), `api.enable_docs`, `api.whitelist_ips`,
  `api.bruteforce_*` / thresholds — see `ApiSection` in `pawn_agent/utils/config.py`

Chat models are a catalog under `agent.providers`. Each entry pairs an OpenAI-compatible provider (`base_url`, `api_key`) with a sallm compiled profile. The selectable id is `provider@model` (for example `ollama@gemma4:4b`). LiteLLM always sees `openai/{model}` plus that provider’s URL and key. `agent.profiles_dir` is a folder outside the repo; relative profile names also fall back to `pawn_agent/profiles/` (`large.yaml`, 10× token budgets). `agent.default` is the background model. `/model <id>` changes it at runtime and writes `{state_dir}/background_model`; `/model reset` restores the yaml default. Set `vision: true` on a catalog model to let Matrix and Obsidian chat attach images; `/v1/pawn/models` exposes that flag. Each image is captioned once per conversation unless the user re-reads it. Interactive chat and UI jobs send the composer’s id; autonomous work (scheduler, coworker, Matrix, vault watcher, queue without `model`) uses the background default. A legacy `agent.openai` (or anthropic/google/groq/mistral) block still boots when `providers` is omitted. `agent.sallm` is harness settings only (`state_dir`, `max_steps`, embeddings). Optional Tempo: `agent.sallm.otlp_endpoint` / `metrics_port` (off by default for the server).

Default DB uses PostgreSQL on port `5433` and requires `pgvector`.

## Testing Pointers

- `tests/conftest.py` has minimal audio/DB fixtures.
- Scheduler coverage: `tests/test_agent_scheduler.py`.
- Sallm harness: `tests/test_sallm_registry.py`, `tests/test_sallm_cli_tools.py`, `tests/test_pawn_agent_cli.py`, `tests/test_push_queue_message.py`.
- Queue listener coverage: `tests/test_agent_queue_listener.py`.
- Matrix bot helpers: `tests/test_matrix_bot.py`.
- Vault: `tests/test_vault_store.py`, `tests/test_vault_transcript.py`, `tests/test_vault_tasks.py`, `tests/test_vault_watcher.py`.
- HTTP API: `tests/test_api_chat_compat.py` (OpenAI/Copilot compat), `tests/test_api_jobs.py` (jobs + `/v1/pawn/chat`).
- Plugin: `cd obsidian-plugin/pawn && npm run build` (bumps the `manifest.json` patch version, then `tsc -noEmit`); no JS test runner. CI sets `CI=true`, which skips the bump so an `obsidian-plugin-v*` tag still matches the manifest. `make dist` in that directory (or `make obsidian-plugin-dist` from the repo root) writes `pawn.zip` (`pawn/main.js`, `manifest.json`, `styles.css`). `.github/workflows/obsidian-plugin.yml` uploads that zip as the `pawn-obsidian-plugin` artifact and, on an `obsidian-plugin-v*` tag whose version matches `manifest.json`, publishes it on the GitHub release.
- Browser capture extension: plain MV3 JS (no compile). `make browser-extension` verifies files; `make browser-extension-dist` writes `browser-extension/pawn/pawn-capture.zip` (`pawn-capture/{manifest.json,…}`). Load unpacked that folder in Chrome/Edge; see `docs/BROWSER_CAPTURE.md`.
- No CI workflows are present in `.github/workflows/`.

## Constraints / Gotchas

- `pawn_diarize/core/__init__.py` lazy-loads heavy ML modules via `__getattr__`; keep lightweight commands from importing pyannote/NeMo accidentally.
- `pawn_core.database.Base` is the shared SQLAlchemy declarative base. Package-specific models must inherit from it.
- `alembic.ini` intentionally omits `sqlalchemy.url`; `migrations/env.py` reads `PawnConfig().db_dsn`.
- Prefer structured DB/API helpers over parsing human-readable tool output. Only parse rendered output where compatibility requires it.
- Keep secrets out of commits, especially `pawnai.yaml`.
- CliTool subprocesses inherit the parent environment and discover `pawnai.yaml` / `PAWN_*` the same way as the host process.
- Vault Sync Engine: asymmetric storage and client-side encryption must stay off so Pawn can read/write plain Markdown keys.

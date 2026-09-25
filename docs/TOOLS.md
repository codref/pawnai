# pawn-agent Tools

Production tools are **sallm CliTools**: thin CLI scripts under
`pawn_agent/tools/cli/` that call `*_impl` helpers in sibling modules.

There is no PydanticAI / Copilot auto-discovery registry anymore.
`pawn-agent tools` lists the CliTool registry from `sallm_tools.py`.

## Layout

```text
pawn_agent/tools/
  list_sessions.py       # list_sessions_impl / list_session_candidates_impl
  query_conversation.py  # query_conversation_impl
  analyze_summary.py     # analyze_summary_impl
  delete_session.py      # delete_session_impl
  session_relabel.py     # session_relabel_impl (wraps pawn_diarize session_relabel)
  notes_impl.py          # note_read/search/write/append + task_update
  save_to_vault.py       # analysis Markdown → Pawn/Analyses/
  propose_schedule.py    # propose_schedule_change_impl
  push_queue_message.py  # push_queue_message_impl
  cli/
    sessions_list.py
    session_transcript.py
    session_analyze.py
    session_delete.py
    session_relabel.py
    note_read.py
    note_search.py
    note_write.py
    note_append.py
    task_update.py
    schedule_propose.py
    queue_push.py
```

## Adding a tool

1. Put reusable logic in `pawn_agent/tools/<name>.py` as an `*_impl` function.
2. Add `pawn_agent/tools/cli/<tool_name>.py` with `argparse` + `--help`.
3. Register a `CliTool` in `pawn_agent/core/sallm_tools.py` (summary = flag cheat-sheet).
4. Optionally expose it from a skill in `sallm_skills.py`.

CLI contract: flags only, human-readable stdout, inherit `pawnai.yaml` / `PAWN_*`
from the parent process.

### Session analysis + vault notes

`session_analyze --save` persists analysis to PostgreSQL and writes Markdown
to `Pawn/Analyses/{session_id}.md` via `save_to_vault`. Free-form notes use
`note_write --content-file @note`. See `docs/OBSIDIAN_AGENT.md`.

## Available CliTools

| CliTool | Impl module | Description |
|---|---|---|
| `sessions_list` | `list_sessions` | List diarization sessions |
| `session_transcript` | `query_conversation` | Fetch one transcript |
| `session_analyze` | `analyze_summary` | Structured analysis (optional `--save` to vault) |
| `session_delete` | `delete_session` | Permanently delete one session (requires `--confirm`) |
| `session_relabel` | `session_relabel` | Rename a speaker across a session (segments + embeddings) |
| `note_read` | `notes_impl` | Read a vault Markdown note |
| `note_search` | `notes_impl` | List/filter vault notes |
| `note_write` | `notes_impl` | Create/overwrite a vault note (write guards) |
| `note_append` | `notes_impl` | Append to a vault note |
| `task_update` | `notes_impl` | Update task note status / Result |
| `schedule_propose` | `propose_schedule` | Create schedule proposals (approve via CLI) |
| `queue_push` | `push_queue_message` | Publish to a named queue producer |

## Shared helpers

| Module | Contents |
|---|---|
| `pawn_core/vault.py` | `VaultStore`, frontmatter helpers, write guards |
| `pawn_core/vault_config.py` | `vault_store_from_config` |
| `pawn_core/vault_db.py` | `vault_notes` ORM helpers |

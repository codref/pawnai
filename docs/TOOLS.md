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
  save_to_siyuan.py      # save_to_siyuan_impl / save_analysis_to_siyuan_impl
  delete_session.py      # delete_session_impl
  session_relabel.py     # session_relabel_impl (wraps pawn_diarize session_relabel)
  propose_schedule.py    # propose_schedule_change_impl
  push_queue_message.py  # push_queue_message_impl
  cli/
    sessions_list.py
    session_transcript.py
    session_analyze.py
    session_delete.py
    session_relabel.py
    siyuan_save.py
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

### Large / multiline bodies

Do not paste long Markdown into a ```run` line (`--content` rejects newlines and
long strings). Prefer:

1. `siyuan_save --session-id ID --from-analysis` after `session_analyze`
2. `session_analyze --session-id ID --save`
3. Free-form: `siyuan_save ... --content-file @note` plus a ```file note` block
   (sallm writes a temp file and rewrites `@note` to that path)

## Available CliTools

| CliTool | Impl module | Description |
|---|---|---|
| `sessions_list` | `list_sessions` | List diarization sessions |
| `session_transcript` | `query_conversation` | Fetch one transcript |
| `session_analyze` | `analyze_summary` | Structured analysis (+ optional SiYuan) |
| `session_delete` | `delete_session` | Permanently delete one session (requires `--confirm`) |
| `session_relabel` | `session_relabel` | Rename a speaker across a session (segments + embeddings) |
| `siyuan_save` | `save_to_siyuan` | Save Markdown / `--from-analysis` / `--content-file` to SiYuan |
| `schedule_propose` | `propose_schedule` | Create schedule proposals (approve via CLI) |
| `queue_push` | `push_queue_message` | Publish to a named queue producer |

## Shared helpers

| Module | Contents |
|---|---|
| `utils/db.py` | ORM + `get_session_analysis` / schedule helpers |
| `utils/transcript.py` | `fetch_transcript` |
| `utils/siyuan.py` | `do_save_to_siyuan` |
| `utils/analysis.py` | `run_analysis` (uses `llm_sub`) |

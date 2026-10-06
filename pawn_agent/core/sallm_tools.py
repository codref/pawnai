"""Register pawn domain capabilities as sallm CliTools.

Each tool is a thin subprocess CLI under ``pawn_agent.tools.cli``.
Business logic stays in the existing ``*_impl`` modules — CLIs only adapt
argparse flags → impl calls.

Assumption: CliTool children inherit the server/CLI process environment, so
``pawnai.yaml`` / ``PAWN_*`` discovery works the same as for the parent.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sallm import CliTool

# Directory that holds the CLI entry modules (sessions_list.py, …).
_CLI_DIR = Path(__file__).resolve().parent.parent / "tools" / "cli"


def _cli_argv(script_name: str) -> list[str]:
    """Build argv for ``python path/to/cli/<script_name>``."""
    return [sys.executable, str(_CLI_DIR / script_name)]


def build_pawn_clitools() -> dict[str, CliTool]:
    """Return the CliTool registry used by every pawn sallm Agent."""
    return {
        "sessions_list": CliTool(
            name="sessions_list",
            argv=_cli_argv("sessions_list.py"),
            summary=(
                "List diarization conversation sessions (newest first). "
                "Flags: --name-filter TEXT --limit N. "
                "Use before session_transcript / session_analyze when the "
                "session id is unknown. Never invent session ids."
            ),
        ),
        "session_transcript": CliTool(
            name="session_transcript",
            argv=_cli_argv("session_transcript.py"),
            summary=(
                "Fetch the full transcript for one session. "
                "Flags: --session-id ID (required). "
                "Use when the user wants to inspect or quote the transcript."
            ),
        ),
        "session_analyze": CliTool(
            name="session_analyze",
            argv=_cli_argv("session_analyze.py"),
            summary=(
                "Run the standard structured session analysis and persist it "
                "to the database. "
                "Required: --session-id ID (or bare ID). "
                "Optional: --save --title TEXT — when the user asks to analyze "
                "AND save to the vault, use this once with --save; do not also "
                "call note_write for the same analysis. "
                "Analyze one session per invocation."
            ),
        ),
        "session_screenshots": CliTool(
            name="session_screenshots",
            argv=_cli_argv("session_screenshots.py"),
            summary=(
                "List screenshots captured during a diarization session "
                "(time, display output, summary state, vault embed). "
                "Required: --session-id ID (or bare ID). "
                "Pass --summarize to describe changes with the background "
                "vision model. Pass --id ITEM with --summarize for one shot. "
                "Use when the user asks to process screenshots after the "
                "session, or to refer to a single screenshot."
            ),
        ),
        "session_delete": CliTool(
            name="session_delete",
            argv=_cli_argv("session_delete.py"),
            summary=(
                "Permanently delete one diarization session from PostgreSQL "
                "(segments, analyses, session_state, graph triples, captures). "
                "Required: --session-id ID --confirm ID where both values "
                "match exactly. ALWAYS ask the user to confirm the exact "
                "session name in chat before calling. Never invent ids. "
                "Does not clear sallm chat memory or vault notes."
            ),
        ),
        "session_relabel": CliTool(
            name="session_relabel",
            argv=_cli_argv("session_relabel.py"),
            summary=(
                "Rename a speaker across one diarization session and propagate "
                "the display name to speaker_names / session_speaker_map / "
                "session_state. "
                "Required: --session-id ID --from LABEL --to NAME. "
                "--from may be SPEAKER_XX or a current display name "
                "(e.g. --from SPEAKER_00 --to Davide). "
                "Existing vault transcript notes refresh automatically; pass "
                "--push-vault to create/update even without a prior mapping. "
                "Use when the user asks to change / correct / rename a speaker "
                "on a session. Prefer speakers_list + speaker_enroll + "
                "session_reidentify when building lasting voice identity. "
                "Never invent session ids — sessions_list first."
            ),
        ),
        "speakers_list": CliTool(
            name="speakers_list",
            argv=_cli_argv("speakers_list.py"),
            summary=(
                "List curated Speakers gallery people (id, name, enrollment count). "
                "Optional: --include-inactive. Use before speaker_enroll / people_show."
            ),
        ),
        "speakers_show": CliTool(
            name="speakers_show",
            argv=_cli_argv("speakers_show.py"),
            summary=(
                "Show one Speakers gallery person: aliases, short notes, enrollments, "
                "and the linked People/{id}.md path. Arg: SPEAKER id or name."
            ),
        ),
        "speakers_update": CliTool(
            name="speakers_update",
            argv=_cli_argv("speakers_update.py"),
            summary=(
                "Update gallery card fields only (not voice). "
                "Flags: --speaker NAME --alias A (repeatable) --notes TEXT "
                "--display-name NAME. Prefer people_append for vault bios. "
                "Never enrolls voiceprints."
            ),
        ),
        "people_show": CliTool(
            name="people_show",
            argv=_cli_argv("people_show.py"),
            summary=(
                "Read the vault person bio (People/{speaker_id}.md). "
                "Arg: SPEAKER id or display name. Use when asked who someone is."
            ),
        ),
        "people_ensure": CliTool(
            name="people_ensure",
            argv=_cli_argv("people_ensure.py"),
            summary=(
                "Create a stub People/{speaker_id}.md for a gallery speaker if missing. "
                "Required: --speaker NAME. Does not create gallery people."
            ),
        ),
        "people_append": CliTool(
            name="people_append",
            argv=_cli_argv("people_append.py"),
            summary=(
                "Append Facts/Appearances/tags to a person note. "
                "Required: --speaker. Optional: --fact TEXT (repeatable), "
                "--fact-file @note, --appearance, --tag, --alias, --summary, "
                "--source [[wikilink]]. Never overwrites ## Notes. "
                "Confirm durable facts with the user before writing."
            ),
        ),
        "speaker_enroll": CliTool(
            name="speaker_enroll",
            argv=_cli_argv("speaker_enroll.py"),
            summary=(
                "Manually approve a voiceprint into the Speakers gallery. "
                "Required: --speaker NAME. Provide either "
                "--session ID --from LABEL or --audio PATH. "
                "ALWAYS confirm with the user that the label/span is correct "
                "and audio quality is good before calling. Never auto-enroll."
            ),
        ),
        "session_reidentify": CliTool(
            name="session_reidentify",
            argv=_cli_argv("session_reidentify.py"),
            summary=(
                "Re-match an existing session against the Speakers gallery "
                "(no re-diarization). Required: --session-id ID. "
                "Use after speaker_enroll or gallery edits. "
                "Never invent session ids."
            ),
        ),
        "note_read": CliTool(
            name="note_read",
            argv=_cli_argv("note_read.py"),
            summary=(
                "Read a Markdown note from the vault. Required: --path KEY. "
                "Optional: --follow-links N to also include [[wiki-linked]] "
                "notes up to depth N."
            ),
        ),
        "note_search": CliTool(
            name="note_search",
            argv=_cli_argv("note_search.py"),
            summary=(
                "List vault notes. Optional: --folder PREFIX --tag TAG "
                "--limit N. Defaults to listing under the agent root (Pawn/)."
            ),
        ),
        "note_write": CliTool(
            name="note_write",
            argv=_cli_argv("note_write.py"),
            summary=(
                "Create or overwrite a vault Markdown note. Required: --path. "
                "Body: --content-file @note (preferred) or short --content. "
                "Writes only under Pawn/ unless the note has pawn: editable. "
                "Never touch .obsidian/. Refuses a checklist standing in for "
                "tasks (Pawn/Boards/..., or a page of boxes). Those belong to "
                "tasknotes_commit and tasknotes_board."
            ),
        ),
        "note_append": CliTool(
            name="note_append",
            argv=_cli_argv("note_append.py"),
            summary=(
                "Append Markdown to a vault note. Required: --path. "
                "Body: --content-file @note (preferred) or short --content. "
                "Same write guards as note_write."
            ),
        ),
        "task_update": CliTool(
            name="task_update",
            argv=_cli_argv("task_update.py"),
            summary=(
                "Update a vault task note. Required: --task-id. "
                "Optional: --status todo|running|review|done|blocked "
                "--result-file @note / --result TEXT. "
                "Use --status review with a result when finishing a vault task."
            ),
        ),
        "schedule_propose": CliTool(
            name="schedule_propose",
            argv=_cli_argv("schedule_propose.py"),
            summary=(
                "Propose a schedule create/update/pause/resume/cancel. "
                "Flags: --action ACTION --schedule-id ID --name TEXT "
                "--prompt TEXT --schedule JSON --timezone TZ --model MODEL "
                "--rationale TEXT. Does NOT apply changes."
            ),
        ),
        "knowledge_search": CliTool(
            name="knowledge_search",
            argv=_cli_argv("knowledge_search.py"),
            summary=(
                "Semantic search across vault notes, transcripts, and coworker items. "
                "Flags: --query TEXT --kind note|transcript|analysis|item --limit N. "
                "Use when the user asks where something was discussed."
            ),
        ),
        "queue_push": CliTool(
            name="queue_push",
            argv=_cli_argv("queue_push.py"),
            summary=(
                "Publish a notification / progress message to a named queue. "
                "Flags: --target NAME --command CMD --payload JSON. "
                "Payload must be a JSON object and must not include 'command'."
            ),
        ),
        "tasknotes_list": CliTool(
            name="tasknotes_list",
            argv=_cli_argv("tasknotes_list.py"),
            summary=(
                "List TaskNotes tasks grouped by person. Done tasks are hidden. "
                "Flags: --assignee NAME --mine --project NAME --status open|in-progress|done "
                "--include-done --undated --scheduled-from YYYY-MM-DD --scheduled-to YYYY-MM-DD "
                "--limit N. Call this before proposing or creating tasks. "
                "Notes under TaskNotes/Tasks are read-only."
            ),
        ),
        "tasknotes_propose": CliTool(
            name="tasknotes_propose",
            argv=_cli_argv("tasknotes_propose.py"),
            summary=(
                "Write a TaskNotes pick-list and do NOT create tasks. "
                "Required: --items JSON or --items-file @note. "
                "Optional: --title TEXT --boards none|assignee|project|both. "
                "JSON items use id, title, details, assignee, due, scheduled, project, "
                "priority, status, time_estimate, source, source_note, blocked_by, contexts. "
                "Stop after this unless the user already asked to create the tasks."
            ),
        ),
        "tasknotes_commit": CliTool(
            name="tasknotes_commit",
            argv=_cli_argv("tasknotes_commit.py"),
            summary=(
                "Create TaskNotes tasks from a pick-list or from JSON. "
                "Flags: --proposal PATH (omit to use the latest open pick-list) "
                "--items JSON / --items-file @note --pick 1,3 --all --force "
                "--boards none|assignee|project|both. "
                "Checked lines are created. --pick overrides checkboxes. "
                "--all includes unchecked lines. Same title+assignee is skipped "
                "unless --force. Does not call the TaskNotes HTTP API."
            ),
        ),
        "tasknotes_update": CliTool(
            name="tasknotes_update",
            argv=_cli_argv("tasknotes_update.py"),
            summary=(
                "Update one TaskNotes task. The file name stays so links keep working. "
                "Required: --id PATH|PAWN_ID|UNIQUE_TITLE. "
                "Optional: --title --status open|in-progress|done --priority "
                "--assignee --due YYYY-MM-DD --scheduled YYYY-MM-DD or YYYY-MM-DDTHH:MM "
                "--project --details --estimate MINUTES "
                "--clear-due --clear-scheduled --clear-assignee --clear-project --clear-estimate. "
                "--status done sets completedDate. Refuses notes outside the agent root."
            ),
        ),
        "tasknotes_board": CliTool(
            name="tasknotes_board",
            argv=_cli_argv("tasknotes_board.py"),
            summary=(
                "Write a TaskNotes .base board (kanban + calendar). "
                "Required: --name. Optional: --assignee --project "
                "--group-by status|assignee|priority --swimlane assignee|priority. "
                "A board file the user customized (pawn marker removed) is left alone. "
                "Open the .base file in Obsidian."
            ),
        ),
        "goal_propose": CliTool(
            name="goal_propose",
            argv=_cli_argv("goal_propose.py"),
            summary=(
                "Draft one goal thread into Pawn/Reviews/goal-proposal.md. "
                "Required: --name. Optional: --why --movement --interrupt --note --do "
                "--status active|parked. Does NOT write Goals.md. "
                "Tell the user to run /goal apply or use Apply goals proposal."
            ),
        ),
    }

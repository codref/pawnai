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
        "session_delete": CliTool(
            name="session_delete",
            argv=_cli_argv("session_delete.py"),
            summary=(
                "Permanently delete one diarization session from PostgreSQL "
                "(segments, analyses, session_state, graph triples). "
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
                "the display name to speaker_names / embeddings / session_state. "
                "Required: --session-id ID --from LABEL --to NAME. "
                "--from may be SPEAKER_XX or a current display name "
                "(e.g. --from SPEAKER_00 --to Davide). "
                "Existing vault transcript notes refresh automatically; pass "
                "--push-vault to create/update even without a prior mapping. "
                "Use when the user asks to change / correct / rename a speaker "
                "on a session. Never invent session ids — sessions_list first."
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
                "Never touch .obsidian/."
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
        "queue_push": CliTool(
            name="queue_push",
            argv=_cli_argv("queue_push.py"),
            summary=(
                "Publish a notification / progress message to a named queue. "
                "Flags: --target NAME --command CMD --payload JSON. "
                "Payload must be a JSON object and must not include 'command'."
            ),
        ),
    }

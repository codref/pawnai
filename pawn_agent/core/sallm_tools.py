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
                "Optional: --save --title TEXT — ONLY when the user asks to "
                "save to SiYuan. Analyze one session per invocation."
            ),
        ),
        "siyuan_save": CliTool(
            name="siyuan_save",
            argv=_cli_argv("siyuan_save.py"),
            summary=(
                "Save Markdown to SiYuan Notes. MUST be invoked inside a ```run "
                "fence, never as plain text. "
                "Preferred after analysis: "
                "siyuan_save --session-id ID --from-analysis "
                "(loads stored DB analysis; do NOT paste the summary). "
                "Alternatives: --content-file PATH, or short --content TEXT only. "
                "Optional: --title TEXT --path PATH."
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

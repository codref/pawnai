"""sessions_list — list diarization conversation sessions."""

from __future__ import annotations

import argparse
import sys

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.list_sessions import list_sessions_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sessions_list",
        description=(
            "List diarization conversation sessions from the database "
            "(newest first). Use before session_transcript when the id is unknown."
        ),
    )
    parser.add_argument(
        "--name-filter",
        default="",
        help="Case-insensitive substring match against session_id",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of sessions to return (default 10)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to pawnai.yaml",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_agent_config(args.config)
        print_out(list_sessions_impl(cfg, name_filter=args.name_filter, limit=args.limit))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

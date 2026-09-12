"""session_transcript — fetch one session's transcript."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.query_conversation import query_conversation_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="session_transcript",
        description=(
            "Fetch and print the full transcript for a diarization session. "
            "Do not invent session ids — list first when unsure."
        ),
    )
    parser.add_argument(
        "session_id_pos",
        nargs="?",
        default=None,
        help="Diarization session id (same as --session-id)",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Diarization session identifier stored in the database",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    session_id = args.session_id or args.session_id_pos
    if not session_id:
        return fail("session id required: pass --session-id ID (or a bare ID)")

    try:
        cfg = load_agent_config(args.config)
        print_out(query_conversation_impl(cfg, session_id))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

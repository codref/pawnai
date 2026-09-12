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
        "--session-id",
        required=True,
        help="Diarization session identifier stored in the database",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    try:
        cfg = load_agent_config(args.config)
        print_out(query_conversation_impl(cfg, args.session_id))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

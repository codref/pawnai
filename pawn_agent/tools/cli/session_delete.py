"""session_delete — permanently delete one diarization session from the DB."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.delete_session import delete_session_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="session_delete",
        description=(
            "Permanently delete diarization DB data for one session "
            "(segments, analyses, session_state, graph triples). "
            "Requires --confirm to exactly match --session-id. "
            "Ask the user to confirm the exact session name in chat first. "
            "Never invent session ids."
        ),
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="Diarization session identifier to delete",
    )
    parser.add_argument(
        "--confirm",
        required=True,
        help="Must exactly equal --session-id (confirmation gate)",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    try:
        cfg = load_agent_config(args.config)
        print_out(
            delete_session_impl(cfg, session_id=args.session_id, confirm=args.confirm)
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

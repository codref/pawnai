"""session_screenshots — list or summarize screenshots for one session."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.session_screenshots import (
    list_session_screenshots,
    summarize_session_screenshots_impl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="session_screenshots",
        description=(
            "List screenshots captured during a diarization session. "
            "--summarize asks the background vision model to describe changes. "
            "--id limits that to one screenshot."
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
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Fill summaries for screenshots that do not have one yet",
    )
    parser.add_argument(
        "--id",
        default=None,
        help="With --summarize, only this screenshot id",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    session_id = args.session_id or args.session_id_pos
    if not session_id:
        return fail("session id required: pass --session-id ID (or a bare ID)")

    try:
        cfg = load_agent_config(args.config)
        if args.summarize:
            print_out(summarize_session_screenshots_impl(cfg, session_id, only_id=args.id))
        else:
            print_out(list_session_screenshots(cfg, session_id))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

"""session_analyze — standard structured analysis for one session."""

from __future__ import annotations

import argparse
import asyncio

from pawn_agent.tools.analyze_summary import analyze_summary_impl
from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="session_analyze",
        description=(
            "Run the standard structured analysis (Title / Summary / Topics / …) "
            "and persist it. Optional --save also writes analysis Markdown to the vault."
        ),
    )
    # Positional is accepted because models often omit the flag name.
    parser.add_argument(
        "session_id_pos",
        nargs="?",
        default=None,
        help="Diarization session id (same as --session-id)",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Diarization session id (preferred flag form)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Also save analysis Markdown to the vault (DB save always)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title hint when --save is set",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    session_id = args.session_id or args.session_id_pos
    if not session_id:
        return fail("session id required: pass --session-id ID (or a bare ID)")

    try:
        cfg = load_agent_config(args.config)
        # analyze_summary_impl is async (LLM analysis); CliTool runner is sync.
        text = asyncio.run(
            analyze_summary_impl(
                cfg,
                session_id,
                save=bool(args.save),
                title=args.title,
            )
        )
        print_out(text)
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

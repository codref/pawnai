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
            "and persist it. Optional --save also writes to SiYuan."
        ),
    )
    parser.add_argument("--session-id", required=True, help="Diarization session id")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Also save the analysis report to SiYuan Notes",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional SiYuan document title when --save is set",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    try:
        cfg = load_agent_config(args.config)
        # analyze_summary_impl is async (LLM analysis); CliTool runner is sync.
        text = asyncio.run(
            analyze_summary_impl(
                cfg,
                args.session_id,
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

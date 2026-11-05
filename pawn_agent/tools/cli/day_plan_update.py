"""day_plan_update — replace ## Plan on Pawn/Daily/{date}.md."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.cli._content import read_content_file, reject_unsafe_inline_content
from pawn_agent.tools.day_plan_impl import day_plan_update_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="day_plan_update",
        description=(
            "Replace the ## Plan section on a coworker daily note "
            "(Pawn/Daily/{date}.md). Creates a stub note if missing."
        ),
    )
    parser.add_argument(
        "--date",
        default="",
        help="YYYY-MM-DD (default: today in coworker.timezone)",
    )
    parser.add_argument("--content", default=None, help="Short plan Markdown")
    parser.add_argument(
        "--content-file",
        default=None,
        help="Path to UTF-8 Markdown (@plan temp file)",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    if not args.content and not args.content_file:
        return fail("provide --content or --content-file")
    if args.content and args.content_file:
        return fail("use only one of --content or --content-file")

    try:
        if args.content_file:
            content = read_content_file(args.content_file)
        else:
            content = args.content or ""
            bad = reject_unsafe_inline_content(content)
            if bad:
                return fail(bad)

        cfg = load_agent_config(args.config)
        print_out(
            day_plan_update_impl(
                cfg,
                content=content,
                date=args.date or None,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

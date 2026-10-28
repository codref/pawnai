"""event_prep_update — replace ## Prep on a pawn: event note."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.cli._content import read_content_file, reject_unsafe_inline_content
from pawn_agent.tools.event_prep_impl import event_prep_update_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="event_prep_update",
        description=(
            "Replace the ## Prep section on a planner event note. "
            "Does not rewrite the rest of the note body."
        ),
    )
    parser.add_argument("--event-id", default="", help="Event id (evt_…)")
    parser.add_argument("--path", default="", help="Vault key of the event note")
    parser.add_argument("--content", default=None, help="Short prep text")
    parser.add_argument(
        "--content-file",
        default=None,
        help="Path to UTF-8 Markdown (@prep temp file)",
    )
    parser.add_argument("--job-id", default="", help="Optional meeting_prep job id")
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    if not args.content and not args.content_file:
        return fail("provide --content or --content-file")
    if args.content and args.content_file:
        return fail("use only one of --content or --content-file")
    if not args.event_id and not args.path:
        return fail("provide --event-id or --path")

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
            event_prep_update_impl(
                cfg,
                event_id=args.event_id or "",
                path=args.path or "",
                content=content,
                job_id=args.job_id or "",
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

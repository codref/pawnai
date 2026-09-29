"""tasknotes_list — show TaskNotes tasks grouped by person."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.tasknotes_impl import tasknotes_list_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tasknotes_list",
        description=(
            "List task notes Pawn can see, grouped by assignee. "
            "Done tasks are hidden unless --include-done or --status done. "
            "Notes in the TaskNotes plugin folder are read-only."
        ),
    )
    parser.add_argument("--assignee", default="", help="Filter to one person (me is the user)")
    parser.add_argument("--mine", action="store_true", help="Only the configured display name")
    parser.add_argument("--project", default="", help="Filter by project name")
    parser.add_argument("--status", default="", help="open, in-progress, or done")
    parser.add_argument("--include-done", action="store_true", help="Include completed tasks")
    parser.add_argument(
        "--undated", action="store_true", help="Only tasks with no due and no scheduled"
    )
    parser.add_argument(
        "--scheduled-from", default="", help="YYYY-MM-DD inclusive (due or scheduled)"
    )
    parser.add_argument(
        "--scheduled-to", default="", help="YYYY-MM-DD inclusive (due or scheduled)"
    )
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        print_out(
            tasknotes_list_impl(
                cfg,
                assignee=args.assignee,
                mine=bool(args.mine),
                project=args.project,
                status=args.status,
                include_done=bool(args.include_done),
                undated=bool(args.undated),
                scheduled_from=args.scheduled_from,
                scheduled_to=args.scheduled_to,
                limit=args.limit,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

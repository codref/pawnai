"""tasknotes_board — write a TaskNotes kanban and calendar .base file."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.tasknotes_impl import tasknotes_board_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tasknotes_board",
        description=(
            "Write a .base file with a kanban and a calendar. "
            "Filter by assignee and/or project. A file whose first line is no longer "
            "the pawn marker is left unchanged."
        ),
    )
    parser.add_argument("--name", required=True, help="Board title, also the file name")
    parser.add_argument("--assignee", default="", help="Only this person. me is the user")
    parser.add_argument("--project", default="", help="Only this project")
    parser.add_argument(
        "--group-by",
        default="status",
        help="Kanban columns: status, assignee, or priority",
    )
    parser.add_argument(
        "--swimlane",
        default="",
        help="Optional row grouping: assignee or priority",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        print_out(
            tasknotes_board_impl(
                cfg,
                name=args.name,
                assignee=args.assignee,
                project=args.project,
                group_by=args.group_by,
                swimlane=args.swimlane,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

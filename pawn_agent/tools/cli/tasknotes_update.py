"""tasknotes_update — change one TaskNotes task without renaming the file."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.tasknotes_impl import tasknotes_update_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tasknotes_update",
        description=(
            "Update one task by path, pawn id (or a unique prefix), or unique title. "
            "The file name is left unchanged so links keep working. "
            "Notes outside the agent root are not rewritten."
        ),
    )
    parser.add_argument("--id", required=True, help="Path, pawn id, or unique title")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--status", default=None, help="open, in-progress, done (done sets completedDate)"
    )
    parser.add_argument("--priority", default=None, help="none, low, normal, high")
    parser.add_argument(
        "--assignee", default=None, help="Person. me resolves to the configured name"
    )
    parser.add_argument("--due", default=None, help="YYYY-MM-DD, or empty to clear")
    parser.add_argument(
        "--scheduled",
        default=None,
        help="YYYY-MM-DD or YYYY-MM-DDTHH:MM local wall time, no timezone suffix",
    )
    parser.add_argument("--project", default=None)
    parser.add_argument(
        "--details", default=None, help="Replace the note body; keeps a Source line"
    )
    parser.add_argument("--estimate", default=None, help="Minutes")
    parser.add_argument("--clear-due", action="store_true")
    parser.add_argument("--clear-scheduled", action="store_true")
    parser.add_argument("--clear-assignee", action="store_true")
    parser.add_argument("--clear-project", action="store_true")
    parser.add_argument("--clear-estimate", action="store_true")
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        print_out(
            tasknotes_update_impl(
                cfg,
                ident=args.id,
                title=args.title,
                status=args.status,
                priority=args.priority,
                assignee=args.assignee,
                due=args.due,
                scheduled=args.scheduled,
                project=args.project,
                details=args.details,
                estimate=args.estimate,
                clear_due=bool(args.clear_due),
                clear_scheduled=bool(args.clear_scheduled),
                clear_assignee=bool(args.clear_assignee),
                clear_project=bool(args.clear_project),
                clear_estimate=bool(args.clear_estimate),
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

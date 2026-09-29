"""tasknotes_commit — create TaskNotes tasks from a pick-list or JSON."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.cli._content import read_content_file
from pawn_agent.tools.tasknotes_impl import tasknotes_commit_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tasknotes_commit",
        description=(
            "Create TaskNotes tasks. From a pick-list, only checked lines are created "
            "unless --pick or --all is set. --pick overrides checkboxes. "
            "With no --proposal and no items, the latest open pick-list is used."
        ),
    )
    parser.add_argument("--proposal", default="", help="Vault path of a pick-list note")
    parser.add_argument("--items", default=None, help="JSON list or object (direct create)")
    parser.add_argument("--items-file", default=None, help="UTF-8 JSON file (@note)")
    parser.add_argument(
        "--pick",
        default="",
        help="Comma-separated item ids to create, including unchecked ones",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create every item, including unchecked lines",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create another copy even when title and assignee already match a task",
    )
    parser.add_argument(
        "--boards",
        default="",
        help="none, assignee, project, or both. Empty uses the pick-list's saved intent.",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)
    if args.items and args.items_file:
        return fail("use only one of --items or --items-file")
    if (args.items or args.items_file) and args.proposal:
        return fail("pass a proposal or items, not both")
    document = ""
    if args.items_file:
        document = read_content_file(args.items_file)
    elif args.items:
        document = args.items
    try:
        cfg = load_agent_config(args.config)
        print_out(
            tasknotes_commit_impl(
                cfg,
                proposal=args.proposal,
                document=document,
                pick=args.pick,
                take_all=bool(args.all),
                force=bool(args.force),
                boards=args.boards,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

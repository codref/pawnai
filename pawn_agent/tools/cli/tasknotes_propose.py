"""tasknotes_propose — write a TaskNotes pick-list without creating tasks."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.cli._content import read_content_file
from pawn_agent.tools.tasknotes_impl import tasknotes_propose_impl


def _document(items: str | None, items_file: str | None) -> str | None:
    if items and items_file:
        return None
    if items_file:
        return read_content_file(items_file)
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tasknotes_propose",
        description=(
            "Write a TaskNotes pick-list the user can edit. Does not create tasks. "
            "JSON is a list, or {title, boards, items:[{id, title, details, assignee, "
            "due, scheduled, project, priority, status, time_estimate, source, "
            "source_note, blocked_by, contexts}]}."
        ),
    )
    parser.add_argument("--title", default="", help="Heading on the pick-list")
    parser.add_argument("--items", default=None, help="JSON list or object (short lists)")
    parser.add_argument("--items-file", default=None, help="UTF-8 JSON file (@note)")
    parser.add_argument(
        "--boards",
        default="",
        help="Remember a board intent: none, assignee, project, or both",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)
    if args.items and args.items_file:
        return fail("use only one of --items or --items-file")
    document = _document(args.items, args.items_file)
    if not document:
        return fail("provide --items or --items-file")
    try:
        cfg = load_agent_config(args.config)
        print_out(
            tasknotes_propose_impl(
                cfg,
                document=document,
                title=args.title,
                boards=args.boards,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

"""task_update — update vault task note status and/or Result section."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.cli._content import read_content_file
from pawn_agent.tools.notes_impl import task_update_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="task_update",
        description=(
            "Update a task note under Pawn/Tasks/ (or vault.task_path_template). "
            "Sets frontmatter status and/or replaces the ## Result section."
        ),
    )
    parser.add_argument(
        "--task-id",
        required=True,
        help="Task id (maps to template) or full vault path",
    )
    parser.add_argument("--status", default=None, help="New frontmatter status (e.g. review)")
    parser.add_argument(
        "--result",
        default=None,
        help="Short inline Result body (one line)",
    )
    parser.add_argument(
        "--result-file",
        default=None,
        help="UTF-8 file for ## Result body (@note temp file)",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    if args.result and args.result_file:
        return fail("use only one of --result or --result-file")

    result_text: str | None = None
    if args.result_file:
        result_text = read_content_file(args.result_file)
    elif args.result is not None:
        result_text = args.result

    try:
        cfg = load_agent_config(args.config)
        print_out(
            task_update_impl(
                cfg,
                args.task_id,
                status=args.status,
                result=result_text,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

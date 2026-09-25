"""note_write — create or overwrite a vault Markdown note."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.cli._content import read_content_file, reject_unsafe_inline_content
from pawn_agent.tools.notes_impl import note_write_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="note_write",
        description=(
            "Write Markdown to the vault. Prefer --content-file @note + ```file note "
            "for long bodies; do NOT paste long Markdown in --content."
        ),
    )
    parser.add_argument("--path", required=True, help="Vault key to write")
    parser.add_argument(
        "--content",
        default=None,
        help="Short one-line Markdown only; use --content-file for real notes",
    )
    parser.add_argument(
        "--content-file",
        default=None,
        help="Path to a UTF-8 Markdown file (sallm @note temp file)",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Fail if the note already exists",
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
            note_write_impl(
                cfg,
                args.path,
                content,
                create_only=bool(args.create_only),
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

"""note_append — append Markdown to a vault note."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.cli._content import read_content_file, reject_unsafe_inline_content
from pawn_agent.tools.notes_impl import note_append_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="note_append",
        description="Append Markdown to a vault note (creates the note if missing).",
    )
    parser.add_argument("--path", required=True, help="Vault key")
    parser.add_argument("--content", default=None, help="Short one-line append text")
    parser.add_argument(
        "--content-file",
        default=None,
        help="Path to UTF-8 Markdown to append (@note temp file)",
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
        print_out(note_append_impl(cfg, args.path, content))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

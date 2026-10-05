"""note_search — list vault notes under a folder, optionally filter by tag."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.notes_impl import note_search_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="note_search",
        description="List .md keys under a vault folder; filter by frontmatter or #tag.",
    )
    parser.add_argument(
        "--folder",
        default="",
        help="Vault prefix to list (default: vault agent_root, usually Pawn/)",
    )
    parser.add_argument("--tag", default="", help="Tag name (with or without #)")
    parser.add_argument("--limit", type=int, default=50, help="Maximum paths to return")
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    try:
        cfg = load_agent_config(args.config)
        print_out(
            note_search_impl(
                cfg,
                folder=args.folder,
                tag=args.tag,
                limit=args.limit,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

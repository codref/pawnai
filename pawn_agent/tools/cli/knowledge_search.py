"""knowledge_search — semantic search across the vault and transcripts."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.knowledge_search import knowledge_search_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="knowledge_search",
        description="Search notes, transcripts, and coworker items by meaning.",
    )
    parser.add_argument("--query", required=True, help="What to look for")
    parser.add_argument(
        "--kind",
        default="",
        help="Optional source kind: note, transcript, analysis, item",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        print_out(
            knowledge_search_impl(cfg, args.query, kind=args.kind, limit=args.limit)
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

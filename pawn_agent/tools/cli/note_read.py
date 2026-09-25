"""note_read — read a vault Markdown note (optional wiki link follow)."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.notes_impl import note_read_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="note_read",
        description="Read one vault note by path; optionally follow [[wiki links]].",
    )
    parser.add_argument("--path", required=True, help="Vault key (e.g. Pawn/Tasks/abc.md)")
    parser.add_argument(
        "--follow-links",
        type=int,
        default=0,
        help="Follow [[wiki links]] up to this depth (default 0)",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    try:
        cfg = load_agent_config(args.config)
        print_out(note_read_impl(cfg, args.path, follow_links=max(0, args.follow_links)))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

"""board_read — a board as text, or one card in depth."""

from __future__ import annotations

import argparse

from pawn_agent.tools.board_impl import board_read_impl
from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="board_read", description="Read a brainmap board.")
    parser.add_argument("--board", required=True, help="Board id (from the board snapshot)")
    parser.add_argument("--node", default="", help="One card id: its evidence and strings")
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        print_out(board_read_impl(cfg, board=args.board, node=args.node))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

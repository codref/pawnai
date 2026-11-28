"""board_propose — propose a ghost string, note card, group or pin on a board."""

from __future__ import annotations

import argparse

from pawn_agent.tools.board_impl import board_propose_impl
from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.cli._content import read_content_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="board_propose",
        description="Propose a change to a board. The user accepts or burns it.",
    )
    parser.add_argument("--board", required=True)
    parser.add_argument("--kind", required=True, choices=["string", "card", "group", "pin"])
    parser.add_argument("--from", dest="frm", default="", help="string: first card id")
    parser.add_argument("--to", default="", help="string: second card id")
    parser.add_argument("--role", default="", help="string: mentions, evidence_for, …")
    parser.add_argument("--title", default="", help="card/group: title")
    parser.add_argument("--content", default="", help="card: short body")
    parser.add_argument("--content-file", default="", help="card: body from a ```file block")
    parser.add_argument("--source", action="append", default=[], help="card: a card it is about")
    parser.add_argument("--node", action="append", default=[], help="group/pin: card ids")
    parser.add_argument("--reason", default="", help="One line: why")
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)
    try:
        body = read_content_file(args.content_file) if args.content_file else args.content
        cfg = load_agent_config(args.config)
        print_out(
            board_propose_impl(
                cfg,
                board=args.board,
                kind=args.kind,
                frm=args.frm,
                to=args.to,
                role=args.role,
                title=args.title,
                body=body or "",
                sources=args.source,
                nodes=args.node,
                reason=args.reason,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

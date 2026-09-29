"""idea_capture — write one idea skeleton under the watch folder."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.ideas_impl import idea_capture_impl


def _questions(joined: str, repeated: list[str]) -> list[str]:
    items = list(repeated)
    if joined.strip():
        items.append(joined)
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="idea_capture",
        description=(
            "Write an idea skeleton from one user line. "
            "Does not edit the note again and does not write the Pawn/Ideas companion."
        ),
    )
    parser.add_argument("--title", default="", help="Short noun phrase taken from the line")
    parser.add_argument("--seed", default="", help="The user's line, unchanged")
    parser.add_argument("--why", default="", help="One or two sentences from that line")
    parser.add_argument("--sketch", default="", help="A short outline from that line")
    parser.add_argument("--question", action="append", default=[], help="One open question")
    parser.add_argument(
        "--open-questions",
        default="",
        help="Open questions, one per line",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        print_out(
            idea_capture_impl(
                cfg,
                title=args.title,
                seed=args.seed,
                why=args.why,
                sketch=args.sketch,
                open_questions=_questions(args.open_questions, args.question),
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

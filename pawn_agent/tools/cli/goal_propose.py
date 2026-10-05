"""goal_propose — draft a Goals.md proposal. Does not write Goals.md."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.goals_impl import goal_propose_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="goal_propose",
        description=(
            "Write Pawn/Reviews/goal-proposal.md with a ```goals fence. "
            "Does not change Goals.md. The user applies it with /goal apply."
        ),
    )
    parser.add_argument("--name", required=True, help="Thread title")
    parser.add_argument("--why", default="", help="Why this thread is on the list")
    parser.add_argument("--movement", default="", help="What done-enough looks like")
    parser.add_argument("--interrupt", default="", help="The only reason to notify")
    parser.add_argument("--note", default="", help="Parked-thread note link")
    parser.add_argument("--do", default="", help="Parked-thread instruction")
    parser.add_argument(
        "--status",
        default="active",
        help="active or parked",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        print_out(
            goal_propose_impl(
                cfg,
                name=args.name,
                why=args.why,
                movement=args.movement,
                interrupt=args.interrupt,
                note=args.note,
                do=args.do,
                status=args.status,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

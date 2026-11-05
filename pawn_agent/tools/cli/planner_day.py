"""planner_day — compact merged timeline for one local calendar day."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.day_plan_impl import planner_day_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="planner_day",
        description=(
            "Show events, sessions, and open items for one day "
            "(same merge as GET /v1/planner/day). Compact text for planning."
        ),
    )
    parser.add_argument(
        "--date",
        default="",
        help="YYYY-MM-DD (default: today in coworker.timezone)",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    try:
        cfg = load_agent_config(args.config)
        print_out(planner_day_impl(cfg, date=args.date or None))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

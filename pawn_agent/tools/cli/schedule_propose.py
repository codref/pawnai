"""schedule_propose — create a schedule-management proposal (not applied)."""

from __future__ import annotations

import argparse
import asyncio
import json

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.propose_schedule import propose_schedule_change_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="schedule_propose",
        description=(
            "Propose create/update/pause/resume/cancel for durable schedules. "
            "The application must approve the proposal before it applies."
        ),
    )
    parser.add_argument(
        "--action",
        required=True,
        help="create | update | pause | resume | cancel",
    )
    parser.add_argument("--schedule-id", default=None, help="Required for non-create actions")
    parser.add_argument("--name", default=None, help="Human-readable schedule name")
    parser.add_argument("--prompt", default=None, help="Future agent prompt for create/update")
    parser.add_argument(
        "--schedule",
        default=None,
        help=(
            "JSON object, e.g. " '\'{"schedule_kind":"once","run_at":"...","session_id":"..."}\''
        ),
    )
    parser.add_argument("--timezone", default=None, help="IANA timezone, e.g. UTC")
    parser.add_argument("--model", default=None, help="Optional per-schedule model override")
    parser.add_argument("--rationale", default=None, help="Why this matches the user request")
    parser.add_argument(
        "--proposed-by-session-id",
        default=None,
        help="Optional conversation session that created the proposal",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    schedule_obj = None
    if args.schedule:
        try:
            schedule_obj = json.loads(args.schedule)
        except json.JSONDecodeError as exc:
            return fail(f"invalid --schedule JSON: {exc}")
        if not isinstance(schedule_obj, dict):
            return fail("--schedule must be a JSON object")

    try:
        cfg = load_agent_config(args.config)
        text = asyncio.run(
            propose_schedule_change_impl(
                cfg,
                action=args.action,
                schedule_id=args.schedule_id,
                name=args.name,
                prompt=args.prompt,
                schedule=schedule_obj,
                timezone=args.timezone,
                model=args.model,
                rationale=args.rationale,
                proposed_by_session_id=args.proposed_by_session_id,
            )
        )
        print_out(text)
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

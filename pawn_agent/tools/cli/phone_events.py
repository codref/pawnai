"""phone_events — calls, SMS, notifications, location, status pushed by the phone."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.phone_impl import phone_events_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phone_events",
        description="List phone events the user shared with pawn-server, newest first.",
    )
    parser.add_argument(
        "--kind",
        default="",
        help="call, sms, notification, location or status (default: all)",
    )
    parser.add_argument("--since", default="24h", help="24h, 7d, 30m, 2w or an ISO date")
    parser.add_argument("--contact", default="", help="Name or phone number to filter on")
    parser.add_argument("--query", default="", help="Substring to look for in the text")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--device", default="", help="Device name or id (default: all)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        print_out(
            phone_events_impl(
                cfg,
                kind=args.kind,
                since=args.since,
                contact=args.contact,
                query=args.query,
                limit=args.limit,
                device=args.device,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

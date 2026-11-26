"""event_upsert — create or update a Day-tab planner activity."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.cli._content import read_content_file, reject_unsafe_inline_content
from pawn_agent.tools.event_upsert_impl import event_upsert_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="event_upsert",
        description=(
            "Create or upsert a Day-tab activity (Pawn/Events/*.md with "
            "pawn: event). Fingerprint upsert on meeting URL, source URL, "
            "or title+start. This is what appears on the Obsidian Day grid — "
            "not ## Plan on the daily note."
        ),
    )
    parser.add_argument("--title", required=True, help="Activity title")
    parser.add_argument(
        "--start",
        default="",
        help="Local start YYYY-MM-DD or YYYY-MM-DDTHH:MM (coworker.timezone)",
    )
    parser.add_argument(
        "--end",
        default="",
        help="Local end YYYY-MM-DDTHH:MM",
    )
    parser.add_argument(
        "--source",
        default="manual",
        help="outlook | manual | capture (default manual)",
    )
    parser.add_argument("--source-url", default="", help="Outlook/item deeplink")
    parser.add_argument(
        "--attendees",
        default="",
        help="Comma-separated attendee names",
    )
    parser.add_argument("--topics", default="", help="Comma-separated topics")
    parser.add_argument("--organizer", default="", help="Organizer name")
    parser.add_argument("--location", default="", help="Room / place")
    parser.add_argument("--meeting-url", default="", help="Teams/Zoom join URL")
    parser.add_argument("--session-id", default="", help="Linked diarization session")
    parser.add_argument("--event-id", default="", help="Existing evt_… id to update")
    parser.add_argument("--body", default=None, help="Short note body Markdown")
    parser.add_argument(
        "--body-file",
        default=None,
        help="Path to UTF-8 Markdown body (@note temp file)",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    if args.body is not None and args.body_file:
        return fail("use only one of --body or --body-file")

    try:
        body = ""
        if args.body_file:
            body = read_content_file(args.body_file)
        elif args.body is not None:
            bad = reject_unsafe_inline_content(args.body)
            if bad:
                return fail(bad)
            body = args.body

        cfg = load_agent_config(args.config)
        print_out(
            event_upsert_impl(
                cfg,
                title=args.title,
                start=args.start,
                end=args.end,
                source=args.source,
                source_url=args.source_url,
                attendees=args.attendees,
                topics=args.topics,
                organizer=args.organizer,
                location=args.location,
                meeting_url=args.meeting_url,
                session_id=args.session_id,
                body=body,
                event_id=args.event_id,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

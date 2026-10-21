"""people_append — append Facts / Appearances / tags on a person note."""

from __future__ import annotations

import argparse

from pawn_agent.core.people.notes import update_person_note
from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.cli._content import read_content_file
from pawn_diarize.core.speaker_gallery import SpeakerGallery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="people_append",
        description=(
            "Append durable facts (and optional appearance / tags / summary) "
            "to People/{speaker_id}.md. Never touches ## Notes. "
            "Prefer --fact / --fact-file over inventing biography."
        ),
    )
    parser.add_argument("--speaker", required=True, help="Gallery id or display name")
    parser.add_argument(
        "--fact",
        action="append",
        default=[],
        help="One fact bullet (repeatable)",
    )
    parser.add_argument(
        "--fact-file",
        default=None,
        help="File or @note with one fact per line",
    )
    parser.add_argument(
        "--appearance",
        action="append",
        default=[],
        help="Appearance bullet (usually a transcript wikilink)",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Topical tag without # (repeatable)",
    )
    parser.add_argument("--alias", action="append", default=[], help="Add alias")
    parser.add_argument("--summary", default=None, help="Replace ## Summary")
    parser.add_argument(
        "--source",
        default="",
        help="Optional source wikilink suffix for facts, e.g. [[Pawn/Transcripts/...]]",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        gallery = SpeakerGallery(cfg.db_dsn, config=cfg.speakers)
        sp = gallery.get_speaker(args.speaker) or gallery.find_speaker_by_name(args.speaker)
        if sp is None:
            return fail(f"Unknown gallery speaker: {args.speaker}")

        facts = list(args.fact or [])
        if args.fact_file:
            body = read_content_file(args.fact_file)
            facts.extend(line.strip() for line in body.splitlines() if line.strip())
        if args.source:
            facts = [f"{f} ({args.source})" if args.source not in f else f for f in facts]
        if (
            not facts
            and not args.appearance
            and not args.tag
            and not args.alias
            and not args.summary
        ):
            return fail("Provide at least one of --fact, --appearance, --tag, --alias, --summary")

        key = update_person_note(
            cfg,
            speaker_id=sp.id,
            display_name=sp.display_name,
            summary=args.summary,
            aliases=args.alias or None,
            tags=args.tag or None,
            add_facts=facts or None,
            add_appearances=args.appearance or None,
        )
        print_out(f"Updated {key}")
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

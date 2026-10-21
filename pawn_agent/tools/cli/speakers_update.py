"""speakers_update — set gallery aliases / short notes after user confirm."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_diarize.core.speaker_gallery import SpeakerGallery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="speakers_update",
        description=(
            "Update Speakers gallery card fields (aliases, short notes). "
            "Does not enroll voiceprints. Prefer people_append for vault bios."
        ),
    )
    parser.add_argument("--speaker", required=True, help="Speaker id or display name")
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Add an alias (repeatable). Merged with existing aliases.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Replace the short gallery notes card (empty string clears).",
    )
    parser.add_argument("--display-name", default=None, help="Rename display name")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        gallery = SpeakerGallery(cfg.db_dsn, config=cfg.speakers)
        sp = gallery.get_speaker(args.speaker) or gallery.find_speaker_by_name(args.speaker)
        if sp is None:
            return fail(f"Unknown speaker: {args.speaker}")
        aliases = list(sp.aliases or [])
        for a in args.alias or []:
            a = a.strip()
            if a and a.lower() not in {x.lower() for x in aliases}:
                aliases.append(a)
        updated = gallery.update_speaker(
            sp.id,
            aliases=aliases if args.alias else None,
            notes=args.notes,
            display_name=args.display_name,
        )
        print_out(
            f"Updated {updated.id}: name={updated.display_name} "
            f"aliases={updated.aliases!r} notes={updated.notes!r}"
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

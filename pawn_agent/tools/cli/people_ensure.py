"""people_ensure — create a stub People/{speaker_id}.md when missing."""

from __future__ import annotations

import argparse

from pawn_agent.core.people.notes import ensure_person_note, people_note_key
from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_diarize.core.speaker_gallery import SpeakerGallery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="people_ensure",
        description=(
            "Ensure a vault person stub exists for a gallery speaker. "
            "Never creates gallery people or voice enrollments."
        ),
    )
    parser.add_argument(
        "--speaker",
        required=True,
        help="Gallery speaker id or display name",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        gallery = SpeakerGallery(cfg.db_dsn, config=cfg.speakers)
        sp = gallery.get_speaker(args.speaker) or gallery.find_speaker_by_name(args.speaker)
        if sp is None:
            return fail(
                f"Unknown gallery speaker: {args.speaker}. "
                "Create with pawn-diarize speakers create first."
            )
        me_names = {n.strip().lower() for n in (cfg.coworker.me or []) if n.strip()}
        key, created = ensure_person_note(
            cfg,
            speaker_id=sp.id,
            display_name=sp.display_name,
            aliases=sp.aliases or [],
            me=sp.display_name.lower() in me_names,
        )
        if created:
            print_out(f"Created {key}")
        else:
            print_out(f"Already exists: {people_note_key(cfg, sp.id)}")
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

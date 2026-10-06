"""people_show — read a vault People/{speaker_id}.md bio."""

from __future__ import annotations

import argparse

from pawn_agent.core.people.notes import format_person_card, read_person_note
from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_diarize.core.speaker_gallery import SpeakerGallery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="people_show",
        description=(
            "Show the vault person bio for a gallery speaker. "
            "Path is People/{speaker_id}.md (stable across renames)."
        ),
    )
    parser.add_argument(
        "speaker",
        help="Speaker id or display name (resolved via Speakers gallery)",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        gallery = SpeakerGallery(cfg.db_dsn, config=cfg.speakers)
        sp = gallery.get_speaker(args.speaker) or gallery.find_speaker_by_name(args.speaker)
        sid = sp.id if sp else args.speaker.strip()
        note = read_person_note(cfg, sid)
        if note is None:
            return fail(
                f"No person note for {sid!r}. "
                "Use people_ensure after the speaker exists in the gallery."
            )
        print_out(format_person_card(note))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

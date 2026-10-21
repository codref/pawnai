"""speakers_list — list curated Speakers gallery people."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_diarize.core.speaker_gallery import SpeakerGallery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="speakers_list",
        description=(
            "List people in the curated Speakers gallery. "
            "Use before speaker_enroll / session_reidentify."
        ),
    )
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        gallery = SpeakerGallery(cfg.db_dsn, config=cfg.speakers)
        rows = gallery.list_speakers(include_inactive=bool(args.include_inactive))
        if not rows:
            print_out("No speakers in gallery.")
            return 0
        lines = []
        for sp in rows:
            n = len(gallery.list_enrollments(sp.id))
            lines.append(f"{sp.id}\t{sp.display_name}\tenrollments={n}\tactive={sp.active}")
        print_out("\n".join(lines))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

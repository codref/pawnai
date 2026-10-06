"""speakers_show — gallery person card (aliases, notes, enrollments)."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_diarize.core.speaker_gallery import SpeakerGallery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="speakers_show",
        description="Show one Speakers gallery person (aliases, notes, enrollments).",
    )
    parser.add_argument("speaker", help="Speaker id or display name")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        gallery = SpeakerGallery(cfg.db_dsn, config=cfg.speakers)
        sp = gallery.get_speaker(args.speaker) or gallery.find_speaker_by_name(args.speaker)
        if sp is None:
            return fail(f"Unknown speaker: {args.speaker}")
        lines = [
            f"{sp.display_name}  id={sp.id}  active={sp.active}",
        ]
        if sp.aliases:
            lines.append(f"aliases: {', '.join(sp.aliases)}")
        if sp.notes:
            lines.append(f"notes: {sp.notes}")
        # Hint the vault bio path (stable id-based key).
        people_dir = getattr(cfg.coworker, "people_dir", "People") or "People"
        lines.append(f"people_note: {people_dir.rstrip('/')}/{sp.id}.md")
        for enr in gallery.list_enrollments(sp.id):
            span = ""
            if enr.start_time is not None and enr.end_time is not None:
                span = f" [{enr.start_time:.1f}-{enr.end_time:.1f}s]"
            lines.append(
                f"enrollment {enr.id[:8]}… model={enr.embedding_model} "
                f"dim={enr.embedding_dim} dur={enr.duration:.1f}s "
                f"quality={enr.quality_score}{span}"
            )
        print_out("\n".join(lines))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

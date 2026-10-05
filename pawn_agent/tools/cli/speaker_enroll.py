"""speaker_enroll — approve a voiceprint into the curated Speakers gallery."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_diarize.core.diarization import DiarizationEngine
from pawn_diarize.core.speaker_gallery import SpeakerGallery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="speaker_enroll",
        description=(
            "Manually enroll a voiceprint for a gallery person. "
            "ALWAYS confirm with the user that the span/label is correct "
            "and audio quality is good before calling."
        ),
    )
    parser.add_argument("--speaker", required=True, help="Display name or gallery id")
    parser.add_argument("--session", default=None, help="Source session id")
    parser.add_argument("--from", dest="from_label", default=None, help="Local label")
    parser.add_argument("--audio", default=None, help="Clean enrollment audio path")
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    try:
        cfg = load_agent_config(args.config)
        gallery = SpeakerGallery(cfg.db_dsn, config=cfg.speakers)
        sp = gallery.get_speaker(args.speaker) or gallery.find_speaker_by_name(
            args.speaker
        )
        if sp is None:
            sp = gallery.create_speaker(args.speaker)

        engine = DiarizationEngine(
            device="auto",
            embedding_model=cfg.models.embedding_model,
            hf_token=cfg.models.hf_token,
        )

        if args.audio:
            import soundfile as sf

            embedding = engine.extract_embeddings(args.audio)
            source_audio = args.audio
            start = args.start if args.start is not None else 0.0
            end = args.end
            if end is None:
                end = float(sf.info(args.audio).duration)
            source_session = args.session
        elif args.session and args.from_label:
            import numpy as np
            from pawn_diarize.core.database import load_session_state, get_engine, init_db
            from sqlalchemy import select
            from sqlalchemy.orm import Session as OrmSession
            from pawn_core.database import TranscriptionSegment

            db_engine = get_engine(cfg.db_dsn)
            init_db(db_engine)
            prior, _, _, _ = load_session_state(args.session, db_engine)
            info = (prior or {}).get(args.from_label)
            if not info or info.get("embedding") is None:
                return fail(
                    f"No session_state embedding for label {args.from_label!r}"
                )
            embedding = np.asarray(info["embedding"], dtype="float32")
            source_audio = None
            start, end = args.start, args.end
            with OrmSession(db_engine) as db:
                rows = list(
                    db.scalars(
                        select(TranscriptionSegment)
                        .where(TranscriptionSegment.session_id == args.session)
                        .where(
                            TranscriptionSegment.original_speaker_label
                            == args.from_label
                        )
                        .order_by(TranscriptionSegment.start_time)
                    )
                )
                if rows:
                    source_audio = rows[0].audio_file
                    if start is None:
                        start = float(rows[0].start_time)
                    if end is None:
                        end = float(rows[-1].end_time)
            source_session = args.session
            # Warm extractor so model_id is known.
            engine._initialize_models()
        else:
            return fail("Provide --audio PATH or --session ID --from LABEL")

        result = gallery.enroll(
            sp,
            embedding,
            embedding_model=getattr(
                engine._extractor, "model_id", cfg.models.embedding_model
            ),
            source_session_id=source_session,
            source_audio_file=source_audio,
            start_time=start,
            end_time=end,
            force=bool(args.force),
        )
        print_out(
            f"Enrolled {result.display_name} ({result.enrollment_id}) "
            f"model={result.embedding_model} dim={result.embedding_dim} "
            f"dur={result.duration:.1f}s quality={result.quality_score}"
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

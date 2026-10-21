"""session_reidentify — rematch a session against the Speakers gallery."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_diarize.core.reidentify import reidentify_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="session_reidentify",
        description=(
            "Re-match speaker labels on one diarization session against the "
            "curated Speakers gallery. Does not re-run anonymous diarization. "
            "Use after speakers enroll / gallery edits. Never invent session ids."
        ),
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        result = reidentify_session(
            args.session_id,
            cfg.db_dsn,
            threshold=args.threshold,
            speakers_config=cfg.speakers,
            embedding_model=cfg.models.embedding_model,
            hf_token=cfg.models.hf_token,
        )
        print_out(result.summary())
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

"""session_relabel — rename a speaker across one diarization session."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.session_relabel import session_relabel_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="session_relabel",
        description=(
            "Rename a speaker across one diarization session. Updates "
            "transcript segments, speaker_names mappings (so future "
            "embedding matches use the new name), and session_state "
            "prior-speaker keys. --from may be SPEAKER_XX or a current "
            "display name. Never invent session ids."
        ),
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="Diarization session identifier",
    )
    parser.add_argument(
        "--from",
        dest="from_speaker",
        required=True,
        help="Current speaker label or display name (e.g. SPEAKER_00)",
    )
    parser.add_argument(
        "--to",
        dest="to_speaker",
        required=True,
        help="Correct human-readable name to apply (e.g. Davide)",
    )
    parser.add_argument(
        "--push-vault",
        action="store_true",
        help="Force vault Speakers+Transcript create/update (Annotations preserved)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to pawnai.yaml",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_agent_config(args.config)
        print_out(
            session_relabel_impl(
                cfg,
                session_id=args.session_id,
                from_speaker=args.from_speaker,
                to_speaker=args.to_speaker,
                push_vault=bool(args.push_vault),
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

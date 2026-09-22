"""siyuan_save — persist Markdown (or a stored session analysis) into SiYuan."""

from __future__ import annotations

import argparse
from pathlib import Path

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.save_to_siyuan import (
    save_analysis_to_siyuan_impl,
    save_to_siyuan_impl,
)

# Short one-liners only; long / multiline bodies must use --content-file (@note).
_CONTENT_MAX_CHARS = 240
_CONTENT_REJECT_MSG = (
    "Error: --content is limited to short one-line text "
    f"(max {_CONTENT_MAX_CHARS} chars, no newlines). "
    "For Markdown use --content-file @note with a ```file note block, "
    "or --from-analysis after session_analyze."
)


def _reject_unsafe_content(content: str) -> str | None:
    """Return an error message if *content* is too long or multiline."""
    if "\n" in content or "\r" in content:
        return _CONTENT_REJECT_MSG
    if len(content) > _CONTENT_MAX_CHARS:
        return _CONTENT_REJECT_MSG
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="siyuan_save",
        description=(
            "Save Markdown to SiYuan Notes. "
            "Prefer --from-analysis after session_analyze (loads DB row; no paste). "
            "For free-form Markdown use --content-file (e.g. @note + ```file note). "
            "Do NOT put long Markdown in --content."
        ),
    )
    parser.add_argument("--session-id", required=True, help="Parent session id in SiYuan tree")
    parser.add_argument(
        "--from-analysis",
        action="store_true",
        help="Load the latest session_analysis row for --session-id and save that",
    )
    parser.add_argument(
        "--content",
        default=None,
        help=(
            f"Short one-line Markdown only (max {_CONTENT_MAX_CHARS} chars; "
            "no newlines). Prefer --from-analysis or --content-file."
        ),
    )
    parser.add_argument(
        "--content-file",
        default=None,
        help="Path to a UTF-8 Markdown file (use @name with a ```file name block)",
    )
    parser.add_argument("--title", default=None, help="Optional document title")
    parser.add_argument("--path", default=None, help="Optional explicit SiYuan path override")
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    sources = sum(bool(x) for x in (args.from_analysis, args.content, args.content_file))
    if sources == 0:
        return fail("provide --from-analysis, --content, or --content-file")
    if sources > 1:
        return fail("use only one of --from-analysis, --content, or --content-file")

    try:
        cfg = load_agent_config(args.config)
        if args.from_analysis:
            print_out(
                save_analysis_to_siyuan_impl(
                    cfg,
                    args.session_id,
                    title=args.title,
                    path=args.path,
                )
            )
            return 0

        if args.content_file:
            content = Path(args.content_file).expanduser().read_text(encoding="utf-8")
        else:
            content = args.content or ""
            bad = _reject_unsafe_content(content)
            if bad:
                return fail(bad)

        print_out(
            save_to_siyuan_impl(
                cfg,
                session_id=args.session_id,
                content=content,
                title=args.title,
                path=args.path,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

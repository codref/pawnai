"""siyuan_save — persist Markdown (or a stored session analysis) into SiYuan."""

from __future__ import annotations

import argparse
from pathlib import Path

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.save_to_siyuan import (
    save_analysis_to_siyuan_impl,
    save_to_siyuan_impl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="siyuan_save",
        description=(
            "Save Markdown to SiYuan Notes. "
            "Prefer --from-analysis after session_analyze (loads DB row; no paste). "
            "Do NOT put long Markdown in --content — use --from-analysis or "
            "--content-file instead."
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
        help="Short Markdown only (avoid for summaries — use --from-analysis)",
    )
    parser.add_argument(
        "--content-file",
        default=None,
        help="Path to a UTF-8 Markdown file",
    )
    parser.add_argument("--title", default=None, help="Optional document title")
    parser.add_argument("--path", default=None, help="Optional explicit SiYuan path override")
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    sources = sum(
        bool(x) for x in (args.from_analysis, args.content, args.content_file)
    )
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

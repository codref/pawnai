"""siyuan_append — append Markdown under a SiYuan parent block (append-only)."""

from __future__ import annotations

from pathlib import Path

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.siyuan_blocks import siyuan_append_impl

_CONTENT_MAX_CHARS = 240
_CONTENT_REJECT_MSG = (
    "Error: --content is limited to short one-line text "
    f"(max {_CONTENT_MAX_CHARS} chars, no newlines). "
    "For Markdown use --content-file @note with a ```file note block."
)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="siyuan_append",
        description=(
            "Append Markdown under a SiYuan parent block. "
            "Prefer --content-file @note for long bodies."
        ),
    )
    parser.add_argument("--parent-id", required=True, help="Parent document or block id")
    parser.add_argument("--content", default=None, help="Short one-line Markdown only")
    parser.add_argument(
        "--content-file",
        default=None,
        help="Path to UTF-8 Markdown (use @name with a ```file name block)",
    )
    parser.add_argument(
        "--as-result",
        action="store_true",
        help="Wrap content in the standard Pawn review checklist template",
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help="Request UUID (used when --as-result)",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    if not args.content and not args.content_file:
        return fail("provide --content or --content-file")
    if args.content and args.content_file:
        return fail("use only one of --content or --content-file")

    try:
        cfg = load_agent_config(args.config)
        if args.content_file:
            content = Path(args.content_file).expanduser().read_text(encoding="utf-8")
        else:
            content = args.content or ""
            if "\n" in content or "\r" in content or len(content) > _CONTENT_MAX_CHARS:
                return fail(_CONTENT_REJECT_MSG)
        print_out(
            siyuan_append_impl(
                cfg,
                parent_id=args.parent_id,
                content=content,
                as_result=args.as_result,
                request_id=args.request_id,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

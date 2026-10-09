"""capture_update — merge research-capture frontmatter via parse/dump."""

from __future__ import annotations

import argparse

from pawn_agent.tools.captures_impl import capture_update_impl
from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="capture_update",
        description=(
            "Update frontmatter on a research capture inbox note. "
            "Uses YAML dump so values with ':' stay valid for Obsidian. "
            "Does not rewrite the snippet body."
        ),
    )
    parser.add_argument("--path", required=True, help="Vault key of the capture note")
    parser.add_argument("--collection", default=None, help="Collection slug (slugified)")
    parser.add_argument("--entity", default=None, help="Entity slug (slugified)")
    parser.add_argument(
        "--type", dest="cap_type", default=None, help="Capture type (still, quote, …)"
    )
    parser.add_argument("--caption", default=None, help="Caption / description text")
    parser.add_argument(
        "--proposed-tags",
        default=None,
        help="Comma-separated tags or JSON list",
    )
    parser.add_argument(
        "--status",
        default=None,
        help="inbox | proposed | filed | ignored",
    )
    parser.add_argument(
        "--enriched-at", default=None, help="ISO timestamp (default: now when enriching)"
    )
    parser.add_argument("--hint", default=None, help="Optional user hint")
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    try:
        cfg = load_agent_config(args.config)
        print_out(
            capture_update_impl(
                cfg,
                args.path,
                collection=args.collection,
                entity=args.entity,
                type=args.cap_type,
                caption=args.caption,
                proposed_tags=args.proposed_tags,
                status=args.status,
                enriched_at=args.enriched_at,
                hint=args.hint,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

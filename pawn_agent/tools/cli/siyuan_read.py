"""siyuan_read — fetch SiYuan block kramdown / children / attrs / refs."""

from __future__ import annotations

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.siyuan_blocks import siyuan_read_impl


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="siyuan_read",
        description="Read a SiYuan block (kramdown, optional children/attrs/refs).",
    )
    parser.add_argument("--block-id", required=True, help="SiYuan block id")
    parser.add_argument(
        "--include-children",
        action="store_true",
        help="List immediate child blocks",
    )
    parser.add_argument(
        "--include-attrs",
        action="store_true",
        help="Include custom block attributes",
    )
    parser.add_argument(
        "--resolve-refs",
        action="store_true",
        help="Fetch kramdown for ((block refs)) found in the block",
    )
    parser.add_argument("--max-ref-depth", type=int, default=1)
    parser.add_argument("--max-blocks", type=int, default=40)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        print_out(
            siyuan_read_impl(
                cfg,
                block_id=args.block_id,
                include_children=args.include_children,
                include_attrs=args.include_attrs,
                resolve_refs=args.resolve_refs,
                max_ref_depth=args.max_ref_depth,
                max_blocks=args.max_blocks,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

"""siyuan_set_status — set custom-agent-* attrs on a SiYuan block."""

from __future__ import annotations

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.siyuan_blocks import siyuan_set_status_impl


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="siyuan_set_status",
        description="Set allowed custom-agent-* attributes on a SiYuan block.",
    )
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--status", default=None, help="queued|claimed|running|review|done|…")
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--output-id", default=None)
    parser.add_argument("--source-hash", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        print_out(
            siyuan_set_status_impl(
                cfg,
                block_id=args.block_id,
                status=args.status,
                request_id=args.request_id,
                output_id=args.output_id,
                source_hash=args.source_hash,
                run_id=args.run_id,
            )
        )
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

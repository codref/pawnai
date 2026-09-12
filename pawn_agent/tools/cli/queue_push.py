"""queue_push — publish a notification / progress envelope to a named target."""

from __future__ import annotations

import argparse
import asyncio
import json

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.push_queue_message import push_queue_message_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="queue_push",
        description=(
            "Publish a JSON command-envelope to a named queue_producers target. "
            "Use only when the user explicitly asks to notify / publish / enqueue."
        ),
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Producer target name from pawnai.yaml (e.g. matrix)",
    )
    parser.add_argument(
        "--command",
        required=True,
        help="Consumer command string (e.g. run, process)",
    )
    parser.add_argument(
        "--payload",
        default="{}",
        help="JSON object payload without a 'command' key (default: {})",
    )
    parser.add_argument("--config", default=None, help="Optional path to pawnai.yaml")
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError as exc:
        return fail(f"invalid --payload JSON: {exc}")
    if not isinstance(payload, dict):
        return fail("--payload must be a JSON object")

    try:
        cfg = load_agent_config(args.config)
        text = asyncio.run(push_queue_message_impl(cfg, args.target, args.command, payload))
        print_out(text)
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

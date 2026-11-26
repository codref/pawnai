"""phone_devices — registered phones and what each one shares."""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out
from pawn_agent.tools.phone_impl import phone_devices_impl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phone_devices",
        description="List phones registered with the phone bridge.",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        print_out(phone_devices_impl(load_agent_config(args.config)))
        return 0
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

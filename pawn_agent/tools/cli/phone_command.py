"""phone_command — run a live tool on the user's phone and wait for the answer.

Registered as several CliTools (phone_location_now, phone_status_now,
phone_sms_send, phone_notify) whose argv fixes the first positional TOOL.
"""

from __future__ import annotations

import argparse

from pawn_agent.tools.cli._boot import fail, load_agent_config, print_out


def main(argv: list[str] | None = None) -> int:
    from pawn_server.core import device_commands  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="phone_command", description="Run a live phone tool.")
    parser.add_argument("tool", choices=sorted(device_commands.TOOLS))
    parser.add_argument("--to", default=None, help="sms_send: phone number")
    parser.add_argument("--body", default=None, help="sms_send: message text")
    parser.add_argument("--title", default=None, help="notify: title")
    parser.add_argument("--text", default=None, help="notify: text")
    parser.add_argument("--device", default="", help="Device name or id (needed with several phones)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        spec = device_commands.TOOLS[args.tool]
        wanted = set(spec.required) | set(spec.optional)
        tool_args = {
            key: getattr(args, key)
            for key in ("to", "body", "title", "text")
            if key in wanted and getattr(args, key) is not None
        }
        timeout = float(getattr(getattr(cfg, "devices", None), "command_timeout_s", 45.0))
        view = device_commands.run_command(
            cfg.db_dsn, args.tool, tool_args, device=args.device, timeout_s=timeout
        )
        text = device_commands.describe(view)
        print_out(text)
        return 0 if view["status"] in ("ok", "denied") else 1
    except device_commands.CommandError as exc:
        return fail(str(exc))
    except Exception as exc:
        return fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

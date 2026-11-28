"""phone_command — run a live tool on a phone or assistant and wait for the answer.

Registered as several CliTools (phone_location_now, phone_status_now,
phone_sms_send, phone_notify, assistant_speak, assistant_timer) whose argv
fixes the first positional TOOL. ``phone_*`` tools only consider phones;
``assistant_*`` tools pass ``--any-kind`` and default to ``--device auto``
(the assistant the user was last near, see docs/ASSISTANTS.md).
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
    parser.add_argument("--text", default=None, help="notify / speak: text")
    parser.add_argument("--seconds", default=None, help="timer_set: duration in seconds")
    parser.add_argument("--label", default=None, help="timer_set: what the timer is for")
    parser.add_argument(
        "--device", default=None, help="Device name or id, or 'auto' (the one the user is near)"
    )
    parser.add_argument(
        "--any-kind", action="store_true", help="Consider desktops too, not only phones"
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_agent_config(args.config)
        spec = device_commands.TOOLS[args.tool]
        wanted = set(spec.required) | set(spec.optional)
        tool_args = {
            key: getattr(args, key)
            for key in ("to", "body", "title", "text", "seconds", "label")
            if key in wanted and getattr(args, key) is not None
        }
        settings = getattr(cfg, "devices", None)
        timeout = float(getattr(settings, "command_timeout_s", 45.0))
        device = args.device if args.device is not None else ("auto" if args.any_kind else "")
        view = device_commands.run_command(
            cfg.db_dsn,
            args.tool,
            tool_args,
            device=device,
            timeout_s=timeout,
            fresh_s=float(getattr(settings, "presence_fresh_s", 600.0)),
            kind="" if args.any_kind else "phone",
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

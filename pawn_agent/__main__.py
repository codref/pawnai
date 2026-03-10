"""Entry point for pawn-agent CLI (``python -m pawn_agent`` or ``pawn-agent``)."""

import sys

from pawn_agent.cli.commands import app, console


def main() -> None:
    """Main entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Fatal error:[/red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

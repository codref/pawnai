"""Entry point for pawn-server CLI."""

import sys

from pawn_server.cli.commands import app, console


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)
    except Exception as exc:
        console.print(f"[red]Fatal error:[/red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

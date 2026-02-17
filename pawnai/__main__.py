"""Main entry point for the PawnAI CLI application.

This module serves as the single entrypoint for the CLI.
Run with: python -m pawnai [command] [options]
Or after installation: pawnai [command] [options]
"""

import sys
import os
import warnings
from typing import Optional

# Suppress NeMo warnings before any imports
os.environ["NEMO_LOGGING_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from .cli.commands import app
from .cli.utils import console


def main() -> None:
    """Main entry point function for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Operation cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]✗ Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

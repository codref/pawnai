"""CLI commands for pawn-agent."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

app = typer.Typer(
    name="pawn-agent",
    help="LLM-powered conversational agent for pawn-diarize sessions.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

from pawn_agent.utils.model_utils import _PYDANTIC_PREFIXES, _apply_model_override  # noqa: F401


@app.command()
def chat(
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML config file. Defaults to pawnai.yaml in cwd."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model string override (e.g. 'openai:gpt-4o', 'anthropic:claude-sonnet-4-5-20251001')."
    ),
    db_dsn: Optional[str] = typer.Option(
        None, "--db-dsn",
        help="PostgreSQL DSN. Overrides DATABASE_URL env var and config."
    ),
    trace_state: bool = typer.Option(
        False, "--trace-state",
        help="Attach the full LangGraph state as JSON to Phoenix spans for debugging.",
    ),
) -> None:
    """Start an interactive multi-turn [bold]CHAT[/bold] session (LangGraph).

    Uses the LangGraph router with fast/deep reply routing, session-aware
    tool nodes, artifact carry-forward, and optional Phoenix tracing.

    Type [bold]/exit[/bold] or [bold]/quit[/bold] to end, or press Ctrl-D / Ctrl-C.
    Type [bold]/reset[/bold] to clear the conversation state and start fresh.

    \b
    Examples
    --------
    pawn-agent chat
    pawn-agent chat --model openai:gpt-4o
    """
    import asyncio  # noqa: PLC0415

    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from rich.markdown import Markdown  # noqa: PLC0415
    from pawn_agent.core.langgraph_chat import run_langgraph_chat  # noqa: PLC0415

    cfg = load_config(config)
    if model:
        _apply_model_override(cfg, model)
    if db_dsn:
        cfg.db_dsn = db_dsn

    name = cfg.agent_name
    status = console.status(f"[dim]{name} is thinking…[/dim]")

    def _on_thinking() -> None:
        status.start()

    def _rich_emit(text: str) -> None:
        status.stop()
        console.print(f"\n[bold magenta]{name}[/bold magenta]")
        console.print(Markdown(text))
        console.print()

    console.print(
        f"\n[bold cyan]pawn-agent chat[/bold cyan] "
        f"[dim]model={cfg.pydantic_model}  agent={name}  mode=langgraph[/dim]\n"
        "[dim]Fast/deep routing, session-aware tool nodes, artifact carry-forward. "
        "Type /exit or /quit to end. /reset starts a fresh LangGraph state.[/dim]\n"
    )
    try:
        asyncio.run(
            run_langgraph_chat(
                cfg=cfg,
                emit=_rich_emit,
                on_thinking=_on_thinking,
                trace_full_state=trace_state,
            )
        )
    except Exception as exc:
        status.stop()
        console.print(f"[red]Agent error:[/red] {exc}")
        raise typer.Exit(1)

    console.print("\n[dim]Session ended.[/dim]")


@app.command(name="tools")
def list_tools() -> None:
    """List available agent tools and a brief description of each."""
    from rich.table import Table  # noqa: PLC0415
    from pawn_agent.tools import get_registry  # noqa: PLC0415

    table = Table(title="Available Tools", show_lines=True, show_header=True)
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Description")

    for i, (name, description) in enumerate(get_registry(), start=1):
        table.add_row(str(i), name, description)

    console.print(table)


@app.command()
def models() -> None:
    """List available Copilot models."""
    import asyncio  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415
    from copilot import CopilotClient  # noqa: PLC0415

    async def _list() -> list:
        client = CopilotClient()
        await client.start()
        try:
            return await client.list_models()
        finally:
            await client.stop()

    with console.status("[bold green]Fetching models…[/bold green]"):
        try:
            model_list = asyncio.run(_list())
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)

    table = Table(title="Available Copilot Models", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Policy", justify="center")
    table.add_column("Multiplier", justify="right")
    table.add_column("Reasoning efforts")

    for m in model_list:
        policy_state = m.policy.state if m.policy else "—"
        policy_color = {"enabled": "green", "disabled": "red"}.get(policy_state, "yellow")
        multiplier = f"{m.billing.multiplier:.2f}x" if m.billing is not None else "—"
        reasoning = ", ".join(m.supported_reasoning_efforts) if m.supported_reasoning_efforts else "—"
        table.add_row(
            m.id,
            m.name,
            f"[{policy_color}]{policy_state}[/{policy_color}]",
            multiplier,
            reasoning,
        )

    console.print(table)
    console.print(f"[dim]{len(model_list)} model(s)[/dim]")

"""CLI commands for pawn-agent."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

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
        None, "--config", "-c", help="Path to YAML config file. Defaults to pawnai.yaml in cwd."
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model string override (e.g. 'openai:gpt-4o', 'anthropic:claude-sonnet-4-5-20251001').",
    ),
    db_dsn: Optional[str] = typer.Option(
        None, "--db-dsn", help="PostgreSQL DSN. Overrides DATABASE_URL env var and config."
    ),
    otlp: Optional[str] = typer.Option(
        None,
        "--otlp",
        help="Optional OTLP/HTTP endpoint for sallm Tempo traces.",
    ),
    metrics_port: int = typer.Option(
        0,
        "--metrics-port",
        help="Optional Prometheus /metrics port (0 = off).",
    ),
) -> None:
    """Start an interactive multi-turn [bold]CHAT[/bold] session (sallm).

    Uses the durable sallm ReAct agent with skills and CliTools for
    sessions, SiYuan, schedules, and queue publish.

    Type [bold]/exit[/bold] or [bold]/quit[/bold] to end, or press Ctrl-D / Ctrl-C.
    Type [bold]/stats[/bold] for session metrics, [bold]/reset[/bold] to clear
    conversation state.

    \b
    Examples
    --------
    pawn-agent chat
    pawn-agent chat --model openai:gpt-4o
    """
    import asyncio  # noqa: PLC0415

    from rich.markdown import Markdown  # noqa: PLC0415

    from pawn_agent.core.sallm_session import run_sallm_chat  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    cfg = load_config(config)
    if model:
        _apply_model_override(cfg, model)
    if db_dsn:
        cfg.db_dsn = db_dsn
    if otlp:
        cfg.agent.sallm.otlp_endpoint = otlp
    if metrics_port:
        cfg.agent.sallm.metrics_port = metrics_port

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
        f"[dim]model={cfg.litellm_model}  agent={name}  mode=sallm[/dim]\n"
        "[dim]Durable ReAct + skills/CliTools. "
        "Type /exit or /quit to end. /stats shows session metrics; "
        "/reset clears sallm session memory.[/dim]\n"
    )
    try:
        asyncio.run(
            run_sallm_chat(
                cfg=cfg,
                emit=_rich_emit,
                on_thinking=_on_thinking,
                conversation_id="cli",
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

    from pawn_agent.core.sallm_tools import build_pawn_clitools  # noqa: PLC0415

    table = Table(title="Available CliTools (sallm)", show_lines=True, show_header=True)
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Description")

    for i, (name, tool) in enumerate(sorted(build_pawn_clitools().items()), start=1):
        table.add_row(str(i), name, tool.summary)

    console.print(table)


@app.command()
def models() -> None:
    """List available Copilot models."""
    import asyncio  # noqa: PLC0415

    from copilot import CopilotClient  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

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
        reasoning = (
            ", ".join(m.supported_reasoning_efforts) if m.supported_reasoning_efforts else "—"
        )
        table.add_row(
            m.id,
            m.name,
            f"[{policy_color}]{policy_state}[/{policy_color}]",
            multiplier,
            reasoning,
        )

    console.print(table)
    console.print(f"[dim]{len(model_list)} model(s)[/dim]")

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

# PydanticAI provider prefixes — anything else is treated as a bare model name.
_PYDANTIC_PREFIXES = ("openai:", "anthropic:", "google-gla:", "google-vertex:", "groq:", "mistral:", "bedrock:", "cohere:")


def _apply_model_override(cfg, model: str) -> None:
    """Set cfg.pydantic_model from a CLI --model value.

    If the value already carries a PydanticAI provider prefix (e.g. 'openai:gpt-4o')
    it is used as-is.  Otherwise the current provider prefix is preserved and only
    the model name is replaced — so '--model qwen3.5:9b' keeps 'openai:' when the
    config points at an Ollama/OpenAI-compatible endpoint.
    """
    if any(model.startswith(p) for p in _PYDANTIC_PREFIXES):
        cfg.pydantic_model = model
    else:
        # Bare model name (e.g. "qwen3.5:9b") — keep the configured provider prefix
        current_prefix = cfg.pydantic_model.split(":")[0] + ":"
        cfg.pydantic_model = f"{current_prefix}{model}"


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Natural-language instruction for the agent."),
    session: Optional[str] = typer.Option(
        None, "--session", "-s",
        help="Session ID hint. When provided it is injected into the prompt "
             "so the agent can locate the right conversation without guessing."
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML config file. Defaults to pawnai.yaml in cwd."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="PydanticAI model string. Overrides config (e.g. 'openai:gpt-4o', 'anthropic:claude-sonnet-4-5-20251001')."
    ),
    db_dsn: Optional[str] = typer.Option(
        None, "--db-dsn",
        help="PostgreSQL DSN. Overrides DATABASE_URL env var and config."
    ),
) -> None:
    """Run the pawn-agent on a natural-language [bold]PROMPT[/bold].

    The agent selects and invokes the appropriate tools (query DB, analyse,
    save to SiYuan) to satisfy the request.

    \b
    Examples
    --------
    pawn-agent "Summarise the last meeting" --session my-session
    pawn-agent "Analyse session abc123 and push it to SiYuan" --model claude-sonnet-4
    """
    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_agent.core.pydantic_agent import PydanticAgent  # noqa: PLC0415

    # ── Load and patch config ─────────────────────────────────────────────────
    cfg = load_config(config)
    if model:
        _apply_model_override(cfg, model)
    if db_dsn:
        cfg.db_dsn = db_dsn

    # ── Augment prompt with session hint ──────────────────────────────────────
    effective_prompt = prompt
    if session:
        effective_prompt = f"[Session ID: {session}]\n{prompt}"

    # ── Wire agent ────────────────────────────────────────────────────────────
    agent = PydanticAgent(cfg=cfg)

    # ── Run ───────────────────────────────────────────────────────────────────
    console.print(
        f"\n[bold cyan]pawn-agent[/bold cyan] "
        f"[dim]model={cfg.pydantic_model}[/dim]\n"
    )
    with console.status("[bold green]Thinking…[/bold green]"):
        try:
            response = agent.run(effective_prompt)
        except Exception as exc:
            console.print(f"[red]Agent error:[/red] {exc}")
            raise typer.Exit(1)

    console.print(Markdown(response))


@app.command()
def chat(
    session: Optional[str] = typer.Option(
        None, "--session", "-s",
        help="Session ID hint. Injected into the first message so the agent "
             "knows which conversation to reference."
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML config file. Defaults to pawnai.yaml in cwd."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="PydanticAI model string. Overrides config (e.g. 'openai:gpt-4o', 'anthropic:claude-sonnet-4-5-20251001')."
    ),
    db_dsn: Optional[str] = typer.Option(
        None, "--db-dsn",
        help="PostgreSQL DSN. Overrides DATABASE_URL env var and config."
    ),
) -> None:
    """Start an interactive multi-turn [bold]CHAT[/bold] session.

    The session stays alive across turns so the model retains context.
    Type [bold]/exit[/bold] or [bold]/quit[/bold] to end, or press Ctrl-D / Ctrl-C.

    \b
    Examples
    --------
    pawn-agent chat
    pawn-agent chat --session my-meeting --model gpt-4o
    """
    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_agent.core.pydantic_agent import PydanticAgent  # noqa: PLC0415
    from rich.markdown import Markdown  # noqa: PLC0415

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

    agent = PydanticAgent(cfg=cfg, emit=_rich_emit, on_thinking=_on_thinking)

    console.print(
        f"\n[bold cyan]pawn-agent chat[/bold cyan] "
        f"[dim]model={cfg.pydantic_model}  agent={name}[/dim]\n"
        "[dim]Type /exit or /quit to end. Ctrl-D also exits.[/dim]\n"
    )

    first_message: Optional[str] = None
    if session:
        try:
            console.print("[dim]You (first message):[/dim] ", end="")
            raw = input("").strip()
        except (EOFError, KeyboardInterrupt):
            return
        turn = raw or "Hello, let's discuss this session."
        first_message = f"[Session ID: {session}]\n{turn}"

    try:
        agent.chat(first_message=first_message)
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


@app.command()
def listen(
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML config file. Defaults to pawnai.yaml in cwd."
    ),
    topic: Optional[str] = typer.Option(
        None, "--topic", "-T",
        help="Topic name to subscribe to. Overrides the value in the queue: config section."
    ),
    consumer_name: Optional[str] = typer.Option(
        None, "--consumer-name", "-n",
        help="Consumer registration name. Overrides the value in the queue: config section."
    ),
) -> None:
    """Listen for commands on a pawn-queue topic and execute them.

    Connects to the S3-backed pawn-queue configured in the ``queue:`` section
    of ``pawnai.yaml`` and blocks until interrupted.  Each incoming message
    must be a JSON object with a ``command`` key:

    \b
      {
        "command": "run",
        "prompt": "Summarise session abc123 and push to SiYuan",
        "session_id": "abc123",
        "model": "openai:gpt-4o"
      }

    Results are persisted in the ``agent_runs`` database table.
    On success the message is acked; on failure it is sent to the dead-letter
    queue.  Stop with Ctrl-C.
    """
    import asyncio  # noqa: PLC0415
    import logging  # noqa: PLC0415

    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_agent.core.queue_listener import (  # noqa: PLC0415
        start_listener,
        DEFAULT_TOPIC,
        DEFAULT_CONSUMER_NAME,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config(config)

    queue_cfg = cfg.queue_config or {}
    effective_topic = topic or queue_cfg.get("topic", DEFAULT_TOPIC)
    effective_consumer = consumer_name or queue_cfg.get("consumer_name", DEFAULT_CONSUMER_NAME)

    console.print(
        f"[bold green]pawn-agent queue listener starting[/bold green]\n"
        f"  topic    : [cyan]{effective_topic}[/cyan]\n"
        f"  consumer : [cyan]{effective_consumer}[/cyan]\n"
        f"  model    : [dim]{cfg.pydantic_model}[/dim]"
    )
    console.print("[dim]Press Ctrl-C to stop.[/dim]\n")

    try:
        asyncio.run(
            start_listener(
                cfg=cfg,
                topic_override=topic,
                consumer_name_override=consumer_name,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Listener stopped by user.[/yellow]")
    except RuntimeError as exc:
        console.print(f"[red]Configuration error: {exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Listener error: {exc}[/red]")
        raise typer.Exit(1)

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
    Type [bold]/reset[/bold] to clear the conversation history and start fresh.

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
        "[dim]Type /exit or /quit to end. /reset to clear history. Ctrl-D also exits.[/dim]\n"
    )

    first_message: Optional[str] = None
    initial_history = None
    on_turn_complete = None
    if session:
        import uuid as _uuid  # noqa: PLC0415
        from prompt_toolkit import PromptSession as _PromptSession  # noqa: PLC0415
        from pawn_agent.core.session_store import (  # noqa: PLC0415
            append_turn,
            context_size,
            delete_session,
            load_history,
        )

        initial_history = load_history(session, cfg.db_dsn)
        kb, tok = context_size(initial_history)
        tok_str = f"{tok // 1000}k" if tok >= 1000 else str(tok)
        if initial_history:
            console.print(
                f"[dim]Resuming session {session!r} "
                f"({len(initial_history)} stored message(s) · "
                f"{kb:.1f} KB · ~{tok_str} tok)[/dim]\n"
            )

        try:
            raw = _PromptSession().prompt(f"You [{kb:.1f} KB · ~{tok_str} tok]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if raw.startswith("/"):
            # Slash commands must reach _repl_loop unmodified so they are
            # intercepted there.  Do not prepend the session ID hint.
            first_message = raw or None
        else:
            turn = raw or "Hello, let's discuss this session."
            first_message = f"[Session ID: {session}]\n{turn}"

        def on_turn_complete(new_msgs: list) -> None:
            append_turn(str(_uuid.uuid4()), session, new_msgs, cfg.db_dsn)

        def on_reset() -> None:
            delete_session(session, cfg.db_dsn)
            console.print("\n[dim]Conversation reset.[/dim]\n")

    else:
        def on_reset() -> None:
            console.print("\n[dim]Conversation reset.[/dim]\n")

    try:
        agent.chat(
            first_message=first_message,
            initial_history=initial_history,
            on_turn_complete=on_turn_complete,
            on_reset=on_reset,
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


@app.command()
def serve(
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML config file. Defaults to pawnai.yaml in cwd."
    ),
    host: Optional[str] = typer.Option(
        None, "--host", "-H",
        help="Bind host. Overrides api.host in config (default 0.0.0.0)."
    ),
    port: Optional[int] = typer.Option(
        None, "--port", "-p",
        help="Bind port. Overrides api.port in config (default 8000)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="PydanticAI model string. Overrides config."
    ),
    topic: Optional[str] = typer.Option(
        None, "--topic", "-T",
        help="Queue topic to subscribe to. Overrides agent_queue.topic in config."
    ),
    consumer_name: Optional[str] = typer.Option(
        None, "--consumer-name", "-n",
        help="Queue consumer registration name. Overrides agent_queue.consumer_name in config."
    ),
    no_queue: bool = typer.Option(
        False, "--no-queue",
        help="Disable the queue listener. Run the HTTP API server only."
    ),
) -> None:
    """Start the HTTP API server and queue listener together.

    Runs both the REST API and the S3-backed queue listener in a single
    process.  Either can be reached independently — HTTP clients hit the
    API while queue producers push jobs via pawn-queue.

    Pass [bold]--no-queue[/bold] to start only the HTTP API (e.g. when the
    queue infrastructure is unavailable).

    \b
    API Endpoints
    -------------
    POST   /chat                   Send a prompt (Bearer token required)
    DELETE /sessions/{session_id}  Clear a session (Bearer token required)
    GET    /health                 Liveness probe (no auth)
    GET    /docs                   Swagger UI
    GET    /openapi.json           OpenAPI spec

    \b
    Queue message format
    --------------------
    {
      "command": "run",
      "prompt":  "Summarise session abc123 and push to SiYuan",
      "session_id": "abc123",
      "model":   "openai:gpt-4o"
    }
    """
    import asyncio  # noqa: PLC0415
    import logging  # noqa: PLC0415
    import uvicorn  # noqa: PLC0415

    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_agent.core.api_server import create_app  # noqa: PLC0415
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
    if model:
        _apply_model_override(cfg, model)

    effective_host = host or cfg.api_host
    effective_port = port or cfg.api_port

    queue_cfg = cfg.queue_config or {}
    with_queue = not no_queue and bool(queue_cfg)
    effective_topic = topic or queue_cfg.get("topic", DEFAULT_TOPIC)
    effective_consumer = consumer_name or queue_cfg.get("consumer_name", DEFAULT_CONSUMER_NAME)

    console.print(
        f"[bold green]pawn-agent serve starting[/bold green]\n"
        f"  host     : [cyan]{effective_host}[/cyan]\n"
        f"  port     : [cyan]{effective_port}[/cyan]\n"
        f"  model    : [dim]{cfg.pydantic_model}[/dim]\n"
        f"  idle     : [dim]{cfg.api_model_idle_timeout_minutes} min[/dim]\n"
        f"  auth     : [dim]{'token set' if cfg.api_token else 'NO TOKEN — open access'}[/dim]\n"
        f"  queue    : [dim]{'topic=' + effective_topic + ' consumer=' + effective_consumer if with_queue else 'disabled'}[/dim]"
    )
    console.print("[dim]Press Ctrl-C to stop.[/dim]\n")

    fastapi_app = create_app(cfg)

    async def _main() -> None:
        uv_config = uvicorn.Config(fastapi_app, host=effective_host, port=effective_port, log_level="info")
        server = uvicorn.Server(uv_config)

        if not with_queue:
            await server.serve()
            return

        server_task = asyncio.create_task(server.serve())
        listener_task = asyncio.create_task(
            start_listener(cfg, topic_override=topic, consumer_name_override=consumer_name)
        )

        # Stop both when either exits (Ctrl-C, error, or natural completion)
        done, pending = await asyncio.wait(
            [server_task, listener_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        # Re-raise any exception from the completed task
        for task in done:
            if not task.cancelled() and task.exception():
                raise task.exception()  # type: ignore[misc]

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")
    except RuntimeError as exc:
        console.print(f"[red]Configuration error: {exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)

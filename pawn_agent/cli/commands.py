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


def _history_mode(cfg: object) -> str:
    return getattr(cfg, "history_mode", "raw")


def _history_kwargs(cfg: object) -> dict:
    return {
        "strip_thinking": getattr(cfg, "strip_thinking", True),
        "recent_turns": getattr(cfg, "history_recent_turns", 4),
        "replay_max_tokens": getattr(cfg, "history_replay_max_tokens", 8000),
        "max_text_chars": getattr(cfg, "history_max_text_chars", 500),
        "sanitize_leaked_thoughts": getattr(cfg, "history_sanitize_leaked_thoughts", True),
    }


def _print_session_troubleshooting(report: dict) -> None:
    from rich.table import Table  # noqa: PLC0415

    console.print(
        f"\n[bold cyan]Session Inspection[/bold cyan] "
        f"[dim]{report['session_id']}[/dim]\n"
    )

    summary = Table(show_header=False, box=None)
    summary.add_column("Key", style="cyan", no_wrap=True)
    summary.add_column("Value")
    summary.add_row("Turns", str(report["turn_count"]))
    summary.add_row("Raw messages", str(report["raw_message_count"]))
    summary.add_row("Replay messages", str(report["replay_message_count"]))
    summary.add_row(
        "Raw context",
        f"{report['raw_context_kb']:.1f} KB · ~{report['raw_context_tokens']} tok",
    )
    summary.add_row(
        "Replay context",
        f"{report['replay_context_kb']:.1f} KB · ~{report['replay_context_tokens']} tok",
    )
    summary.add_row(
        "Session vars",
        str(report["session_vars"] or {}),
    )
    console.print(summary)

    issues = report["issues"]
    if issues:
        console.print("\n[bold yellow]Potential Issues[/bold yellow]")
        for issue in issues:
            console.print(f"- {issue}")
    else:
        console.print("\n[green]No obvious structural history issues found.[/green]")

    counts = report["counts"]
    interesting_keys = [
        "retry_prompt_requests",
        "responses_without_text",
        "blank_text_responses",
        "thinking_only_responses",
        "tool_call_only_responses",
        "responses_empty_after_strip",
        "consecutive_same_kind",
        "dangling_request_at_end",
    ]
    count_rows = [(key, counts.get(key, 0)) for key in interesting_keys if counts.get(key, 0)]
    if count_rows:
        count_table = Table(title="Issue Counts", show_lines=False)
        count_table.add_column("Signal", style="cyan")
        count_table.add_column("Count", justify="right")
        for key, value in count_rows:
            count_table.add_row(key, str(value))
        console.print()
        console.print(count_table)

    if report["recent_messages"]:
        msg_table = Table(title="Recent Messages", show_lines=True)
        msg_table.add_column("#", justify="right", style="dim", no_wrap=True)
        msg_table.add_column("Kind", style="cyan", no_wrap=True)
        msg_table.add_column("Parts")
        msg_table.add_column("Excerpt")
        msg_table.add_column("Issues")
        for msg in report["recent_messages"]:
            msg_table.add_row(
                str(msg["index"]),
                msg["kind"],
                ", ".join(msg["parts"]) or "—",
                msg["excerpt"],
                "; ".join(msg["issues"]) or "—",
            )
        console.print()
        console.print(msg_table)


def _print_llm_context_dump(report: dict) -> None:
    console.print("\n[bold cyan]LLM Context Dump[/bold cyan]")
    console.print("[dim]This is the replayed message_history payload after strip_thinking.[/dim]\n")
    console.print_json(data=report["replay_payload"])


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
    Type [bold]/set key=value[/bold] to set a session variable (e.g. listen_only=true).
    Type [bold]/unset key[/bold] to remove a session variable. [bold]/vars[/bold] lists all.

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

    agent = PydanticAgent(cfg=cfg, emit=_rich_emit, on_thinking=_on_thinking, session_id=session)

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
            build_replay_history,
            context_size,
            delete_session,
            load_history,
        )

        if _history_mode(cfg) == "raw":
            initial_history = load_history(session, cfg.db_dsn, strip_thinking=cfg.strip_thinking)
        else:
            initial_history = build_replay_history(
                session,
                cfg.db_dsn,
                **_history_kwargs(cfg),
            )
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


@app.command(name="inspect-session")
def inspect_session(
    session_id: str = typer.Argument(..., help="Session ID to inspect."),
    tail: int = typer.Option(
        12, "--tail",
        min=1,
        help="How many recent messages to show in the troubleshooting table."
    ),
    dump_context: bool = typer.Option(
        False, "--dump-context",
        help="Print the exact replayed message_history payload that would be passed to the LLM."
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML config file. Defaults to pawnai.yaml in cwd."
    ),
    db_dsn: Optional[str] = typer.Option(
        None, "--db-dsn",
        help="PostgreSQL DSN. Overrides DATABASE_URL env var and config."
    ),
) -> None:
    """Inspect stored session history and flag suspicious message patterns."""
    from pawn_agent.core.session_store import inspect_session_history  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    cfg = load_config(config)
    if db_dsn:
        cfg.db_dsn = db_dsn

    try:
        report = inspect_session_history(session_id, cfg.db_dsn, tail=tail)
    except Exception as exc:
        console.print(f"[red]Inspection error:[/red] {exc}")
        raise typer.Exit(1)

    _print_session_troubleshooting(report)
    if dump_context:
        _print_llm_context_dump(report)


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

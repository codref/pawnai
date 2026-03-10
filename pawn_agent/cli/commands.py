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
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b",
        help="LM backend: 'copilot' or 'openai'. Overrides config."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model name. Overrides config (e.g. 'gpt-4o', 'llama3')."
    ),
    openai_base_url: Optional[str] = typer.Option(
        None, "--openai-base-url",
        help="Base URL for OpenAI-compatible endpoint (e.g. http://localhost:11434/v1)."
    ),
    openai_api_key: Optional[str] = typer.Option(
        None, "--openai-api-key",
        help="API key for the OpenAI-compatible endpoint."
    ),
    db_dsn: Optional[str] = typer.Option(
        None, "--db-dsn",
        help="PostgreSQL DSN. Overrides DATABASE_URL env var and config."
    ),
    max_iters: int = typer.Option(
        6, "--max-iters",
        help="Maximum ReAct iterations."
    ),
) -> None:
    """Run the pawn-agent on a natural-language [bold]PROMPT[/bold].

    The agent selects and invokes the appropriate tools (query DB, analyse,
    save to SiYuan) to satisfy the request.

    \b
    Examples
    --------
    pawn-agent "Summarise the last meeting" --session my-session
    pawn-agent "Analyse session abc123 and push it to SiYuan" --backend openai --model llama3
    """
    import dspy  # noqa: PLC0415

    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_agent.core.lm import build_lm  # noqa: PLC0415
    from pawn_agent.core.tools import build_tools  # noqa: PLC0415
    from pawn_agent.core.agent import ConversationAgent  # noqa: PLC0415

    # ── Load and patch config ─────────────────────────────────────────────────
    cfg = load_config(config)
    if backend:
        cfg.backend = backend
    if model:
        cfg.model = model
    if openai_base_url:
        cfg.openai_base_url = openai_base_url
    if openai_api_key:
        cfg.openai_api_key = openai_api_key
    if db_dsn:
        cfg.db_dsn = db_dsn

    # ── Augment prompt with session hint ──────────────────────────────────────
    effective_prompt = prompt
    if session:
        effective_prompt = f"[Session ID: {session}]\n{prompt}"

    # ── Wire agent ────────────────────────────────────────────────────────────
    lm = build_lm(cfg)
    # Use ChatAdapter so DSPy sends plain conversational messages rather than
    # JSON-schema-structured prompts (JSONAdapter default) which Copilot rejects.
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    tools = build_tools(cfg)
    agent = ConversationAgent(tools=tools, max_iters=max_iters)

    # ── Run ───────────────────────────────────────────────────────────────────
    console.print(
        f"\n[bold cyan]pawn-agent[/bold cyan] "
        f"[dim]backend={cfg.backend} model={cfg.model}[/dim]\n"
    )
    with console.status("[bold green]Thinking…[/bold green]"):
        try:
            response = agent(user_prompt=effective_prompt)
        except Exception as exc:
            console.print(f"[red]Agent error:[/red] {exc}")
            raise typer.Exit(1)

    console.print(Markdown(response))

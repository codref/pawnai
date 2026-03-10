"""CLI entry point for pawn-agent.

Usage::

    pawn-agent start --config .pawnai.yml
    pawn-agent start --config .pawnai.yml --topic agent-tasks
    python -m pawn_agent start

Commands
---------
start   Boot the agent listener (blocks until Ctrl-C).
skills  List all loaded skills.
tools   List all loaded tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="pawn-agent",
    help="DSPy-powered agentic queue listener for PawnAI.",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


async def _boot(cfg_path: Optional[str]) -> tuple:
    """Load config, skill/tool definitions, and build the agent components."""
    from pawn_agent.agent.planner import AgentPlanner
    from pawn_agent.config import AgentConfig
    from pawn_agent.executors.registration import build_executor_registry
    from pawn_agent.lm import CopilotLM
    from pawn_agent.skills.dir_loader import DirectorySkillLoader, DirectoryToolLoader
    from pawn_agent.skills.registry import SkillRegistry
    from pawn_agent.skills.runner import SkillRunner

    import dspy  # lazy — avoids import-time side effects

    cfg = AgentConfig(config_path=cfg_path)

    # ── Load skills and tools ─────────────────────────────────────────────────
    skill_loader = DirectorySkillLoader(cfg.skills_dir)
    tool_loader = DirectoryToolLoader(cfg.tools_dir)

    skills = await skill_loader.load()
    tools = await tool_loader.load()

    if not skills:
        console.print(
            f"[yellow]Warning:[/yellow] No enabled skills found in {cfg.skills_dir}. "
            "The agent will be unable to plan."
        )

    registry = SkillRegistry(skills=skills, tools=tools)
    executor_registry = build_executor_registry()
    runner = SkillRunner(registry, executor_registry)

    # ── Configure DSPy with Copilot LM ────────────────────────────────────────
    lm = CopilotLM(model=cfg.llm_model, temperature=cfg.llm_temperature)
    dspy.configure(lm=lm)

    planner = AgentPlanner()

    return cfg, planner, runner, registry


@app.command()
def start(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to .pawnai.yml config file (default: .pawnai.yml in cwd)",
        envvar="PAWN_AGENT_CONFIG",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-t",
        help="Queue topic to subscribe to (overrides config value)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Start the pawn-agent queue listener (blocks until Ctrl-C)."""
    _setup_logging(verbose)

    async def _run() -> None:
        from pawn_agent.listener import start_listener

        cfg, planner, runner, _ = await _boot(config)

        if topic:
            # Override topic by patching the config's raw dict
            cfg._raw.setdefault("agent", {})["topic"] = topic  # type: ignore[attr-defined]

        console.print(
            f"[green]pawn-agent starting[/green] | "
            f"topic=[bold]{cfg.topic}[/bold] | "
            f"model=[bold]{cfg.llm_model}[/bold]"
        )

        try:
            await start_listener(cfg, planner, runner)
        except KeyboardInterrupt:
            pass

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — shutting down.[/yellow]")
        sys.exit(0)
    except Exception as exc:
        console.print(f"[red]Fatal error:[/red] {exc}")
        sys.exit(1)


@app.command()
def skills(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List all loaded skills."""
    _setup_logging(False)

    async def _run() -> None:
        _, _, _, registry = await _boot(config)
        skill_list = registry.list_skills()

        if json_output:
            console.print_json(registry.describe_skills())
            return

        table = Table(title="Loaded Skills", show_lines=True)
        table.add_column("Name", style="cyan bold")
        table.add_column("Steps", justify="center")
        table.add_column("Description")
        for s in skill_list:
            table.add_row(s.name, str(len(s.tools)), s.description)
        console.print(table)

    asyncio.run(_run())


@app.command()
def tools(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List all loaded tools."""
    _setup_logging(False)

    async def _run() -> None:
        _, _, _, registry = await _boot(config)
        tool_list = registry.list_tools()

        if json_output:
            data = [
                {"name": t.name, "function": t.function, "description": t.description}
                for t in tool_list
            ]
            console.print_json(json.dumps(data))
            return

        table = Table(title="Loaded Tools", show_lines=True)
        table.add_column("Name", style="cyan bold")
        table.add_column("Function key", style="yellow")
        table.add_column("Description")
        for t in tool_list:
            table.add_row(t.name, t.function, t.description)
        console.print(table)

    asyncio.run(_run())


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()

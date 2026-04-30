"""CLI commands for pawn-server."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="pawn-server",
    help="HTTP API server and queue listener for pawn-agent.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()
schedules_app = typer.Typer(
    name="schedules",
    help="Manage agent schedules and schedule proposals.",
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(schedules_app, name="schedules")


def _load_scheduler_service(config: Optional[str]):
    from pawn_agent.core.scheduler import AgentSchedulerService  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    cfg = load_config(config)
    return cfg, AgentSchedulerService(
        cfg.db_dsn,
        default_timezone=cfg.agent_scheduler.default_timezone,
    )


def _print_schedule_table(rows: list[dict]) -> None:
    table = Table(show_header=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Status", no_wrap=True)
    table.add_column("Kind", no_wrap=True)
    table.add_column("Session")
    table.add_column("Next Run", no_wrap=True)
    for row in rows:
        table.add_row(
            row["id"],
            row["name"],
            row["status"],
            row["schedule_kind"],
            row["session_id"],
            row["next_run_at"] or "-",
        )
    console.print(table)


@schedules_app.command("list")
def schedules_list(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    all: bool = typer.Option(False, "--all", help="Include cancelled/completed schedules."),
) -> None:
    """List schedules."""
    _, service = _load_scheduler_service(config)
    _print_schedule_table(service.list_schedules(include_inactive=all))


@schedules_app.command("show")
def schedules_show(
    schedule_id: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Show one schedule as JSON."""
    import json  # noqa: PLC0415

    _, service = _load_scheduler_service(config)
    console.print_json(json.dumps(service.get_schedule(schedule_id)))


@schedules_app.command("proposals")
def schedules_proposals(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    all: bool = typer.Option(False, "--all", help="Include resolved proposals."),
) -> None:
    """List schedule proposals."""
    _, service = _load_scheduler_service(config)
    rows = service.list_proposals(include_resolved=all)
    table = Table(show_header=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Action", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Schedule ID")
    table.add_column("Created", no_wrap=True)
    table.add_column("Rationale")
    for row in rows:
        table.add_row(
            row["id"],
            row["action"],
            row["status"],
            row["schedule_id"] or "-",
            row["created_at"] or "-",
            row["rationale"] or "",
        )
    console.print(table)


@schedules_app.command("approve")
def schedules_approve(
    proposal_id: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    reviewed_by: str = typer.Option("user", "--reviewed-by"),
) -> None:
    """Approve and apply a schedule proposal."""
    _, service = _load_scheduler_service(config)
    created_id = service.approve_proposal(proposal_id, reviewed_by=reviewed_by)
    if created_id:
        console.print(f"[green]Proposal applied.[/green] Created schedule {created_id}")
    else:
        console.print("[green]Proposal applied.[/green]")


@schedules_app.command("reject")
def schedules_reject(
    proposal_id: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    reviewed_by: str = typer.Option("user", "--reviewed-by"),
) -> None:
    """Reject a schedule proposal."""
    _, service = _load_scheduler_service(config)
    service.reject_proposal(proposal_id, reviewed_by=reviewed_by)
    console.print("[yellow]Proposal rejected.[/yellow]")


@schedules_app.command("pause")
def schedules_pause(
    schedule_id: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Pause an active schedule."""
    _, service = _load_scheduler_service(config)
    service.pause_schedule(schedule_id)
    console.print("[yellow]Schedule paused.[/yellow]")


@schedules_app.command("resume")
def schedules_resume(
    schedule_id: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Resume a paused schedule."""
    _, service = _load_scheduler_service(config)
    service.resume_schedule(schedule_id)
    console.print("[green]Schedule resumed.[/green]")


@schedules_app.command("cancel")
def schedules_cancel(
    schedule_id: str = typer.Argument(...),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Cancel a schedule."""
    _, service = _load_scheduler_service(config)
    service.cancel_schedule(schedule_id)
    console.print("[yellow]Schedule cancelled.[/yellow]")


@app.command()
def serve(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file. Defaults to pawnai.yaml in cwd."
    ),
    host: Optional[str] = typer.Option(
        None, "--host", "-H", help="Bind host. Overrides api.host in config (default 0.0.0.0)."
    ),
    port: Optional[int] = typer.Option(
        None, "--port", "-p", help="Bind port. Overrides api.port in config (default 8000)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="PydanticAI model string. Overrides config."
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-T",
        help="Queue topic to subscribe to. Overrides agent_queue.topic in config.",
    ),
    consumer_name: Optional[str] = typer.Option(
        None,
        "--consumer-name",
        "-n",
        help="Queue consumer registration name. Overrides agent_queue.consumer_name in config.",
    ),
    no_queue: bool = typer.Option(
        False, "--no-queue", help="Disable the queue listener. Run the HTTP API server only."
    ),
    disable_scheduler: bool = typer.Option(
        False,
        "--disable-scheduler",
        help="Disable the durable agent scheduler.",
    ),
    scheduler_only: bool = typer.Option(
        False,
        "--scheduler-only",
        help="Run only the durable scheduler, without HTTP API or queue listener.",
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
    POST   /v1/chat/completions    OpenAI-compatible chat (Bearer token required)
    DELETE /sessions/{session_id}  Clear a session (Bearer token required)
    POST   /knowledge              Index content into RAG (Bearer token required)
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
    from pawn_agent.utils.model_utils import _apply_model_override  # noqa: PLC0415
    from pawn_server.core.api_server import create_app  # noqa: PLC0415
    from pawn_server.core.queue_listener import (  # noqa: PLC0415
        start_listener,
        DEFAULT_TOPIC,
        DEFAULT_CONSUMER_NAME,
    )
    from pawn_agent.core.scheduler import start_scheduler  # noqa: PLC0415

    cfg = load_config(config)

    # Configure logging from pawnai.yaml before uvicorn starts.
    # We also pin pawn_* package loggers explicitly so that uvicorn's
    # dictConfig (which resets the root logger to WARNING) does not silence them.
    _log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for _pkg in ("pawn_agent", "pawn_server", "pawn_core", "pawn_diarize"):
        logging.getLogger(_pkg).setLevel(_log_level)
    if model:
        _apply_model_override(cfg, model)

    effective_host = host or cfg.api_host
    effective_port = port or cfg.api_port

    queue_cfg = cfg.queue_config or {}
    with_queue = not no_queue and bool(queue_cfg)
    with_scheduler = bool(cfg.agent_scheduler.enabled) and not disable_scheduler
    effective_topic = topic or queue_cfg.get("topic", DEFAULT_TOPIC)
    effective_consumer = consumer_name or queue_cfg.get("consumer_name", DEFAULT_CONSUMER_NAME)

    console.print(
        f"[bold green]pawn-server serve starting[/bold green]\n"
        f"  host     : [cyan]{effective_host}[/cyan]\n"
        f"  port     : [cyan]{effective_port}[/cyan]\n"
        f"  model    : [dim]{cfg.pydantic_model}[/dim]\n"
        f"  idle     : [dim]{cfg.api_model_idle_timeout_minutes} min[/dim]\n"
        f"  auth     : [dim]{'token set' if cfg.api_token else 'NO TOKEN — open access'}[/dim]\n"
        f"  queue    : [dim]{'topic=' + effective_topic + ' consumer=' + effective_consumer if with_queue and not scheduler_only else 'disabled'}[/dim]\n"
        f"  scheduler: [dim]{'enabled' if with_scheduler else 'disabled'}[/dim]"
    )
    console.print("[dim]Press Ctrl-C to stop.[/dim]\n")

    fastapi_app = create_app(cfg)

    async def _main() -> None:
        if scheduler_only:
            if not with_scheduler:
                raise RuntimeError("Scheduler is disabled by config or --disable-scheduler")
            await start_scheduler(cfg)
            return

        uv_config = uvicorn.Config(
            fastapi_app, host=effective_host, port=effective_port, log_level="info"
        )
        server = uvicorn.Server(uv_config)

        if not with_queue and not with_scheduler:
            await server.serve()
            return

        tasks = [asyncio.create_task(server.serve())]
        if with_queue:
            tasks.append(
                asyncio.create_task(
                    start_listener(cfg, topic_override=topic, consumer_name_override=consumer_name)
                )
            )
        if with_scheduler:
            tasks.append(asyncio.create_task(start_scheduler(cfg)))

        # Stop both when either exits (Ctrl-C, error, or natural completion)
        done, pending = await asyncio.wait(
            tasks,
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

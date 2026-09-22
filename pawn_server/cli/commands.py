"""CLI commands for pawn-server."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="pawn-server",
    help="HTTP API, queue listener, scheduler, and optional Matrix bot for pawn-agent.",
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
queue_app = typer.Typer(
    name="queue",
    help="Manage the pawn-agent S3-backed job queue.",
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(schedules_app, name="schedules")
app.add_typer(queue_app, name="queue")


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


@queue_app.command("empty")
def queue_empty(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file. Defaults to pawnai.yaml in cwd."
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-N",
        help="Configured queue name: agent, diarize, or a queue_producers key.",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-T",
        help="Select by topic name (e.g. audio-chunks, pawn-agent-jobs).",
    ),
    all_targets: bool = typer.Option(
        False, "--all", help="Empty every configured queue."
    ),
    include_dead_letter: bool = typer.Option(
        False,
        "--include-dead-letter",
        help="Also delete dead-letter messages for the topic.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would be deleted without deleting."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete pending messages (and leases) from one or more configured queues.

    Leaves topic registration markers in place. Dead-letter objects are kept
    unless ``--include-dead-letter`` is set. With multiple queues configured,
    pass ``--name``, ``--topic``, or ``--all``.
    """
    import asyncio  # noqa: PLC0415

    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_server.core.queue_admin import (  # noqa: PLC0415
        empty_queue,
        resolve_queue_targets,
    )

    cfg = load_config(config)
    try:
        targets = resolve_queue_targets(
            cfg, name=name, topic=topic, all_targets=all_targets
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if not dry_run and not yes:
        labels = ", ".join(f"{t.name}={t.topic}" for t in targets)
        extras = " (including dead-letter)" if include_dead_letter else ""
        console.print(
            f"[yellow]About to empty pending messages and leases for "
            f"{labels}{extras}.[/yellow]"
        )
        if not typer.confirm("Proceed?", default=False):
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    try:
        results = asyncio.run(
            empty_queue(
                cfg,
                name=name,
                topic=topic,
                all_targets=all_targets,
                include_dead_letter=include_dead_letter,
                dry_run=dry_run,
            )
        )
    except (RuntimeError, ImportError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Error emptying queue: {exc}[/red]")
        raise typer.Exit(1)

    prefix = "[dim][dry-run] would delete[/dim]" if dry_run else "[green]Deleted[/green]"
    for result in results:
        if result.total == 0:
            console.print(
                f"[dim]{result.name} topic {result.topic!r} is already empty"
                f"{' (dead-letter included)' if include_dead_letter else ''}.[/dim]"
            )
            continue
        if dry_run:
            for key in result.keys:
                console.print(f"{prefix}: {key}")
        console.print(
            f"{prefix}: {result.name} topic={result.topic!r} "
            f"{result.total} object(s) "
            f"(messages={result.messages}, leases={result.leases}"
            f"{', dead_letters=' + str(result.dead_letters) if include_dead_letter else ''})"
        )


@queue_app.command("stats")
def queue_stats_cmd(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file. Defaults to pawnai.yaml in cwd."
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-N",
        help="Configured queue name: agent, diarize, or a queue_producers key.",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-T",
        help="Select by topic name (e.g. audio-chunks, pawn-agent-jobs).",
    ),
) -> None:
    """Show pending message, lease, dead-letter, and pause state.

    With no selector, lists every configured queue (agent, diarize, producers).
    """
    import asyncio  # noqa: PLC0415

    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_server.core.queue_admin import queue_stats  # noqa: PLC0415

    cfg = load_config(config)
    try:
        rows = asyncio.run(queue_stats(cfg, name=name, topic=topic))
    except (RuntimeError, ImportError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Error reading queue stats: {exc}[/red]")
        raise typer.Exit(1)

    table = Table(show_header=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Topic", no_wrap=True)
    table.add_column("Bucket", no_wrap=True)
    table.add_column("Paused", no_wrap=True)
    table.add_column("Pending", justify="right", no_wrap=True)
    table.add_column("Leases", justify="right", no_wrap=True)
    table.add_column("Dead", justify="right", no_wrap=True)
    for stats in rows:
        if stats.paused and stats.paused_at:
            paused = f"[yellow]yes[/yellow]\n[dim]{stats.paused_at}[/dim]"
        elif stats.paused:
            paused = "[yellow]yes[/yellow]"
        else:
            paused = "[green]no[/green]"
        table.add_row(
            stats.name,
            stats.source,
            stats.topic,
            stats.bucket,
            paused,
            str(stats.messages),
            str(stats.leases),
            str(stats.dead_letters),
        )
    console.print(table)


@queue_app.command("pause")
def queue_pause(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file. Defaults to pawnai.yaml in cwd."
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-N",
        help="Configured queue name: agent, diarize, or a queue_producers key.",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-T",
        help="Select by topic name (e.g. audio-chunks, pawn-agent-jobs).",
    ),
    all_targets: bool = typer.Option(
        False, "--all", help="Pause every configured queue."
    ),
) -> None:
    """Pause processing: listeners stop claiming new messages for the topic(s)."""
    import asyncio  # noqa: PLC0415

    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_server.core.queue_admin import set_queue_paused  # noqa: PLC0415

    cfg = load_config(config)
    try:
        results = asyncio.run(
            set_queue_paused(
                cfg, paused=True, name=name, topic=topic, all_targets=all_targets
            )
        )
    except (RuntimeError, ImportError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Error pausing queue: {exc}[/red]")
        raise typer.Exit(1)

    for result in results:
        if result.changed:
            console.print(
                f"[yellow]Paused {result.name} topic={result.topic!r}.[/yellow]"
            )
        else:
            console.print(
                f"[dim]{result.name} topic={result.topic!r} was already paused.[/dim]"
            )


@queue_app.command("resume")
def queue_resume(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file. Defaults to pawnai.yaml in cwd."
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-N",
        help="Configured queue name: agent, diarize, or a queue_producers key.",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-T",
        help="Select by topic name (e.g. audio-chunks, pawn-agent-jobs).",
    ),
    all_targets: bool = typer.Option(
        False, "--all", help="Resume every configured queue."
    ),
) -> None:
    """Resume processing after ``queue pause``."""
    import asyncio  # noqa: PLC0415

    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_server.core.queue_admin import set_queue_paused  # noqa: PLC0415

    cfg = load_config(config)
    try:
        results = asyncio.run(
            set_queue_paused(
                cfg, paused=False, name=name, topic=topic, all_targets=all_targets
            )
        )
    except (RuntimeError, ImportError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Error resuming queue: {exc}[/red]")
        raise typer.Exit(1)

    for result in results:
        if result.changed:
            console.print(
                f"[green]Resumed {result.name} topic={result.topic!r}.[/green]"
            )
        else:
            console.print(
                f"[dim]{result.name} topic={result.topic!r} was not paused.[/dim]"
            )


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
    no_matrix: bool = typer.Option(
        False,
        "--no-matrix",
        help="Disable the Matrix bot even if matrix_bot.enabled is true.",
    ),
    matrix_only: bool = typer.Option(
        False,
        "--matrix-only",
        help="Run only the Matrix bot (no HTTP API, queue, or scheduler).",
    ),
    no_siyuan_watcher: bool = typer.Option(
        False,
        "--no-siyuan-watcher",
        help="Disable the SiYuan @pawn watcher even if siyuan_watcher.enabled.",
    ),
    siyuan_watcher_only: bool = typer.Option(
        False,
        "--siyuan-watcher-only",
        help="Run only the SiYuan @pawn watcher (no HTTP API / queue / scheduler / Matrix).",
    ),
) -> None:
    """Start the HTTP API and optional workers (queue, scheduler, Matrix, SiYuan).

    Workers are armed from config and can be forced off with ``--no-*`` flags.
    Use ``--matrix-only``, ``--scheduler-only``, or ``--siyuan-watcher-only``
    for a single-worker process.

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

    from pawn_agent.core.scheduler import start_scheduler  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_agent.utils.model_utils import _apply_model_override  # noqa: PLC0415
    from pawn_server.core.api_server import create_app  # noqa: PLC0415
    from pawn_server.core.matrix_bot import start_matrix_bot  # noqa: PLC0415
    from pawn_server.core.matrix_notifier import matrix_notifier_enabled  # noqa: PLC0415
    from pawn_server.core.queue_listener import (  # noqa: PLC0415
        DEFAULT_CONSUMER_NAME,
        DEFAULT_TOPIC,
        start_listener,
    )
    from pawn_server.core.siyuan_watcher import start_siyuan_watcher  # noqa: PLC0415

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

    only_flags = sum(bool(x) for x in (scheduler_only, matrix_only, siyuan_watcher_only))
    if only_flags > 1:
        console.print(
            "[red]Use only one of --scheduler-only / --matrix-only / "
            "--siyuan-watcher-only.[/red]"
        )
        raise typer.Exit(1)

    effective_host = host or cfg.api_host
    effective_port = port or cfg.api_port

    queue_cfg = cfg.queue_config or {}
    with_queue = not no_queue and bool(queue_cfg)
    with_scheduler = bool(cfg.agent_scheduler.enabled) and not disable_scheduler
    with_matrix = bool(cfg.matrix_bot.enabled) and not no_matrix
    with_siyuan = bool(cfg.siyuan_watcher.enabled) and not no_siyuan_watcher
    with_matrix_notifier = (
        with_matrix and matrix_notifier_enabled(cfg) and not no_matrix
    )
    effective_topic = topic or queue_cfg.get("topic", DEFAULT_TOPIC)
    effective_consumer = consumer_name or queue_cfg.get("consumer_name", DEFAULT_CONSUMER_NAME)
    only_mode = scheduler_only or matrix_only or siyuan_watcher_only

    console.print(
        f"[bold green]pawn-server serve starting[/bold green]\n"
        f"  host     : [cyan]{effective_host}[/cyan]\n"
        f"  port     : [cyan]{effective_port}[/cyan]\n"
        f"  model    : [dim]{cfg.pydantic_model}[/dim]\n"
        f"  idle     : [dim]{cfg.api_model_idle_timeout_minutes} min[/dim]\n"
        f"  auth     : [dim]{'token set' if cfg.api_token else 'NO TOKEN — open access'}[/dim]\n"
        f"  queue    : [dim]{'topic=' + effective_topic + ' consumer=' + effective_consumer if with_queue and not only_mode else 'disabled'}[/dim]\n"
        f"  scheduler: [dim]{'enabled' if with_scheduler and not matrix_only and not siyuan_watcher_only else 'disabled'}[/dim]\n"
        f"  matrix   : [dim]{'enabled' if with_matrix and not scheduler_only and not siyuan_watcher_only else 'disabled'}[/dim]\n"
        f"  siyuan   : [dim]{'enabled' if with_siyuan and not scheduler_only and not matrix_only else 'disabled'}[/dim]\n"
        f"  notify   : [dim]{'via matrix bot' if with_matrix_notifier and with_matrix and not scheduler_only and not siyuan_watcher_only else 'disabled'}[/dim]"
    )
    console.print("[dim]Press Ctrl-C to stop.[/dim]\n")

    async def _main() -> None:
        if scheduler_only:
            if not with_scheduler:
                raise RuntimeError("Scheduler is disabled by config or --disable-scheduler")
            await start_scheduler(cfg)
            return

        if matrix_only:
            if not with_matrix:
                raise RuntimeError("Matrix bot is disabled by config or --no-matrix")
            await start_matrix_bot(cfg)
            return

        if siyuan_watcher_only:
            if not with_siyuan:
                raise RuntimeError(
                    "SiYuan watcher is disabled by config or --no-siyuan-watcher"
                )
            await start_siyuan_watcher(cfg)
            return

        fastapi_app = create_app(cfg)
        uv_config = uvicorn.Config(
            fastapi_app, host=effective_host, port=effective_port, log_level="info"
        )
        server = uvicorn.Server(uv_config)

        if (
            not with_queue
            and not with_scheduler
            and not with_matrix
            and not with_siyuan
        ):
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
        if with_matrix:
            tasks.append(asyncio.create_task(start_matrix_bot(cfg)))
        if with_siyuan:
            tasks.append(asyncio.create_task(start_siyuan_watcher(cfg)))

        # Stop all when any exits (Ctrl-C, error, or natural completion)
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

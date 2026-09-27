"""CLI commands for pawn-server."""

from __future__ import annotations

from typing import Any, Optional

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
blacklist_app = typer.Typer(
    name="blacklist",
    help="Manage the API IP blacklist (brute-force / scan protection).",
    add_completion=False,
    rich_markup_mode="rich",
)
coworker_app = typer.Typer(
    name="coworker",
    help="Goal-driven inbox: process a session, reindex, pause, or resume.",
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(schedules_app, name="schedules")
app.add_typer(queue_app, name="queue")
app.add_typer(blacklist_app, name="blacklist")
app.add_typer(coworker_app, name="coworker")


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
    all_targets: bool = typer.Option(False, "--all", help="Empty every configured queue."),
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
        targets = resolve_queue_targets(cfg, name=name, topic=topic, all_targets=all_targets)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if not dry_run and not yes:
        labels = ", ".join(f"{t.name}={t.topic}" for t in targets)
        extras = " (including dead-letter)" if include_dead_letter else ""
        console.print(
            f"[yellow]About to empty pending messages and leases for " f"{labels}{extras}.[/yellow]"
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
    all_targets: bool = typer.Option(False, "--all", help="Pause every configured queue."),
) -> None:
    """Pause processing: listeners stop claiming new messages for the topic(s)."""
    import asyncio  # noqa: PLC0415

    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_server.core.queue_admin import set_queue_paused  # noqa: PLC0415

    cfg = load_config(config)
    try:
        results = asyncio.run(
            set_queue_paused(cfg, paused=True, name=name, topic=topic, all_targets=all_targets)
        )
    except (RuntimeError, ImportError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Error pausing queue: {exc}[/red]")
        raise typer.Exit(1)

    for result in results:
        if result.changed:
            console.print(f"[yellow]Paused {result.name} topic={result.topic!r}.[/yellow]")
        else:
            console.print(f"[dim]{result.name} topic={result.topic!r} was already paused.[/dim]")


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
    all_targets: bool = typer.Option(False, "--all", help="Resume every configured queue."),
) -> None:
    """Resume processing after ``queue pause``."""
    import asyncio  # noqa: PLC0415

    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_server.core.queue_admin import set_queue_paused  # noqa: PLC0415

    cfg = load_config(config)
    try:
        results = asyncio.run(
            set_queue_paused(cfg, paused=False, name=name, topic=topic, all_targets=all_targets)
        )
    except (RuntimeError, ImportError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Error resuming queue: {exc}[/red]")
        raise typer.Exit(1)

    for result in results:
        if result.changed:
            console.print(f"[green]Resumed {result.name} topic={result.topic!r}.[/green]")
        else:
            console.print(f"[dim]{result.name} topic={result.topic!r} was not paused.[/dim]")


def _load_blacklist_cfg(config: Optional[str]):
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    return load_config(config)


def resolve_ssl_files(
    certfile: Optional[str],
    keyfile: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Validate optional TLS paths; return ``(cert, key)`` or ``(None, None)``.

    Raises ``ValueError`` when only one path is set or a path is missing.
    """
    from pathlib import Path  # noqa: PLC0415

    cert = (certfile or "").strip() or None
    key = (keyfile or "").strip() or None
    if not cert and not key:
        return None, None
    if not cert or not key:
        raise ValueError(
            "TLS requires both ssl_certfile and ssl_keyfile "
            "(or --ssl-certfile and --ssl-keyfile)."
        )
    cert_path = Path(cert).expanduser()
    key_path = Path(key).expanduser()
    if not cert_path.is_file():
        raise ValueError(f"ssl_certfile not found: {cert_path}")
    if not key_path.is_file():
        raise ValueError(f"ssl_keyfile not found: {key_path}")
    return str(cert_path), str(key_path)


@blacklist_app.command("list")
def blacklist_list(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    all: bool = typer.Option(False, "--all", help="Include expired entries."),
) -> None:
    """List blacklisted client IPs."""
    from pawn_agent.utils.db import list_ip_blacklist  # noqa: PLC0415

    cfg = _load_blacklist_cfg(config)
    rows = list_ip_blacklist(cfg.db_dsn, include_expired=all)
    if not rows:
        console.print("[dim]No blacklisted IPs.[/dim]")
        return
    table = Table(show_header=True)
    table.add_column("IP", style="cyan", no_wrap=True)
    table.add_column("Reason")
    table.add_column("Hits", no_wrap=True)
    table.add_column("Created", no_wrap=True)
    table.add_column("Expires", no_wrap=True)
    for row in rows:
        table.add_row(
            row.ip,
            row.reason,
            str(row.hit_count),
            row.created_at.isoformat() if row.created_at else "-",
            row.expires_at.isoformat() if row.expires_at else "never",
        )
    console.print(table)


@blacklist_app.command("add")
def blacklist_add(
    ip: str = typer.Argument(..., help="Client IP to blacklist."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    reason: str = typer.Option("manual", "--reason", "-r", help="Ban reason."),
    ttl: Optional[int] = typer.Option(
        None,
        "--ttl",
        help=("Seconds until expiry. Default: permanent " "(or api.blacklist_ttl_seconds)."),
    ),
) -> None:
    """Manually blacklist an IP."""
    from pawn_agent.utils.db import add_ip_blacklist  # noqa: PLC0415
    from pawn_server.core.ip_guard import normalize_ip  # noqa: PLC0415

    cfg = _load_blacklist_cfg(config)
    effective_ttl = ttl if ttl is not None else cfg.api.blacklist_ttl_seconds
    row = add_ip_blacklist(
        cfg.db_dsn,
        normalize_ip(ip),
        reason=reason,
        ttl_seconds=effective_ttl,
    )
    expires = row.expires_at.isoformat() if row.expires_at else "never"
    console.print(
        f"[green]Blacklisted {row.ip}[/green] " f"reason={row.reason!r} expires={expires}"
    )


@blacklist_app.command("remove")
def blacklist_remove(
    ip: str = typer.Argument(..., help="Client IP to unban."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Remove an IP from the blacklist."""
    from pawn_agent.utils.db import remove_ip_blacklist  # noqa: PLC0415
    from pawn_server.core.ip_guard import normalize_ip  # noqa: PLC0415

    cfg = _load_blacklist_cfg(config)
    if remove_ip_blacklist(cfg.db_dsn, normalize_ip(ip)):
        console.print(f"[green]Removed {ip} from blacklist.[/green]")
    else:
        console.print(f"[dim]{ip} was not on the blacklist.[/dim]")
        raise typer.Exit(1)


@blacklist_app.command("clear")
def blacklist_clear(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Remove every IP from the blacklist."""
    from pawn_agent.utils.db import clear_ip_blacklist  # noqa: PLC0415

    cfg = _load_blacklist_cfg(config)
    if not yes and not typer.confirm("Clear the entire API IP blacklist?", default=False):
        raise typer.Exit(0)
    n = clear_ip_blacklist(cfg.db_dsn)
    noun = "y" if n == 1 else "ies"
    console.print(f"[green]Cleared {n} blacklist entr{noun}.[/green]")


@coworker_app.command("process")
def coworker_process(
    session: str = typer.Option(..., "--session", "-s", help="Diarization session id."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run the coworker loop for one finished session."""
    import asyncio  # noqa: PLC0415

    from pawn_agent.core.coworker.pipeline import process_session  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    cfg = load_config(config)
    result = asyncio.run(process_session(cfg, session))
    console.print(result)


@coworker_app.command("reindex")
def coworker_reindex(
    notes: bool = typer.Option(False, "--notes", help="Index vault markdown."),
    sessions: bool = typer.Option(False, "--sessions", help="Index diarization transcripts."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Backfill the knowledge index. With no flags, indexes both."""
    from sqlalchemy import select  # noqa: PLC0415
    from sqlalchemy.orm import Session  # noqa: PLC0415

    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_agent.utils.transcript import fetch_transcript  # noqa: PLC0415
    from pawn_core.database import get_engine  # noqa: PLC0415
    from pawn_core.knowledge_index import index_text  # noqa: PLC0415
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415
    from pawn_diarize.core.database import TranscriptionSegment  # noqa: PLC0415

    cfg = load_config(config)
    do_notes = notes or not sessions
    do_sessions = sessions or not notes
    count = 0
    if do_notes:
        store = vault_store_from_config(cfg)
        for key in store.list(""):
            if key.startswith(f"{cfg.vault.agent_root.strip('/')}/") or key.startswith(
                ".obsidian/"
            ):
                continue
            try:
                index_text(cfg, source_kind="note", source_ref=key, text=store.read(key))
                count += 1
            except Exception as exc:
                console.print(f"[yellow]{key}: {exc}[/yellow]")
    if do_sessions:
        with Session(get_engine(cfg.db_dsn)) as db:
            ids = db.scalars(select(TranscriptionSegment.session_id).distinct()).all()
        for session_id in ids:
            text = fetch_transcript(cfg, session_id)
            if text.startswith("Error") or text.startswith("No transcript"):
                continue
            index_text(cfg, source_kind="transcript", source_ref=session_id, text=text)
            count += 1
    console.print(f"[green]Indexed {count} sources.[/green]")


@coworker_app.command("pause")
def coworker_pause(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """Stop unattended follow-ups until resume."""
    from pawn_agent.core.coworker.autonomy import set_paused  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    path = set_paused(load_config(config), True)
    console.print(f"[yellow]Coworker paused ({path}).[/yellow]")


@coworker_app.command("resume")
def coworker_resume(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """Allow unattended follow-ups again."""
    from pawn_agent.core.coworker.autonomy import set_paused  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    set_paused(load_config(config), False)
    console.print("[green]Coworker resumed.[/green]")


@coworker_app.command("status")
def coworker_status(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """Show whether the coworker loop is enabled and paused."""
    from pawn_agent.core.coworker.autonomy import effective_mode, is_paused  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    cfg = load_config(config)
    console.print(
        f"enabled={cfg.coworker.enabled} paused={is_paused(cfg)} mode={effective_mode(cfg)}"
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
    no_vault_watcher: bool = typer.Option(
        False,
        "--no-vault-watcher",
        help="Disable the vault task watcher even if vault_watcher.enabled is true.",
    ),
    vault_watcher_only: bool = typer.Option(
        False,
        "--vault-watcher-only",
        help="Run only the vault task watcher (no HTTP API, queue, or scheduler).",
    ),
    no_coworker: bool = typer.Option(
        False,
        "--no-coworker",
        help="Disable the coworker briefing loop even if coworker.enabled is true.",
    ),
    coworker_only: bool = typer.Option(
        False,
        "--coworker-only",
        help="Run only the coworker loop (briefing, review, vault scan).",
    ),
    ssl_certfile: Optional[str] = typer.Option(
        None,
        "--ssl-certfile",
        help="TLS certificate PEM path. Overrides api.ssl_certfile.",
    ),
    ssl_keyfile: Optional[str] = typer.Option(
        None,
        "--ssl-keyfile",
        help="TLS private key PEM path. Overrides api.ssl_keyfile.",
    ),
) -> None:
    """Start the HTTP API and optional workers (queue, scheduler, Matrix).

    Workers are armed from config and can be forced off with ``--no-*`` flags.
    Use ``--matrix-only`` or ``--scheduler-only`` for a single-worker process.

    \b
    API Endpoints
    -------------
    POST   /v1/chat/completions    OpenAI-compatible chat (Bearer token required)
    DELETE /sessions/{session_id}  Clear a session (Bearer token required)
    POST   /knowledge              Index content into RAG (Bearer token required)
    GET    /health                 Liveness probe (no auth)
    GET    /docs                   Swagger UI (when api.enable_docs)
    GET    /openapi.json           OpenAPI spec (when api.enable_docs)

    \b
    Queue message format
    --------------------
    {
      "command": "run",
      "prompt":  "Summarise session abc123",
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

    only_flags = sum(
        bool(x) for x in (scheduler_only, matrix_only, vault_watcher_only, coworker_only)
    )
    if only_flags > 1:
        console.print(
            "[red]Use only one of --scheduler-only / --matrix-only / "
            "--vault-watcher-only / --coworker-only.[/red]"
        )
        raise typer.Exit(1)

    effective_host = host or cfg.api_host
    effective_port = port or cfg.api_port

    try:
        effective_cert, effective_key = resolve_ssl_files(
            ssl_certfile if ssl_certfile is not None else cfg.api.ssl_certfile,
            ssl_keyfile if ssl_keyfile is not None else cfg.api.ssl_keyfile,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    queue_cfg = cfg.queue_config or {}
    with_queue = not no_queue and bool(queue_cfg)
    with_scheduler = bool(cfg.agent_scheduler.enabled) and not disable_scheduler
    with_matrix = bool(cfg.matrix_bot.enabled) and not no_matrix
    with_matrix_notifier = with_matrix and matrix_notifier_enabled(cfg) and not no_matrix
    with_vault_watcher = bool(cfg.vault_watcher.enabled) and not no_vault_watcher
    with_coworker = bool(cfg.coworker.enabled) and not no_coworker
    effective_topic = topic or queue_cfg.get("topic", DEFAULT_TOPIC)
    effective_consumer = consumer_name or queue_cfg.get("consumer_name", DEFAULT_CONSUMER_NAME)
    only_mode = scheduler_only or matrix_only or vault_watcher_only or coworker_only

    bf = "on" if cfg.api.bruteforce_enabled else "off"
    bf_detail = (
        f"{bf} (auth≥{cfg.api.auth_fail_threshold} "
        f"404≥{cfg.api.not_found_threshold}"
        f"/{cfg.api.bruteforce_window_seconds}s)"
    )
    ssl_line = f"enabled ({effective_cert})" if effective_cert else "disabled"
    console.print(
        f"[bold green]pawn-server serve starting[/bold green]\n"
        f"  host     : [cyan]{effective_host}[/cyan]\n"
        f"  port     : [cyan]{effective_port}[/cyan]\n"
        f"  ssl      : [dim]{ssl_line}[/dim]\n"
        f"  model    : [dim]{cfg.pydantic_model}[/dim]\n"
        f"  idle     : [dim]{cfg.api_model_idle_timeout_minutes} min[/dim]\n"
        f"  auth     : [dim]{'token set' if cfg.api_token else 'NO TOKEN — open access'}[/dim]\n"
        f"  docs     : [dim]{'enabled' if cfg.api.enable_docs else 'disabled'}[/dim]\n"
        f"  bruteforce: [dim]{bf_detail}[/dim]\n"
        f"  queue    : [dim]{'topic=' + effective_topic + ' consumer=' + effective_consumer if with_queue and not only_mode else 'disabled'}[/dim]\n"
        f"  scheduler: [dim]{'enabled' if with_scheduler and not matrix_only else 'disabled'}[/dim]\n"
        f"  matrix   : [dim]{'enabled' if with_matrix and not scheduler_only else 'disabled'}[/dim]\n"
        f"  notify   : [dim]{'via matrix bot' if with_matrix_notifier and with_matrix and not scheduler_only else 'disabled'}[/dim]\n"
        f"  vault    : [dim]{'enabled' if with_vault_watcher and not scheduler_only and not matrix_only and not coworker_only else 'disabled'}[/dim]\n"
        f"  coworker : [dim]{'enabled' if with_coworker and not scheduler_only and not matrix_only else 'disabled'}[/dim]"
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

        if vault_watcher_only:
            if not with_vault_watcher:
                raise RuntimeError("Vault watcher is disabled by config or --no-vault-watcher")
            from pawn_server.core.vault_watcher import start_vault_watcher  # noqa: PLC0415

            await start_vault_watcher(cfg)
            return

        if coworker_only:
            if not with_coworker:
                raise RuntimeError("Coworker is disabled by config or --no-coworker")
            from pawn_server.core.coworker_worker import start_coworker  # noqa: PLC0415

            await start_coworker(cfg)
            return

        fastapi_app = create_app(cfg)
        from pawn_server.core.api_server import get_sallm_registry  # noqa: PLC0415

        shared_registry = get_sallm_registry()
        from pawn_server.core.job_events import job_events  # noqa: PLC0415

        class _Server(uvicorn.Server):
            # Long-lived SSE streams (e.g. /v1/jobs/events) otherwise keep
            # uvicorn in "Waiting for connections to close" forever.
            def handle_exit(self, sig: int, frame: Any) -> None:
                job_events.close()
                super().handle_exit(sig, frame)

        uv_config = uvicorn.Config(
            fastapi_app,
            host=effective_host,
            port=effective_port,
            log_level="info",
            timeout_graceful_shutdown=5,
            ssl_certfile=effective_cert,
            ssl_keyfile=effective_key,
        )
        server = _Server(uv_config)

        if (
            not with_queue
            and not with_scheduler
            and not with_matrix
            and not with_vault_watcher
            and not with_coworker
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
        if with_vault_watcher:
            from pawn_server.core.vault_watcher import start_vault_watcher  # noqa: PLC0415

            tasks.append(asyncio.create_task(start_vault_watcher(cfg, registry=shared_registry)))
        if with_coworker:
            from pawn_server.core.coworker_worker import start_coworker  # noqa: PLC0415
            from pawn_server.core.jobs import recover_abandoned_jobs  # noqa: PLC0415

            try:
                recovered = await recover_abandoned_jobs(cfg, shared_registry)
                if any(recovered.values()):
                    console.print(f"[dim]recovered jobs: {recovered}[/dim]")
            except Exception as exc:
                console.print(f"[yellow]job recovery skipped: {exc}[/yellow]")
            tasks.append(asyncio.create_task(start_coworker(cfg)))

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

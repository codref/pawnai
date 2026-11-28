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
portal_app = typer.Typer(
    name="portal",
    help="Pawn Portal maintenance: import existing sallm chats.",
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(coworker_app, name="coworker")
emergence_app = typer.Typer(
    help="Emergence layer: threads, ledger, night shift, provocations.",
    no_args_is_help=True,
)
app.add_typer(emergence_app, name="emergence")
app.add_typer(portal_app, name="portal")


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
    """Run the coworker item loop for one finished session.

    Creates ``Pawn/Items/`` notes from decisions / commitments / open questions.
    Does **not** refresh People bios — use ``people-refresh`` for that.
    """
    import asyncio  # noqa: PLC0415

    from pawn_agent.core.coworker.pipeline import process_session  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    cfg = load_config(config)
    result = asyncio.run(process_session(cfg, session))
    console.print(result)


@coworker_app.command("people-refresh")
def coworker_people_refresh(
    session: str = typer.Option(..., "--session", "-s", help="Diarization session id."),
    force: bool = typer.Option(
        True,
        "--force/--no-force",
        help=(
            "Re-run even if a refresh already completed today (default: force). "
            "Use --no-force to skip when a completed speakers_refresh exists."
        ),
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Update People/ vault bios for gallery speakers in one session.

    Ensures stubs, appends Appearances, and proposes or applies Facts under
    autonomy ``people_refresh``. Does **not** create coworker Items
    (unlike ``coworker process``).

    Runs even when ``coworker.enabled`` is false — this is an explicit
    operator action.
    """
    import asyncio  # noqa: PLC0415

    from pawn_agent.core.people.refresh import refresh_people_for_session  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    cfg = load_config(config)
    result = asyncio.run(refresh_people_for_session(cfg, session, force=bool(force)))
    console.print(result)


@coworker_app.command("observations")
def coworker_observations(
    session: str = typer.Option(..., "--session", "-s", help="Source ref (session id)."),
    rejected: bool = typer.Option(False, "--rejected", help="Also list rejected ones."),
    type_name: Optional[str] = typer.Option(None, "--type", "-t", help="Only this type."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """List stored observations from the latest emergence run for a source."""
    from pawn_agent.core.coworker.notes import format_offset  # noqa: PLC0415
    from pawn_agent.core.emergence.store import get_source, list_observations  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    cfg = load_config(config)
    source = get_source(cfg.db_dsn, session)
    if source is None:
        console.print(
            f"[yellow]No emergence run for {session!r}. Enable emergence and run "
            "`pawn-server coworker process`.[/yellow]"
        )
        raise typer.Exit(1)
    statuses = ("live", "rejected") if rejected else ("live",)
    rows = list_observations(cfg.db_dsn, session, statuses=statuses)
    if type_name:
        rows = [r for r in rows if r["type"] == type_name]
    table = Table(title=f"{session} — {source['stats']}")
    for col in ("at", "type", "status", "speaker", "text"):
        table.add_column(col)
    for row in rows:
        table.add_row(
            format_offset(row["start_sec"]),
            row["type"],
            row["status"] if row["status"] == "live" else f"{row['reject_reason']}",
            row["speaker_name"],
            row["text"],
        )
    console.print(table)


@coworker_app.command("compare")
def coworker_compare(
    session: str = typer.Option(..., "--session", "-s", help="Diarization session id."),
    out: Optional[str] = typer.Option(
        None,
        "--out",
        "-o",
        help="Vault key for the report (default Pawn/Reviews/extract-compare-{session}.md); "
        "'-' prints it.",
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run the single-prompt and observation extractors side by side.

    Writes a review note with both item lists to tick. Creates no Items and
    stores no observations, and works whether or not emergence is enabled.
    """
    import asyncio  # noqa: PLC0415

    from pawn_agent.core.coworker.pipeline import session_source_text  # noqa: PLC0415
    from pawn_agent.core.emergence.compare import render_report, run_comparison  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

    cfg = load_config(config)
    store = vault_store_from_config(cfg)
    text = session_source_text(cfg, session, store)
    report = render_report(asyncio.run(run_comparison(cfg, source_ref=session, source_text=text)))
    if out == "-":
        console.print(report, markup=False, highlight=False)
        return
    key = out or f"{cfg.coworker.reviews_dir.rstrip('/')}/extract-compare-{session}.md"
    store.write(key, report)
    console.print(f"[green]Wrote {key}[/green]")


labels_app = typer.Typer(help="Emergence labels: your verdicts Pawn learns from.")
coworker_app.add_typer(labels_app, name="labels")


@labels_app.command("import")
def coworker_labels_import(
    note: Optional[str] = typer.Option(
        None, "--note", "-n", help="One review note key (default: every extract-compare note)."
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Read ticks from compare review notes into labels.

    Ticked = wanted. Unticked = not wanted only when the note's frontmatter
    says ``reviewed: true``. Re-importing a note replaces its labels.
    """
    from pawn_agent.core.emergence.labels import (  # noqa: PLC0415
        import_review_note,
        import_review_notes,
    )
    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

    cfg = load_config(config)
    store = vault_store_from_config(cfg)
    results = (
        {note: import_review_note(cfg, note, store.read(note))}
        if note
        else import_review_notes(cfg, store)
    )
    table = Table(title="Imported labels")
    for col in ("note", "accept", "reject", "skipped"):
        table.add_column(col)
    for key, counts in results.items():
        table.add_row(key, str(counts["accept"]), str(counts["reject"]), str(counts["skipped"]))
    console.print(table)


@labels_app.command("stats")
def coworker_labels_stats(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """Label counts by surface and verdict, plus the ranker's state."""
    from pawn_agent.core.emergence import ranker  # noqa: PLC0415
    from pawn_agent.core.emergence.labels import label_counts  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    cfg = load_config(config)
    counts = label_counts(cfg.db_dsn)
    table = Table(title=f"{counts['total']} labels")
    for col in ("surface", "verdict", "count"):
        table.add_column(col)
    for surface, verdict, n in counts["by"]:
        table.add_row(surface, verdict, str(n))
    console.print(table)
    model = ranker.load(cfg)
    console.print(
        {k: v for k, v in (model or {"status": "untrained"}).items() if k not in {"w", "b"}}
    )


@coworker_app.command("ranker")
def coworker_ranker(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """Retrain the relevance ranker on all labels and print its cross-validated AUC."""
    from pawn_agent.core.emergence import ranker  # noqa: PLC0415
    from pawn_agent.core.emergence.labels import label_counts  # noqa: PLC0415
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    cfg = load_config(config)
    meta = ranker.train(cfg, label_total=label_counts(cfg.db_dsn)["total"])
    console.print(meta)


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


@portal_app.command("import")
def portal_import(
    prefix: Optional[list[str]] = typer.Option(
        None,
        "--prefix",
        "-p",
        help="sallm conversation-key prefix to import (repeatable; default: chat: and matrix:).",
    ),
    all_sessions: bool = typer.Option(
        False, "--all", help="Import every sallm session that has chat turns."
    ),
    folder: str = typer.Option(
        "Imported", "--folder", help="Portal chat folder for imported chats ('' for none)."
    ),
    tag: Optional[list[str]] = typer.Option(None, "--tag", "-t", help="Tag to add (repeatable)."),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Only the N most recently active sessions."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be imported."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Bring sallm conversations that predate the portal into its chat list.

    Each imported chat keeps its sallm session id, so replying from the portal
    continues the same memory. Only user/assistant turns are mirrored. Sessions
    already in the portal are skipped, so this is safe to re-run.
    """
    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_server.core.portal_import import (  # noqa: PLC0415
        DEFAULT_PREFIXES,
        import_sallm_chats,
    )

    cfg = load_config(config)
    prefixes = ("*",) if all_sessions else tuple(prefix or DEFAULT_PREFIXES)
    try:
        report = import_sallm_chats(
            cfg,
            prefixes=prefixes,
            folder=folder or None,
            tags=tag or (),
            limit=limit,
            dry_run=dry_run,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if report.imported:
        table = Table(title="Would import" if dry_run else "Imported")
        table.add_column("Conversation")
        table.add_column("Title")
        table.add_column("Msgs", justify="right")
        for key, title, n in report.imported:
            table.add_row(key, title, str(n))
        console.print(table)
    verb = "Would import" if dry_run else "Imported"
    console.print(
        f"{verb} {len(report.imported)} chat(s); skipped {len(report.skipped_existing)} "
        f"already in the portal and {len(report.skipped_empty)} without chat turns."
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
    from pawn_agent.utils.model_catalog import get_background_model  # noqa: PLC0415

    model_label = get_background_model(cfg) or cfg.chat_model_id
    console.print(
        f"[bold green]pawn-server serve starting[/bold green]\n"
        f"  host     : [cyan]{effective_host}[/cyan]\n"
        f"  port     : [cyan]{effective_port}[/cyan]\n"
        f"  ssl      : [dim]{ssl_line}[/dim]\n"
        f"  model    : [dim]{model_label}[/dim]\n"
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
        from pawn_server.core.vault_events import vault_events  # noqa: PLC0415

        class _Server(uvicorn.Server):
            # Long-lived SSE streams (e.g. /v1/jobs/events) and vault long-polls
            # otherwise keep uvicorn in "Waiting for connections to close".
            def handle_exit(self, sig: int, frame: Any) -> None:
                job_events.close()
                vault_events.close()
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

        # Portal full-text file index: background reconcile next to the API.
        from pawn_server.core.file_index import run_index_loop  # noqa: PLC0415

        index_task = asyncio.create_task(run_index_loop(cfg))
        try:
            await _run_workers(server, shared_registry)
        finally:
            index_task.cancel()
            try:
                await index_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run_workers(server: Any, shared_registry: Any) -> None:
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


# ── emergence ────────────────────────────────────────────────────────────────


def _em_cfg(config: Optional[str]) -> Any:
    from pawn_agent.utils.config import load_config  # noqa: PLC0415

    return load_config(config)


@emergence_app.command("night")
def emergence_night(
    job: Optional[list[str]] = typer.Option(None, "--job", "-j", help="Run only these jobs."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run the night shift now (sync, clustering, synthesis, ledger, …)."""
    import asyncio  # noqa: PLC0415

    from pawn_agent.core.emergence.night import JOBS, run_night  # noqa: PLC0415

    unknown = [j for j in job or [] if j not in JOBS]
    if unknown:
        console.print(f"[red]Unknown jobs {unknown}. Choose from {', '.join(JOBS)}.[/red]")
        raise typer.Exit(1)
    cfg = _em_cfg(config)
    console.print(asyncio.run(run_night(cfg, jobs=list(job) if job else None)))


@emergence_app.command("subjects")
def emergence_subjects(
    kind: Optional[str] = typer.Option(None, "--kind", "-k", help="thread, question, decision, …"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Comma-separated."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """List subjects with momentum and status."""
    from pawn_agent.core.emergence import subjects as subj  # noqa: PLC0415

    cfg = _em_cfg(config)
    rows = subj.list_subjects(
        cfg.db_dsn,
        kinds=[kind] if kind else None,
        statuses=[s for s in (status or "").split(",") if s] or None,
    )
    table = Table(title=f"{len(rows)} subjects")
    for col in ("id", "kind", "status", "origin", "momentum", "title"):
        table.add_column(col)
    for r in rows:
        table.add_row(r["id"][:8], r["kind"], r["status"], r["origin"],
                      "" if r["momentum"] is None else str(r["momentum"]), r["title"][:80])
    console.print(table)


@emergence_app.command("status")
def emergence_status(
    subject_id: str = typer.Argument(..., help="Subject id (prefix ok)."),
    status: str = typer.Argument(..., help="active, dismissed, done, parked, resolved, open"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Accept, drop or close a subject."""
    from pawn_agent.core.emergence import subjects as subj  # noqa: PLC0415
    from pawn_agent.core.emergence.threads import USER_STATUSES  # noqa: PLC0415

    cfg = _em_cfg(config)
    if status not in USER_STATUSES:
        console.print(f"[red]Status must be one of {sorted(USER_STATUSES)}.[/red]")
        raise typer.Exit(1)
    match = [s for s in subj.list_subjects(cfg.db_dsn) if s["id"].startswith(subject_id)]
    if len(match) != 1:
        console.print(f"[red]{len(match)} subjects match {subject_id!r}.[/red]")
        raise typer.Exit(1)
    subj.set_status(cfg.db_dsn, match[0]["id"], status, reason="cli")
    console.print(f"[green]{match[0]['title']} → {status}[/green]")


@emergence_app.command("provocations")
def emergence_provocations(
    days: int = typer.Option(14, "--days"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Recent provocations and verdicts."""
    from pawn_agent.core.emergence.provocations import list_recent  # noqa: PLC0415

    cfg = _em_cfg(config)
    table = Table(title="Provocations")
    for col in ("id", "day", "status", "verdict", "score", "text"):
        table.add_column(col)
    for p in list_recent(cfg, days=days):
        table.add_row(p["short_id"], p["day"], p["status"], p["verdict"] or "",
                      "" if p["score"] is None else str(p["score"]), p["text"][:90])
    console.print(table)


@emergence_app.command("verdict")
def emergence_verdict(
    short_id: str = typer.Argument(...),
    verdict: str = typer.Argument(..., help="useful | wrong | not_now"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Record a verdict on a provocation."""
    from pawn_agent.core.emergence.provocations import apply_verdict  # noqa: PLC0415

    console.print(apply_verdict(_em_cfg(config), short_id, verdict))


@emergence_app.command("council")
def emergence_council(
    subject_id: str = typer.Argument(..., help="Decision id (prefix ok)."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Run the council (five personas + pre-mortem) on one decision."""
    import asyncio  # noqa: PLC0415

    from pawn_agent.core.emergence import subjects as subj  # noqa: PLC0415
    from pawn_agent.core.emergence.dream import council  # noqa: PLC0415
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

    cfg = _em_cfg(config)
    match = [s for s in subj.list_subjects(cfg.db_dsn, kinds=("decision",))
             if s["id"].startswith(subject_id)]
    if len(match) != 1:
        console.print(f"[red]{len(match)} decisions match {subject_id!r}.[/red]")
        raise typer.Exit(1)
    key = asyncio.run(council(cfg, vault_store_from_config(cfg), match[0]["id"]))
    console.print(f"[green]Wrote {key}[/green]")


@emergence_app.command("retro")
def emergence_retro(
    kind: str = typer.Option("week", "--kind", "-k", help="week | month | quarter"),
    force: bool = typer.Option(True, "--force/--no-force", help="Rewrite an existing retro."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Write the retrospective for the last completed week, month or quarter."""
    import asyncio  # noqa: PLC0415

    from pawn_agent.core.emergence.reflect import write_retro  # noqa: PLC0415
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

    cfg = _em_cfg(config)
    key = asyncio.run(write_retro(cfg, vault_store_from_config(cfg), kind, force=force))
    console.print(f"[green]Wrote {key}[/green]" if key else "Nothing to write.")


@emergence_app.command("prompt-search")
def emergence_prompt_search(
    type_name: str = typer.Option(..., "--type", "-t", help="decision, commitment, block, …"),
    variants: int = typer.Option(2, "--variants"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Try prompt variants against your labels; promote a clear winner."""
    import asyncio  # noqa: PLC0415

    from pawn_agent.core.emergence.search import prompt_search  # noqa: PLC0415
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

    cfg = _em_cfg(config)
    console.print(asyncio.run(
        prompt_search(cfg, vault_store_from_config(cfg), type_name, variants=variants)
    ))


@emergence_app.command("export")
def emergence_export(
    out: str = typer.Option("emergence-labels.jsonl", "--out", "-o"),
    include_bulk: bool = typer.Option(False, "--include-bulk"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Export labels as JSONL for fine-tuning a model outside Pawn."""
    from pathlib import Path  # noqa: PLC0415

    from pawn_agent.core.emergence.search import export_labels  # noqa: PLC0415

    cfg = _em_cfg(config)
    n = export_labels(cfg.db_dsn, Path(out), include_bulk=include_bulk)
    console.print(f"[green]Wrote {n} labels to {out}[/green]")


@emergence_app.command("render")
def emergence_render(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """Re-render every subject note and the ledger pages."""
    from pawn_agent.core.emergence import subjects as subj  # noqa: PLC0415
    from pawn_agent.core.emergence import vault_notes  # noqa: PLC0415
    from pawn_agent.core.emergence.threads import render_all  # noqa: PLC0415
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

    cfg = _em_cfg(config)
    store = vault_store_from_config(cfg)
    n = render_all(cfg, store)
    vault_notes.render_commitments(cfg, store, subj.list_subjects(cfg.db_dsn, kinds=("commitment",)))
    vault_notes.render_predictions(cfg, store, subj.list_subjects(cfg.db_dsn, kinds=("prediction",)))
    console.print(f"[green]Rendered {n} subject notes plus Commitments and Predictions.[/green]")


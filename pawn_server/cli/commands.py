"""CLI commands for pawn-server."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="pawn-server",
    help="HTTP API server and queue listener for pawn-agent.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


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
    effective_topic = topic or queue_cfg.get("topic", DEFAULT_TOPIC)
    effective_consumer = consumer_name or queue_cfg.get("consumer_name", DEFAULT_CONSUMER_NAME)

    console.print(
        f"[bold green]pawn-server serve starting[/bold green]\n"
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

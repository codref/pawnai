"""PydanticAI-based agent for pawn-agent.

:class:`PydanticAgent` is the primary agent implementation.  It uses
``pydantic_ai.Agent`` as the LLM runner and registers all auto-discovered
tools from :mod:`pawn_agent.tools`.  The GitHub Copilot SDK is available as a
sub-agent for tools that need it (e.g. :mod:`pawn_agent.tools.analyze_custom`).

Usage::

    from pawn_agent.core.pydantic_agent import PydanticAgent
    from pawn_agent.utils.config import load_config

    cfg = load_config()
    agent = PydanticAgent(cfg=cfg)
    print(agent.run("Summarise session abc123 and save it to SiYuan"))
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from prompt_toolkit import PromptSession
from pydantic_ai import Agent

from pawn_agent.core.session_store import context_size
from pawn_agent.core.session_vars import SessionVars
from pawn_agent.tools import build_tools
from pawn_agent.utils.config import AgentConfig

logger = logging.getLogger(__name__)

_MLFLOW_CONNECT_TIMEOUT = 2  # seconds
# Keep this marker plain-text (no angle brackets / markdown-sensitive chars)
# so UI renderers don't treat it as HTML and strip it.
LISTEN_ONLY_BYPASS_MARKER = "PAWN_LISTEN_ONLY_BYPASS"
_RAW_TOOL_COMMAND_RES = (
    re.compile(r"^\s*/(?P<tool>[a-zA-Z_][a-zA-Z0-9_]*)\s*\("),
    re.compile(r"^\s*call:(?P<tool>[a-zA-Z_][a-zA-Z0-9_]*)\s*\{"),
)


class _SyntheticRunResult:
    """Minimal run-result shape used for deterministic listen-only bypass turns."""

    def __init__(self, *, output: str, new_messages: list, all_messages: list) -> None:
        self.output = output
        self._new_messages = new_messages
        self._all_messages = all_messages

    def new_messages(self) -> list:
        return list(self._new_messages)

    def all_messages(self) -> list:
        return list(self._all_messages)


def _latest_tool_interaction(messages: list) -> tuple[str | None, object, object]:
    """Return the most recent ``(tool_name, tool_args, tool_return_content)``."""
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart  # noqa: PLC0415

    latest_call_name: str | None = None
    latest_call_args: object = None
    latest_return_name: str | None = None
    latest_return_content: object = None

    for message in messages:
        for part in getattr(message, "parts", []):
            if isinstance(part, ToolCallPart):
                latest_call_name = part.tool_name
                latest_call_args = part.args
            elif isinstance(part, ToolReturnPart):
                latest_return_name = part.tool_name
                latest_return_content = part.content

    if latest_return_name is not None:
        return latest_return_name, latest_call_args, latest_return_content
    return latest_call_name, latest_call_args, None


def _preferred_tool_interaction(messages: list) -> tuple[str | None, object, object]:
    """Prefer save-related tool interactions when deciding what to show the user."""
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart  # noqa: PLC0415

    interactions: list[tuple[str, object, object]] = []
    latest_args_by_name: dict[str, object] = {}

    for message in messages:
        for part in getattr(message, "parts", []):
            if isinstance(part, ToolCallPart):
                latest_args_by_name[part.tool_name] = part.args
                interactions.append((part.tool_name, part.args, None))
            elif isinstance(part, ToolReturnPart):
                interactions.append(
                    (part.tool_name, latest_args_by_name.get(part.tool_name), part.content)
                )

    for tool_name, tool_args, tool_return in reversed(interactions):
        if tool_name == "save_to_siyuan":
            return tool_name, tool_args, tool_return
        if (
            tool_name == "analyze_custom"
            and isinstance(tool_args, dict)
            and tool_args.get("save")
        ):
            return tool_name, tool_args, tool_return

    for tool_name, tool_args, tool_return in reversed(interactions):
        if tool_return is not None:
            return tool_name, tool_args, tool_return

    return _latest_tool_interaction(messages)


def _friendly_tool_completion(tool_name: str, tool_args: object, tool_return: object) -> str:
    """Build a user-facing completion string for a successful tool action."""
    if isinstance(tool_return, str):
        stripped = tool_return.strip()
        if stripped.lower().startswith("error"):
            return stripped
    else:
        stripped = ""
    siyuan_url_match = re.search(r"siyuan://blocks/[A-Za-z0-9\-]+", stripped)
    siyuan_url = siyuan_url_match.group(0) if siyuan_url_match else None

    if tool_name == "save_to_siyuan":
        title = tool_args.get("title") if isinstance(tool_args, dict) else None
        session_id = tool_args.get("session_id") if isinstance(tool_args, dict) else None
        parts = ["Saved the content to SiYuan"]
        if title:
            parts.append(f"as {title!r}")
        if session_id:
            parts.append(f"for session {session_id!r}")
        message = " ".join(parts) + "."
        if siyuan_url:
            message += f" URL: {siyuan_url}"
        return message

    if tool_name == "analyze_custom":
        title = tool_args.get("title") if isinstance(tool_args, dict) else None
        session_id = tool_args.get("session_id") if isinstance(tool_args, dict) else None
        save_requested = bool(tool_args.get("save")) if isinstance(tool_args, dict) else False
        if save_requested:
            parts = ["Created the SiYuan page with the requested analysis"]
            if title:
                parts.append(f"as {title!r}")
            if session_id:
                parts.append(f"for session {session_id!r}")
            message = " ".join(parts) + "."
            if siyuan_url:
                message += f" URL: {siyuan_url}"
            return message
        if stripped and len(stripped) <= 160 and "\n" not in stripped:
            return stripped
        return "Completed the requested custom analysis."

    if stripped and len(stripped) <= 160 and "\n" not in stripped:
        return stripped
    return f"Completed the requested {tool_name} action."


def _extract_siyuan_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"siyuan://blocks/[A-Za-z0-9\-]+", value)
    return match.group(0) if match else None


def _maybe_append_siyuan_url(output_text: str, tool_name: str | None, tool_args: object, tool_return: object) -> str:
    """Append a SiYuan deep link when a save actually happened but the model omitted it."""
    if not output_text or output_text == LISTEN_ONLY_BYPASS_MARKER:
        return output_text

    siyuan_url = _extract_siyuan_url(output_text) or _extract_siyuan_url(tool_return)
    if not siyuan_url:
        return output_text

    if tool_name == "save_to_siyuan":
        return output_text if _extract_siyuan_url(output_text) else f"{output_text}\n\nURL: {siyuan_url}"

    if tool_name == "analyze_custom" and isinstance(tool_args, dict) and tool_args.get("save"):
        return output_text if _extract_siyuan_url(output_text) else f"{output_text}\n\nURL: {siyuan_url}"

    return output_text


def _finalize_run_result(result):
    """Hide raw tool-command text and enrich save results with SiYuan URLs."""
    output = getattr(result, "output", "")
    output_text = output if isinstance(output, str) else str(output)
    new_messages = list(result.new_messages())
    tool_name, tool_args, tool_return = _preferred_tool_interaction(new_messages)

    match = None
    for pattern in _RAW_TOOL_COMMAND_RES:
        match = pattern.match(output_text)
        if match:
            break
    replacement = output_text
    if match and tool_name in {"save_to_siyuan", "analyze_custom"}:
        replacement = _friendly_tool_completion(tool_name, tool_args, tool_return)
    elif match and tool_name == match.group("tool") and tool_name != "query_conversation":
        replacement = _friendly_tool_completion(tool_name, tool_args, tool_return)
    replacement = _maybe_append_siyuan_url(replacement, tool_name, tool_args, tool_return)

    if replacement == output_text:
        return result

    return _SyntheticRunResult(
        output=replacement,
        new_messages=new_messages,
        all_messages=list(result.all_messages()),
    )


def _coerce_tool_call_content(messages: list) -> list:
    """Ensure ModelResponse messages without text serialise with content='' not null.

    Some local OpenAI-compatible servers (e.g. gpt-oss:20b, which is written in Go)
    reject assistant messages where the ``content`` field is JSON ``null``.
    PydanticAI produces ``content=None`` whenever a ModelResponse has no TextPart
    (e.g. tool-call-only turns, or thinking-only turns after strip_thinking).

    Fix: prepend an empty TextPart to any ModelResponse that has no TextPart
    (including empty ``parts`` after ``strip_thinking``), so they serialise as
    ``{"role": "assistant", "content": "", ...}`` — valid per the OpenAI spec
    and accepted by strict local implementations.

    This covers the incoming history loaded from the session store.  The
    ``_CompatModel`` subclass in ``_resolve_model`` covers the same case for
    responses generated during the current run.
    """
    import dataclasses  # noqa: PLC0415

    from pydantic_ai.messages import ModelResponse, TextPart  # noqa: PLC0415

    result = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            has_text = any(isinstance(p, TextPart) for p in msg.parts)
            if not has_text:
                msg = dataclasses.replace(msg, parts=[TextPart(content=""), *msg.parts])
        result.append(msg)
    return result


def _make_overheard_turn(text: str, model_name: str) -> list:
    """Build a synthetic (request, response) pair for an overheard message.

    Stores the user's text in message history without generating a real reply,
    preserving the alternating request/response structure that PydanticAI expects.
    The ``[listening]`` placeholder makes the bot's silence explicit in history
    so future LLM calls have full context of the think-aloud session.
    """
    from pydantic_ai.messages import (  # noqa: PLC0415
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    return [
        ModelRequest(parts=[UserPromptPart(content=text)]),
        ModelResponse(
            parts=[TextPart(content="[listening]")],
            model_name=model_name,
            timestamp=datetime.now(timezone.utc),
        ),
    ]


def _make_listen_only_bypass_result(
    text: str,
    model_name: str,
    message_history: list | None,
) -> _SyntheticRunResult:
    """Build a synthetic result for an overheard listen-only message.

    The output marker is deterministic so API clients can filter it and avoid
    rendering placeholder text while still preserving full conversation context
    in history persistence.
    """
    new_messages = _make_overheard_turn(text, model_name)
    all_messages = list(message_history or [])
    all_messages.extend(new_messages)
    return _SyntheticRunResult(
        output=LISTEN_ONLY_BYPASS_MARKER,
        new_messages=new_messages,
        all_messages=all_messages,
    )


def _mlflow_reachable(tracking_uri: str) -> bool:
    """Return True if a TCP connection to *tracking_uri* succeeds quickly."""
    try:
        parsed = urllib.parse.urlparse(tracking_uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=_MLFLOW_CONNECT_TIMEOUT):
            return True
    except OSError:
        return False


class PydanticAgent:
    """PydanticAI agent that selects and invokes tools to fulfil a user prompt.

    Args:
        cfg: Populated :class:`~pawn_agent.utils.config.AgentConfig`.
        emit: Optional callback for outputting agent responses. Defaults to
            :func:`print`. Used by the chat REPL to render with Rich.
        on_thinking: Optional callback invoked just before the model is called.
            Used by the chat REPL to show a "thinking…" spinner.
    """

    def __init__(
        self,
        cfg: AgentConfig,
        emit: Optional[Callable[[str], None]] = None,
        on_thinking: Optional[Callable[[], None]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self._emit_fn = emit or print
        self._on_thinking = on_thinking
        self._vars = SessionVars(
            session_id=session_id,
            dsn=cfg.db_dsn if session_id else None,
        )
        if session_id and cfg.db_dsn:
            self._vars.load()
        self._configure_mlflow()
        self._agent = self._build_agent()

    _mlflow_configured = False

    def _configure_mlflow(self) -> None:
        """Enable optional MLflow tracing for PydanticAI runs."""
        if not self.cfg.mlflow_enabled or PydanticAgent._mlflow_configured:
            return
        try:
            import mlflow
            import mlflow.pydantic_ai  # noqa: F401 — ensures the submodule is loaded
        except ImportError as exc:
            logger.warning(
                "MLflow tracing is enabled but a required package is missing (%s); "
                "install mlflow[genai] to enable tracing.",
                exc,
            )
            return

        try:
            if self.cfg.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
                if not _mlflow_reachable(self.cfg.mlflow_tracking_uri):
                    logger.warning(
                        "MLflow tracking server is not reachable at %r; skipping tracing.",
                        self.cfg.mlflow_tracking_uri,
                    )
                    return
            if self.cfg.mlflow_experiment:
                mlflow.set_experiment(self.cfg.mlflow_experiment)

            mlflow.pydantic_ai.autolog()
            PydanticAgent._mlflow_configured = True
            logger.info(
                "Enabled MLflow tracing for PydanticAI (tracking_uri=%r, experiment=%r)",
                self.cfg.mlflow_tracking_uri,
                self.cfg.mlflow_experiment,
            )
        except Exception as exc:
            logger.warning("Failed to configure MLflow tracing: %s", exc)

    def _build_agent(self) -> Agent:
        model = self._resolve_model()
        tools = build_tools(self.cfg, session_vars=self._vars)
        system_prompts: list[str] = []
        anima = self._load_anima()
        if anima:
            system_prompts.append(anima)
        return Agent(
            model,
            tools=tools,
            system_prompt=tuple(system_prompts),
            output_retries=2,
        )

    def _build_direction_checker(self) -> Agent:
        """Return a minimal agent used only for listen-only direction checks.

        No tools, no system prompt — fast and cheap.  A fresh instance is built
        lazily the first time it is needed.
        """
        return Agent(self._resolve_model())

    def _direction_check_prompt(self, text: str) -> str:
        """Build the classifier prompt used to detect directed messages."""
        return (
            f'Is the following message directed at or addressed to someone named '
            f'"{self.cfg.agent_name}"?\n'
            f'Message: "{text}"\n'
            f'Answer with only "yes" or "no".'
        )

    def _resolve_model(self):
        """Return a PydanticAI model object or model string for the configured provider.

        When a custom ``base_url`` is set (Ollama, Azure, etc.) the OpenAI provider
        is configured with a custom ``AsyncOpenAI`` client so the endpoint is honoured.
        Otherwise returns the plain model string (e.g. ``"openai:gpt-4o"``).
        """
        model_str = self.cfg.pydantic_model
        api_key = self.cfg.pydantic_api_key
        base_url = self.cfg.pydantic_base_url

        if base_url and model_str.startswith("openai:"):
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            model_name = model_str[len("openai:"):]
            provider = OpenAIProvider(base_url=base_url, api_key=api_key or "no-key")

            class _CompatModel(OpenAIChatModel):
                """OpenAIChatModel that emits content='' for assistant messages that
                would otherwise use content=None, for compatibility with strict local
                model servers (e.g. gpt-oss:20b) that reject null content."""

                def _map_model_response(self, message):
                    param = super()._map_model_response(message)
                    if param.get("content") is None:
                        param["content"] = ""
                    return param

            return _CompatModel(model_name, provider=provider)

        # No custom base URL — set API key via env var if provided, then return string
        if api_key:
            if model_str.startswith("openai:"):
                os.environ.setdefault("OPENAI_API_KEY", api_key)
            elif model_str.startswith("anthropic:"):
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            elif model_str.startswith("google"):
                os.environ.setdefault("GEMINI_API_KEY", api_key)
            elif model_str.startswith("groq:"):
                os.environ.setdefault("GROQ_API_KEY", api_key)
            elif model_str.startswith("mistral:"):
                os.environ.setdefault("MISTRAL_API_KEY", api_key)

        return model_str

    async def _direction_check(self, text: str) -> bool:
        """Return True if *text* appears to be directed at this agent by name.

        Uses a context-free LLM call (no conversation history, no tools) so
        it is fast and unaffected by the ongoing session content.
        """
        if not hasattr(self, "_direction_checker"):
            self._direction_checker = self._build_direction_checker()
        result = await self._direction_checker.run(self._direction_check_prompt(text))
        answer = result.output.strip().lower()
        directed = answer.startswith("yes")
        logger.debug("Direction check for %r → %s", text[:60], "directed" if directed else "overheard")
        return directed

    def _direction_check_sync(self, text: str) -> bool:
        """Synchronous direction check variant used by ``run_sync``."""
        if not hasattr(self, "_direction_checker"):
            self._direction_checker = self._build_direction_checker()
        result = self._direction_checker.run_sync(self._direction_check_prompt(text))
        answer = result.output.strip().lower()
        directed = answer.startswith("yes")
        logger.debug("Direction check for %r → %s", text[:60], "directed" if directed else "overheard")
        return directed

    def _load_anima(self) -> str | None:
        if not self.cfg.anima_path:
            return None
        p = Path(self.cfg.anima_path)
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8").strip() or None

    def run(self, user_prompt: str) -> str:
        """Run the agent on a user prompt and return its response.

        Args:
            user_prompt: Free-text instruction from the user.

        Returns:
            Agent's final response string.
        """
        if self._on_thinking is not None:
            self._on_thinking()
        result = self.run_sync(user_prompt)
        return result.output

    def run_sync(
        self,
        user_prompt: str,
        *,
        message_history: List | None = None,
        model_settings: dict | None = None,
    ):
        """Run one synchronous agent turn and return the full PydanticAI result."""
        history = _coerce_tool_call_content(message_history) if message_history else message_history
        if self._vars.get_bool("listen_only"):
            directed = self._direction_check_sync(user_prompt)
            if not directed:
                logger.debug(
                    "Listen-only bypass (sync) for message %r → returning marker %s",
                    user_prompt[:60],
                    LISTEN_ONLY_BYPASS_MARKER,
                )
                return _make_listen_only_bypass_result(
                    user_prompt,
                    self.cfg.pydantic_model,
                    history,
                )
        result = self._agent.run_sync(
            user_prompt,
            message_history=history,
            model_settings=model_settings,
            instructions=self._vars.build_instructions() or None,
        )
        return _finalize_run_result(result)

    async def run_async(
        self,
        user_prompt: str,
        *,
        message_history: List | None = None,
        model_settings: dict | None = None,
    ):
        """Run one async agent turn and return the full PydanticAI result."""
        history = _coerce_tool_call_content(message_history) if message_history else message_history
        if self._vars.get_bool("listen_only"):
            directed = await self._direction_check(user_prompt)
            if not directed:
                logger.debug(
                    "Listen-only bypass (async) for message %r → returning marker %s",
                    user_prompt[:60],
                    LISTEN_ONLY_BYPASS_MARKER,
                )
                return _make_listen_only_bypass_result(
                    user_prompt,
                    self.cfg.pydantic_model,
                    history,
                )
        result = await self._agent.run(
            user_prompt,
            message_history=history,
            model_settings=model_settings,
            instructions=self._vars.build_instructions() or None,
        )
        return _finalize_run_result(result)

    def chat(
        self,
        first_message: str | None = None,
        initial_history: List | None = None,
        on_turn_complete: Callable[[list], None] | None = None,
        on_reset: Callable[[], None] | None = None,
    ) -> None:
        """Start an interactive multi-turn chat session.

        Threads ``message_history`` across turns so the model retains context.
        Exits on ``/exit``, ``/quit``, Ctrl-D, or Ctrl-C.
        Type ``/set key=value`` to set a session variable.
        Type ``/unset key`` to remove a session variable.
        Type ``/vars`` to list all active session variables.

        Args:
            first_message: Optional pre-built first turn (e.g. with a session
                hint already prepended). If ``None`` the REPL prompts the user
                immediately.
            initial_history: Optional message history to seed the session with
                (e.g. loaded from the session store). Defaults to empty.
            on_turn_complete: Optional callback invoked with ``result.new_messages()``
                after each turn completes. Used to persist turns to the session store.
            on_reset: Optional callback invoked when the user runs ``/reset``.
                Called after in-memory history is cleared; use it to also wipe
                persistent storage.
        """
        asyncio.run(self._repl_loop(first_message, initial_history, on_turn_complete, on_reset))

    async def _repl_loop(
        self,
        first_message: str | None,
        initial_history: List | None = None,
        on_turn_complete: Callable[[list], None] | None = None,
        on_reset: Callable[[], None] | None = None,
    ) -> None:
        message_history: list = list(initial_history or [])
        pending = first_message
        pt_session: PromptSession = PromptSession()
        while True:
            if pending is None:
                try:
                    kb, tok = context_size(message_history)
                    tok_str = f"{tok // 1000}k" if tok >= 1000 else str(tok)
                    pending = await pt_session.prompt_async(f"You [{kb:.1f} KB · ~{tok_str} tok]: ")
                except (EOFError, KeyboardInterrupt):
                    return
            text = pending.strip()
            pending = None
            if not text:
                continue
            if text.lower() in {"/exit", "/quit"}:
                return
            if text.lower() == "/reset":
                message_history = []
                if on_reset is not None:
                    on_reset()
                continue
            if text.startswith("/set "):
                rest = text[5:].strip()
                if "=" not in rest:
                    self._emit_fn("Usage: /set key=value")
                else:
                    key, _, val = rest.partition("=")
                    self._emit_fn(self._vars.set(key.strip(), val.strip()))
                continue
            if text.startswith("/unset "):
                self._emit_fn(self._vars.unset(text[7:].strip()))
                continue
            if text.lower() == "/vars":
                self._emit_fn(self._vars.format_for_display())
                continue
            if self._vars.get_bool("listen_only"):
                # Direction check: context-free LLM call to decide whether the
                # message addresses this agent by name.
                if self._on_thinking is not None:
                    self._on_thinking()
                directed = await self._direction_check(text)
                if not directed:
                    # Overheard — store in history so the full think-aloud
                    # session is available for later analysis, but emit nothing.
                    overheard = _make_overheard_turn(text, self.cfg.pydantic_model)
                    message_history.extend(overheard)
                    if on_turn_complete is not None:
                        on_turn_complete(overheard)
                    continue
                # Directed at the agent — fall through to the normal LLM call.
            if self._on_thinking is not None:
                self._on_thinking()
            result = await self.run_async(text, message_history=message_history)
            message_history = list(result.all_messages())
            if on_turn_complete is not None:
                on_turn_complete(list(result.new_messages()))
            self._emit_fn(result.output)

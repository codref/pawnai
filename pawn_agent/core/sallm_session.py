"""One durable sallm conversation bound to a conversation key.

Owns: wrapping sync ``Agent.ask`` / ``Agent.clear`` for async callers.
Does not own: pooling multiple conversations (see sallm_registry.py).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from sallm import Agent

from pawn_agent.core.sallm_factory import build_sallm_agent, rebuild_agent_for_config
from pawn_agent.utils.config import AgentConfig

logger = logging.getLogger(__name__)


def format_session_stats(snap: dict[str, Any]) -> str:
    """Render a stats snapshot as short Markdown for chat / Matrix."""
    lines = [
        "### Session stats",
        f"- **conversation**: `{snap.get('conversation_id', '?')}`",
        f"- **model**: `{snap.get('model', '?')}`",
        f"- **skill**: `{snap.get('active_skill', '?')}`",
    ]
    goal = (snap.get("goal") or "").strip()
    if goal:
        lines.append(f"- **goal**: {goal}")
    stack = snap.get("stack") or []
    if len(stack) > 1:
        stack_s = " → ".join(
            f.get("skill", "?") if isinstance(f, dict) else str(f) for f in stack
        )
        lines.append(f"- **skill stack**: {stack_s}")
    lines.append(f"- **messages**: {snap.get('message_count', 0)}")
    lines.append(f"- **memory chunks**: {snap.get('chunk_count', 0)}")
    pending = snap.get("pending_extracts", 0)
    if pending:
        lines.append(f"- **pending extracts**: {pending}")
    lines.append(f"- **max_steps**: {snap.get('max_steps', '?')}")

    last = snap.get("last_metrics") or {}
    if last:
        lines.extend(
            [
                "",
                "### Last turn",
                f"- **tokens**: in {last.get('prompt_tokens', 0)} / "
                f"out {last.get('completion_tokens', 0)} / "
                f"total {last.get('total_tokens', 0)}",
                f"- **context msgs**: {last.get('context_messages', 0)} "
                f"(prompt view {last.get('prompt_messages', 0)})",
                f"- **elapsed**: {last.get('elapsed_ms', 0)} ms",
            ]
        )
        if last.get("reasoning_tokens"):
            lines.append(f"- **reasoning tokens**: {last.get('reasoning_tokens')}")
    else:
        lines.extend(["", "_No turn metrics yet — send a message first._"])
    return "\n".join(lines)


class SallmChatSession:
    """Async façade around one sallm.Agent instance.

    ``Agent.ask`` is synchronous (LiteLLM + subprocess tools). We offload it
    with ``asyncio.to_thread`` so FastAPI / queue handlers stay responsive.
    """

    def __init__(self, agent: Agent, cfg: AgentConfig) -> None:
        self._agent = agent
        self._cfg = cfg
        self.conversation_id = agent.session_id
        self.last_metrics: dict[str, Any] = {}

    @classmethod
    def create(
        cls,
        cfg: AgentConfig,
        *,
        conversation_id: str,
        trace: Any = None,
    ) -> "SallmChatSession":
        """Build a session (sync). Safe to call from a worker thread."""
        agent = build_sallm_agent(
            cfg,
            conversation_id=conversation_id,
            trace=trace,
        )
        return cls(agent, cfg)

    def apply_config(self, cfg: AgentConfig) -> None:
        """Replace the underlying Agent when model/settings change mid-session."""
        if cfg.litellm_model == self._cfg.litellm_model and cfg.sallm == self._cfg.sallm:
            self._cfg = cfg
            return
        self._agent = rebuild_agent_for_config(self._agent, cfg)
        self._cfg = cfg

    def collect_stats(self) -> dict[str, Any]:
        """Snapshot sallm session state + last-turn metrics (sync, cheap)."""
        agent = self._agent
        repo = getattr(agent, "repo", None)
        sid = self.conversation_id
        message_count = 0
        chunk_count = 0
        pending = 0
        active_skill = "converse"
        if repo is not None:
            message_count = len(repo.list_messages(sid))
            chunk_count = len(repo.list_chunks(sid))
            pending = repo.count_pending_extracts(sid)
            active_skill = repo.active_skill(sid)
        stack = [
            {"skill": f.skill, "depth": f.depth, "note": f.note}
            for f in (getattr(agent, "stack", None) or [])
        ]
        return {
            "conversation_id": sid,
            "model": getattr(self._cfg, "litellm_model", None)
            or getattr(self._cfg, "pydantic_model", "?"),
            "active_skill": active_skill,
            "goal": getattr(agent, "goal", "") or "",
            "stack": stack,
            "message_count": message_count,
            "chunk_count": chunk_count,
            "pending_extracts": pending,
            "max_steps": getattr(agent, "max_steps", None)
            or getattr(self._cfg.sallm, "max_steps", "?"),
            "last_metrics": dict(self.last_metrics) if self.last_metrics else {},
        }

    def format_stats(self) -> str:
        return format_session_stats(self.collect_stats())

    def _ask_sync(
        self,
        text: str,
        on_progress: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ) -> Any:
        """Run ``Agent.ask`` on the calling thread; optionally bridge Tracer events."""
        if on_progress is None:
            return self._agent.ask(text)

        from sallm.trace import Tracer, multi_sink  # noqa: PLC0415

        def progress_emit(event: dict[str, Any]) -> None:
            kind = str(event.get("kind") or "")
            attrs = event.get("attrs") if isinstance(event.get("attrs"), dict) else {}
            try:
                on_progress(kind, attrs)
            except Exception:
                logger.exception("on_progress callback failed kind=%s", kind)

        old_trace = self._agent.trace
        if old_trace is not None and callable(getattr(old_trace, "emit", None)):
            previous_emit = old_trace.emit
            old_trace.emit = multi_sink(progress_emit, previous_emit)
            try:
                return self._agent.ask(text)
            finally:
                old_trace.emit = previous_emit
        else:
            self._agent.trace = Tracer(
                progress_emit,
                session_id=self.conversation_id,
            )
            try:
                return self._agent.ask(text)
            finally:
                self._agent.trace = old_trace

    async def handle_user_input(
        self,
        text: str,
        *,
        on_progress: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ) -> str:
        """Run one user turn; return the assistant answer string.

        When tools ran, prefix a short dim trail so the CLI shows whether a
        `` ```run `` block actually executed (vs the model printing argv as prose).

        ``on_progress(kind, attrs)`` is invoked synchronously from the ask()
        worker thread when a temporary Tracer sink is installed — keep it fast
        and thread-safe (schedule Matrix I/O onto the event loop).
        """
        # Offload: ask() blocks on LLM + CliTool subprocesses.
        result = await asyncio.to_thread(self._ask_sync, text, on_progress)
        if not isinstance(result, dict):
            return str(result or "")

        metrics = result.get("metrics")
        if isinstance(metrics, dict):
            self.last_metrics = dict(metrics)

        answer = str(result.get("answer") or "")
        tool_lines: list[str] = []
        for step in result.get("steps") or []:
            if not isinstance(step, dict) or step.get("kind") != "action":
                continue
            for tc in step.get("tool_calls") or []:
                name = tc.get("action") or "?"
                obs = str(tc.get("observation") or "").replace("\n", " ")
                if len(obs) > 120:
                    obs = obs[:117] + "..."
                tool_lines.append(f"[tool] {name} → {obs}")
        if tool_lines:
            return "\n".join(tool_lines) + "\n\n" + answer
        return answer

    async def reset(self) -> None:
        """Wipe durable memory for this conversation id (SQLite + vectors)."""
        self.last_metrics = {}
        await asyncio.to_thread(self._agent.clear)


async def run_sallm_chat(
    cfg: AgentConfig,
    emit: Callable[[str], None] = print,
    on_thinking: Optional[Callable[[], None]] = None,
    *,
    conversation_id: str = "cli",
) -> None:
    """Interactive REPL for ``pawn-agent chat`` (sallm-backed)."""
    from prompt_toolkit import PromptSession  # noqa: PLC0415

    session = SallmChatSession.create(cfg, conversation_id=conversation_id)
    prompt_session: Any = PromptSession()
    while True:
        try:
            raw = await prompt_session.prompt_async("You: ")
        except (EOFError, KeyboardInterrupt):
            return
        text = raw.strip()
        if not text:
            continue
        if text.lower() in {"/exit", "/quit"}:
            return
        if text.lower() == "/reset":
            await session.reset()
            emit("Session cleared.")
            continue
        if text.lower() == "/stats":
            emit(await asyncio.to_thread(session.format_stats))
            continue
        if text.startswith("/"):
            emit("Supported slash commands: /stats, /reset, /exit, /quit.")
            continue
        if on_thinking is not None:
            on_thinking()
        try:
            reply = await session.handle_user_input(text)
        except Exception as exc:
            logger.exception("sallm chat turn failed")
            emit(f"Error: {exc}")
            continue
        emit(reply)

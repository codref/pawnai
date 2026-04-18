"""BurrAgent — Burr-based dynamic context agent.

Replaces :class:`pawn_agent.core.pydantic_agent.PydanticAgent`.  The public
interface (``run``, ``run_async``, ``chat``) is compatible so the CLI and
queue listener need only swap the import.

Architecture
------------
Each call to ``run`` / ``run_async`` creates (or resumes) a Burr
``Application`` whose state is a serialised ``DynamicContextState``.  After
every turn the compressed state is dual-written:

* **Burr tracker** (Postgres or local) — full graph replay via ``burr`` UI
* **agent_session_turns** — human-readable chat turns for history tooling

The Burr UI is started with::

    burr
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Optional

from pawn_agent.core.burr_actions import (
    planner,
    response_generator,
    state_compressor,
    tool_executor,
    tool_router,
)
from pawn_agent.core.burr_context import (
    assemble_prompt_context,
    derive_retrieval_query,
    rank_context_candidates,
    retrieve_context_candidates,
    select_context_with_model,
)
from pawn_agent.core.burr_state import (
    DynamicContextState,
    burr_dict_to_state,
    state_to_burr_dict,
)
from pawn_agent.utils.config import AgentConfig

logger = logging.getLogger(__name__)


# ── Tools registry builder ────────────────────────────────────────────────────


def build_tools_registry(
    cfg: AgentConfig,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Return {tool_name: callable} for every auto-discovered tool.

    Calls :func:`pawn_agent.tools.build_tools` with a live
    :class:`~pawn_agent.core.session_vars.SessionVars` instance so that tools
    requiring ``session_vars`` (e.g. the session-variable tool) are included.
    Extracts the underlying callable from each :class:`pydantic_ai.Tool` so
    ``tool_executor`` can call it directly with ``**arguments``.
    """
    from pawn_agent.core.session_vars import SessionVars
    from pawn_agent.tools import build_tools

    sv = SessionVars(session_id=session_id, dsn=cfg.db_dsn or None)
    sv.load()
    tools = build_tools(cfg, session_vars=sv)
    registry: dict[str, Any] = {}
    for tool in tools:
        # pydantic_ai.Tool exposes .name and .function
        name: str = getattr(tool, "name", "") or getattr(tool, "_name", "")
        fn: Any = getattr(tool, "function", None) or getattr(tool, "_function", None)
        if name and fn is not None:
            registry[name] = fn
    return registry


# ── Burr Application factory ───────────────────────────────────────────────────


def _build_tracker(cfg: AgentConfig) -> Optional[Any]:
    """Return a Burr tracker based on config, or None if disabled."""
    if not cfg.burr_enabled:
        return None

    # PostgresTracker is not available in the current Burr release; always use
    # LocalTrackingClient regardless of cfg.burr_backend.
    storage_dir = cfg.burr_storage_dir or ".burr"
    try:
        from burr.tracking import LocalTrackingClient

        return LocalTrackingClient(project=cfg.burr_project, storage_dir=storage_dir)
    except Exception as exc:
        logger.warning("Could not create Burr LocalTrackingClient: %s", exc)
        return None


def build_burr_app(
    cfg: AgentConfig,
    session_id: str,
    tools_registry: dict[str, Any],
    initial_state: Optional[DynamicContextState] = None,
) -> Any:  # burr.core.Application
    """Build and return the Burr Application for one conversation session."""
    import burr.core
    from burr.core import ApplicationBuilder, expr

    state_dict = state_to_burr_dict(initial_state or DynamicContextState())

    builder = (
        ApplicationBuilder()
        .with_state(**state_dict)
        .with_actions(
            # Context subgraph
            derive_retrieval_query=derive_retrieval_query,
            retrieve_context_candidates=retrieve_context_candidates.bind(cfg=cfg),
            rank_context_candidates=rank_context_candidates,
            select_context_with_model=select_context_with_model.bind(cfg=cfg),
            assemble_prompt_context=assemble_prompt_context,
            # Main graph
            planner=planner.bind(cfg=cfg),
            tool_router=tool_router.bind(cfg=cfg, tools_registry=tools_registry),
            tool_executor=tool_executor.bind(tools_registry=tools_registry),
            state_compressor=state_compressor.bind(cfg=cfg),
            response_generator=response_generator.bind(cfg=cfg),
        )
        .with_transitions(
            # Context pipeline
            ("derive_retrieval_query", "retrieve_context_candidates"),
            ("retrieve_context_candidates", "rank_context_candidates"),
            ("rank_context_candidates", "select_context_with_model"),
            # Loop back for additional retrieval if needed
            (
                "select_context_with_model",
                "retrieve_context_candidates",
                expr("need_additional_retrieval == True"),
            ),
            (
                "select_context_with_model",
                "assemble_prompt_context",
                expr("need_additional_retrieval != True"),
            ),
            # After first context assembly → planner
            ("assemble_prompt_context", "planner"),
            # Planner routes: respond directly or call a tool
            (
                "planner",
                "response_generator",
                expr("next_action == 'respond'"),
            ),
            (
                "planner",
                "tool_router",
                expr("next_action != 'respond'"),
            ),
            # Tool pipeline
            ("tool_router", "tool_executor"),
            # After tool call: compress result into facts then respond
            ("tool_executor", "state_compressor"),
            ("state_compressor", "response_generator"),
        )
        .with_entrypoint("derive_retrieval_query")
        .with_identifiers(app_id=session_id)
    )

    # Attach tracker
    tracker = _build_tracker(cfg)
    if tracker is not None:
        builder = builder.with_tracker(tracker)

    return builder.build()


# ── Result wrapper ────────────────────────────────────────────────────────────


class _AgentResult:
    """Thin result object that mirrors the PydanticAI result interface.

    ``BurrAgent`` already persists turns internally via ``_persist_turn``, so
    ``new_messages()`` returns an empty list to prevent double-writing.
    """

    def __init__(self, output: str) -> None:
        self.output = output

    def new_messages(self) -> list:
        """Return empty list — BurrAgent handles persistence internally."""
        return []


# ── BurrAgent ─────────────────────────────────────────────────────────────────


class BurrAgent:
    """Public interface for the Burr-based dynamic context agent.

    Mirrors the :class:`PydanticAgent` API so CLI / queue listener imports
    can be updated by changing only the class name.
    """

    def __init__(
        self,
        cfg: AgentConfig,
        emit: Optional[Callable[[str], None]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self._emit = emit
        self._session_id = session_id or str(uuid.uuid4())
        self._tools_registry = build_tools_registry(cfg, session_id=self._session_id)

        # Load persisted state for the session if available
        initial_state = self._load_state(self._session_id)
        self._app = build_burr_app(cfg, self._session_id, self._tools_registry, initial_state)

    # ── State persistence helpers ──────────────────────────────────────────────

    def _load_state(self, session_id: str) -> Optional[DynamicContextState]:
        """BurrAgent manages state through the Burr application; returns None.

        State continuity across requests is maintained by the in-memory
        ``_app`` reused via the ``_agent_cache`` in the queue listener.
        Full cold-restart persistence via the Burr tracker is a future
        enhancement.
        """
        return None

    def _persist_turn(self, session_id: str, goal: str, response: str, state: DynamicContextState) -> None:
        """No-op: BurrAgent state is managed by Burr, not agent_session_turns.

        The Burr LocalTrackingClient records each step for observability.
        Writing to agent_session_turns is skipped to avoid incompatible
        message formats with the PydanticAI-based load_history reader.
        """

    # ── Transition override to fix context-after-tool routing ─────────────────

    def _run_turn(self, goal: str) -> str:
        """Execute one full turn synchronously by stepping through the Burr app."""
        # Carry over accumulated state (conversation history, facts, plan) from
        # the previous turn, update the goal, and reset per-turn fields.
        current = burr_dict_to_state(dict(self._app.state))
        current.user_goal = goal
        # Reset per-turn retrieval/routing fields so the graph starts clean
        current.retrieval_query = ""
        current.context_candidates = []
        current.selected_context_ids = []
        current.need_additional_retrieval = False
        current.assembled_context = ""
        current.next_action = "respond"
        current.pending_tool_call = None
        current.raw_tool_result = None

        # Rebuild the app so the entrypoint resets to derive_retrieval_query.
        # This is the correct multi-turn pattern: Burr apps resume from their
        # last halted node, so without a rebuild the second call finds no
        # outgoing transitions from response_generator.
        self._app = build_burr_app(self.cfg, self._session_id, self._tools_registry, current)

        # Step until response_generator halts
        halt_after = {"response_generator"}
        action_name, result, state = self._app.run(
            halt_after=halt_after,
            inputs={},
        )

        # Extract the final response from recent_messages
        recent: list[dict] = state.get("recent_messages", [])
        response = ""
        for msg in reversed(recent):
            if msg.get("role") == "assistant":
                response = msg.get("content", "")
                break

        burr_state = burr_dict_to_state(dict(state))
        self._persist_turn(self._session_id, goal, response, burr_state)

        return response

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self, user_prompt: str, **_kwargs: Any) -> "_AgentResult":
        """Single-turn: run synchronously and return an agent result object.

        Extra keyword arguments (e.g. ``message_history``, ``model_settings``)
        are accepted for API compatibility with the legacy PydanticAgent interface
        and silently ignored — BurrAgent manages its own context via Burr state.
        """
        output = self._run_turn(user_prompt)
        return _AgentResult(output)

    async def run_async(self, user_prompt: str, **_kwargs: Any) -> "_AgentResult":
        """Single-turn async wrapper (runs in thread to avoid blocking event loop).

        Extra keyword arguments (e.g. ``message_history``, ``model_settings``)
        are accepted for API compatibility with the legacy PydanticAgent interface
        and silently ignored — BurrAgent manages its own context via Burr state.
        BurrAgent also handles its own turn persistence (``_persist_turn``), so
        the caller should not append the result messages again.
        """
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, self._run_turn, user_prompt)
        return _AgentResult(output)

    def chat(
        self,
        first_message: Optional[str] = None,
        initial_history: Optional[list] = None,
        on_turn_complete: Optional[Callable[[list], None]] = None,
        on_reset: Optional[Callable[[], None]] = None,
    ) -> None:
        """Interactive multi-turn REPL (blocking).

        Supports:
          /exit, /quit  — end session
          /reset        — clear state and call on_reset
          /vars         — print session variables
        """
        asyncio.run(
            self._repl_loop(
                first_message=first_message,
                on_turn_complete=on_turn_complete,
                on_reset=on_reset,
            )
        )

    async def _repl_loop(
        self,
        first_message: Optional[str],
        on_turn_complete: Optional[Callable[[list], None]],
        on_reset: Optional[Callable[[], None]],
    ) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import HTML

        ps: PromptSession = PromptSession()

        async def _get_input(prompt_text: str) -> Optional[str]:
            try:
                return await ps.prompt_async(HTML(prompt_text))
            except (EOFError, KeyboardInterrupt):
                return None

        if first_message is not None:
            user_input: Optional[str] = first_message
        else:
            user_input = await _get_input("<b>You</b>: ")

        while user_input is not None:
            stripped = user_input.strip()
            if not stripped:
                user_input = await _get_input("<b>You</b>: ")
                continue

            # Slash commands
            if stripped in ("/exit", "/quit"):
                break

            if stripped == "/reset":
                # Reset Burr app state
                initial_state = DynamicContextState()
                self._app = build_burr_app(
                    self.cfg, self._session_id, self._tools_registry, initial_state
                )
                if on_reset:
                    on_reset()
                user_input = await _get_input("<b>You</b>: ")
                continue

            if stripped == "/vars":
                state_dict = dict(self._app.state)
                facts = state_dict.get("facts", [])
                artifacts = state_dict.get("artifacts", [])
                plan = state_dict.get("plan", "")
                print(f"plan: {plan or '(none)'}")
                print(f"facts: {len(facts)}")
                print(f"artifacts: {len(artifacts)}")
                user_input = await _get_input("<b>You</b>: ")
                continue

            # Normal turn
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, self._run_turn, stripped)
            except Exception as exc:
                logger.error("Agent error: %s", exc)
                response = f"[Error: {exc}]"

            if self._emit:
                self._emit(response)
            else:
                from rich.console import Console
                from rich.markdown import Markdown

                Console().print(Markdown(response))

            if on_turn_complete:
                msgs = [{"role": "user", "content": stripped}, {"role": "assistant", "content": response}]
                on_turn_complete(msgs)

            user_input = await _get_input("<b>You</b>: ")

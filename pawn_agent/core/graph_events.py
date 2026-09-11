"""Lightweight graph execution event capture for LangGraph runs."""

from __future__ import annotations

import inspect
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable

from pawn_agent.core.langgraph_state import get_state_field
from pawn_agent.utils.db import save_graph_run_event, upsert_graph_topology

logger = logging.getLogger(__name__)

LANGGRAPH_CHAT_GRAPH_NAME = "langgraph_chat"
LANGGRAPH_CHAT_GRAPH_VERSION = "v1"

LANGGRAPH_CHAT_TOPOLOGY: dict[str, Any] = {
    "nodes": [
        {"id": "__start__", "label": "START", "kind": "system"},
        {"id": "human_input", "label": "human_input", "kind": "node"},
        {"id": "recall_memories", "label": "recall_memories", "kind": "node"},
        {"id": "plan", "label": "plan", "kind": "node"},
        {"id": "dispatch", "label": "dispatch", "kind": "router"},
        {"id": "extract_session_id", "label": "extract_session_id", "kind": "router"},
        {"id": "tool_list_sessions", "label": "tool_list_sessions", "kind": "tool"},
        {"id": "tool_analyze_summary", "label": "tool_analyze_summary", "kind": "tool"},
        {"id": "tool_query_conversation", "label": "tool_query_conversation", "kind": "tool"},
        {"id": "tool_save_to_siyuan", "label": "tool_save_to_siyuan", "kind": "tool"},
        {"id": "tool_memorize", "label": "tool_memorize", "kind": "tool"},
        {"id": "tool_recall_memory", "label": "tool_recall_memory", "kind": "tool"},
        {"id": "tool_search_knowledge", "label": "tool_search_knowledge", "kind": "tool"},
        {"id": "tool_vectorize", "label": "tool_vectorize", "kind": "tool"},
        {"id": "tool_push_queue_message", "label": "tool_push_queue_message", "kind": "tool"},
        {
            "id": "tool_propose_schedule_change",
            "label": "tool_propose_schedule_change",
            "kind": "tool",
        },
        {"id": "respond_fast", "label": "respond_fast", "kind": "response"},
        {"id": "respond_deep", "label": "respond_deep", "kind": "response"},
        {"id": "__end__", "label": "END", "kind": "system"},
    ],
    "edges": [
        {"id": "__start__-human_input", "source": "__start__", "target": "human_input"},
        {"id": "human_input-recall_memories", "source": "human_input", "target": "recall_memories"},
        {"id": "recall_memories-plan", "source": "recall_memories", "target": "plan"},
        {"id": "plan-dispatch", "source": "plan", "target": "dispatch"},
        {"id": "tool_list_sessions-dispatch", "source": "tool_list_sessions", "target": "dispatch"},
        {
            "id": "tool_analyze_summary-dispatch",
            "source": "tool_analyze_summary",
            "target": "dispatch",
        },
        {
            "id": "tool_query_conversation-dispatch",
            "source": "tool_query_conversation",
            "target": "dispatch",
        },
        {
            "id": "tool_save_to_siyuan-dispatch",
            "source": "tool_save_to_siyuan",
            "target": "dispatch",
        },
        {"id": "tool_memorize-dispatch", "source": "tool_memorize", "target": "dispatch"},
        {"id": "tool_recall_memory-dispatch", "source": "tool_recall_memory", "target": "dispatch"},
        {
            "id": "tool_search_knowledge-dispatch",
            "source": "tool_search_knowledge",
            "target": "dispatch",
        },
        {"id": "tool_vectorize-dispatch", "source": "tool_vectorize", "target": "dispatch"},
        {
            "id": "tool_push_queue_message-dispatch",
            "source": "tool_push_queue_message",
            "target": "dispatch",
        },
        {
            "id": "tool_propose_schedule_change-dispatch",
            "source": "tool_propose_schedule_change",
            "target": "dispatch",
        },
        {"id": "respond_fast-dispatch", "source": "respond_fast", "target": "dispatch"},
        {"id": "respond_deep-dispatch", "source": "respond_deep", "target": "dispatch"},
        {
            "id": "extract_session_id-tool_analyze_summary",
            "source": "extract_session_id",
            "target": "tool_analyze_summary",
            "conditional": True,
        },
        {
            "id": "extract_session_id-tool_query_conversation",
            "source": "extract_session_id",
            "target": "tool_query_conversation",
            "conditional": True,
        },
        {
            "id": "dispatch-tool_list_sessions",
            "source": "dispatch",
            "target": "tool_list_sessions",
            "conditional": True,
        },
        {
            "id": "dispatch-extract_session_id",
            "source": "dispatch",
            "target": "extract_session_id",
            "conditional": True,
        },
        {
            "id": "dispatch-tool_save_to_siyuan",
            "source": "dispatch",
            "target": "tool_save_to_siyuan",
            "conditional": True,
        },
        {
            "id": "dispatch-tool_memorize",
            "source": "dispatch",
            "target": "tool_memorize",
            "conditional": True,
        },
        {
            "id": "dispatch-tool_recall_memory",
            "source": "dispatch",
            "target": "tool_recall_memory",
            "conditional": True,
        },
        {
            "id": "dispatch-tool_search_knowledge",
            "source": "dispatch",
            "target": "tool_search_knowledge",
            "conditional": True,
        },
        {
            "id": "dispatch-tool_vectorize",
            "source": "dispatch",
            "target": "tool_vectorize",
            "conditional": True,
        },
        {
            "id": "dispatch-tool_push_queue_message",
            "source": "dispatch",
            "target": "tool_push_queue_message",
            "conditional": True,
        },
        {
            "id": "dispatch-tool_propose_schedule_change",
            "source": "dispatch",
            "target": "tool_propose_schedule_change",
            "conditional": True,
        },
        {
            "id": "dispatch-respond_fast",
            "source": "dispatch",
            "target": "respond_fast",
            "conditional": True,
        },
        {
            "id": "dispatch-respond_deep",
            "source": "dispatch",
            "target": "respond_deep",
            "conditional": True,
        },
        {"id": "dispatch-__end__", "source": "dispatch", "target": "__end__", "conditional": True},
    ],
}


class GraphEventRecorder:
    """Best-effort append-only recorder for one agent graph run."""

    def __init__(
        self,
        dsn: str,
        *,
        run_id: str,
        thread_id: str | None,
        graph_name: str = LANGGRAPH_CHAT_GRAPH_NAME,
        graph_version: str = LANGGRAPH_CHAT_GRAPH_VERSION,
        trace_id: str | None = None,
    ) -> None:
        self.dsn = dsn
        self.run_id = run_id
        self.thread_id = thread_id
        self.trace_id = trace_id
        self.graph_name = graph_name
        self.graph_version = graph_version
        self._sequence = 0
        self._enabled = os.getenv("PAWN_GRAPH_EVENTS_ENABLED", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        if self._enabled:
            self._save_topology()

    def _save_topology(self) -> None:
        try:
            upsert_graph_topology(
                self.dsn,
                graph_name=self.graph_name,
                graph_version=self.graph_version,
                topology=LANGGRAPH_CHAT_TOPOLOGY,
            )
        except Exception:
            logger.warning("Failed to persist graph topology", exc_info=True)

    def record(
        self,
        event_type: str,
        *,
        node_name: str | None = None,
        from_node: str | None = None,
        to_node: str | None = None,
        router_choice: str | None = None,
        duration_ms: int | None = None,
        status: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append one event. Failures are logged and swallowed."""
        if not self._enabled:
            return
        self._sequence += 1
        try:
            save_graph_run_event(
                self.dsn,
                run_id=self.run_id,
                sequence=self._sequence,
                thread_id=self.thread_id,
                trace_id=self.trace_id,
                graph_name=self.graph_name,
                graph_version=self.graph_version,
                event_type=event_type,
                node_name=node_name,
                from_node=from_node,
                to_node=to_node,
                router_choice=router_choice,
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                status=status,
                payload=payload,
            )
        except Exception:
            logger.warning("Graph event capture failed", exc_info=True)


def _resolve_recorder(
    graph_recorder: GraphEventRecorder | Callable[[], GraphEventRecorder | None] | None,
) -> GraphEventRecorder | None:
    if callable(graph_recorder):
        return graph_recorder()
    return graph_recorder


def instrument_node(
    name: str,
    fn: Callable[..., Any],
    graph_recorder: GraphEventRecorder | Callable[[], GraphEventRecorder | None] | None,
) -> Callable[..., Any]:
    """Wrap a LangGraph node with node start/end/error event capture."""

    async def async_wrapped(state: Any) -> Any:
        recorder = _resolve_recorder(graph_recorder)
        if recorder is None:
            return await fn(state)
        recorder.record("node_start", node_name=name, status="running")
        start = time.perf_counter()
        try:
            result = await fn(state)
        except Exception as exc:
            recorder.record(
                "error",
                node_name=name,
                duration_ms=int((time.perf_counter() - start) * 1000),
                status="failed",
                payload={"error": str(exc)},
            )
            raise
        recorder.record(
            "node_end",
            node_name=name,
            duration_ms=int((time.perf_counter() - start) * 1000),
            status="completed",
        )
        return result

    def sync_wrapped(state: Any) -> Any:
        recorder = _resolve_recorder(graph_recorder)
        if recorder is None:
            return fn(state)
        recorder.record("node_start", node_name=name, status="running")
        start = time.perf_counter()
        try:
            result = fn(state)
        except Exception as exc:
            recorder.record(
                "error",
                node_name=name,
                duration_ms=int((time.perf_counter() - start) * 1000),
                status="failed",
                payload={"error": str(exc)},
            )
            raise
        recorder.record(
            "node_end",
            node_name=name,
            duration_ms=int((time.perf_counter() - start) * 1000),
            status="completed",
        )
        return result

    return async_wrapped if inspect.iscoroutinefunction(fn) else sync_wrapped


def instrument_router(
    from_node: str,
    router_fn: Callable[..., str],
    graph_recorder: GraphEventRecorder | Callable[[], GraphEventRecorder | None] | None,
) -> Callable[..., str]:
    """Wrap a conditional router to capture the decision and selected edge."""

    def wrapped(state: Any) -> str:
        choice = router_fn(state)
        recorder = _resolve_recorder(graph_recorder)
        if recorder is not None:
            to_node = "__end__" if choice == "__end__" else choice
            recorder.record(
                "router_decision",
                node_name=from_node,
                from_node=from_node,
                to_node=None if choice == "__end__" else choice,
                router_choice=choice,
                status="completed",
                payload={
                    "route_kind": get_state_field(state, "route_kind"),
                    "requested_session_id": get_state_field(state, "requested_session_id"),
                },
            )
            recorder.record(
                "edge_taken",
                from_node=from_node,
                to_node=to_node,
                router_choice=choice,
                status="completed",
            )
        return choice

    return wrapped

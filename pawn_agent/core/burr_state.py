"""Pydantic state models and Burr dict conversion helpers for the Burr agent.

``DynamicContextState`` is the single source of truth flowing through the Burr
application.  All Burr actions receive and return slices of this state.

Conversion helpers
------------------
Burr's ``State`` object is dict-like; Pydantic models cannot be stored directly
unless serialised.  ``state_to_burr_dict`` / ``burr_dict_to_state`` perform the
round-trip as JSON-serialisable dicts so Burr's Postgres / local backends can
persist the full state without custom hooks.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Leaf models ───────────────────────────────────────────────────────────────


class Fact(BaseModel):
    """A single compressed fact extracted from a tool result."""

    text: str
    source: str = ""  # tool name or session_id
    confidence: float = 1.0
    entity_ids: list[str] = Field(default_factory=list)


class ArtifactSummary(BaseModel):
    """Compact summary of a tool invocation's artefacts."""

    tool_name: str
    key_findings: str
    token_estimate: int = 0


class OpenQuestion(BaseModel):
    """A question the planner needs answered to make progress."""

    question: str
    priority: int = 0  # higher = more urgent


class ToolCall(BaseModel):
    """Selected tool and typed arguments produced by ``tool_router``."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


# ── Retrieval pipeline schemas ─────────────────────────────────────────────────


class ContextCandidate(BaseModel):
    """A single retrieval hit to be ranked and possibly selected."""

    id: str
    text: str
    score: float = 0.0
    source_type: str = ""  # "chunk", "fact", "artifact"
    token_estimate: int = 0
    entities: list[str] = Field(default_factory=list)
    freshness_score: float = 0.0


class ContextSelection(BaseModel):
    """Result of the model-based context selector."""

    selected_ids: list[str] = Field(default_factory=list)
    need_additional_retrieval: bool = False
    additional_query: Optional[str] = None


# ── Main state ────────────────────────────────────────────────────────────────


class DynamicContextState(BaseModel):
    """Full Burr agent state for one conversation turn.

    Fields are grouped by lifecycle:
    - *persistent* fields survive across turns (facts, artifacts, plan)
    - *per-turn* fields are reset each turn (raw_tool_result, assembled_context, …)
    """

    # -- Persistent across turns ---
    user_goal: str = ""
    constraints: list[str] = Field(default_factory=list)
    plan: str = ""
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    recent_messages: list[dict[str, str]] = Field(default_factory=list)  # {role, content}

    # -- Context retrieval (per-turn) ---
    retrieval_query: str = ""
    context_candidates: list[ContextCandidate] = Field(default_factory=list)
    selected_context_ids: list[str] = Field(default_factory=list)
    need_additional_retrieval: bool = False
    token_budget_for_context: int = 4096
    assembled_context: str = ""

    # -- Action routing (per-turn) ---
    next_action: str = "respond"  # "respond" | tool name
    pending_tool_call: Optional[ToolCall] = None
    raw_tool_result: Optional[str] = None


# ── Conversion helpers ─────────────────────────────────────────────────────────


def state_to_burr_dict(state: DynamicContextState) -> dict[str, Any]:
    """Serialise ``DynamicContextState`` to a plain dict suitable for Burr ``State``."""
    return state.model_dump(mode="json")


def burr_dict_to_state(d: dict[str, Any]) -> DynamicContextState:
    """Deserialise a plain dict from Burr ``State`` back to ``DynamicContextState``."""
    return DynamicContextState.model_validate(d)

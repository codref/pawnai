"""Dynamic context retrieval subgraph for the Burr agent.

Five pure Burr ``@action`` functions implement the context pipeline that runs
before both ``planner`` and ``tool_executor``.  Each action reads from the
current ``burr.State`` dict (keys match ``DynamicContextState`` fields) and
returns a dict of updated keys.

Pipeline
--------
derive_retrieval_query
  → retrieve_context_candidates
    → rank_context_candidates
      → select_context_with_model
          (need_additional_retrieval=True)  → retrieve_context_candidates
          (need_additional_retrieval=False) → assemble_prompt_context
"""

from __future__ import annotations

import json
import logging
from typing import Any

import burr.core
from burr.core import State, action

from pawn_agent.core.burr_state import (
    ContextCandidate,
    ContextSelection,
    DynamicContextState,
    burr_dict_to_state,
)
from pawn_agent.utils.config import AgentConfig

logger = logging.getLogger(__name__)

_MAX_CANDIDATES = 30
_TOKEN_BUDGET_DEFAULT = 4096
_CHARS_PER_TOKEN = 4  # rough approximation


# ── 1. derive_retrieval_query ──────────────────────────────────────────────────


@action(reads=["user_goal", "open_questions", "constraints"], writes=["retrieval_query"])
def derive_retrieval_query(state: State) -> tuple[dict, State]:
    """Build a retrieval query from the current goal and open questions."""
    goal: str = state["user_goal"]
    open_qs: list[dict] = state.get("open_questions", [])
    constraints: list[str] = state.get("constraints", [])

    # Top-3 open questions by priority (highest first)
    sorted_qs = sorted(open_qs, key=lambda q: q.get("priority", 0), reverse=True)[:3]
    q_texts = [q["question"] for q in sorted_qs]

    parts = [goal]
    if q_texts:
        parts.append("Questions: " + "; ".join(q_texts))
    if constraints:
        parts.append("Constraints: " + "; ".join(constraints))

    query = " | ".join(parts)
    result = {"retrieval_query": query}
    return result, state.update(**result)


# ── 2. retrieve_context_candidates ────────────────────────────────────────────


@action(
    reads=["retrieval_query", "facts", "artifacts", "token_budget_for_context"],
    writes=["context_candidates"],
)
def retrieve_context_candidates(state: State, cfg: AgentConfig) -> tuple[dict, State]:
    """Search pgvector text_chunks + in-state facts/artifacts; return top-30 candidates."""
    from sqlalchemy import text as sa_text

    from pawn_agent.utils.db import make_db_session
    from pawn_agent.utils.vectorize import load_embedding_model

    query: str = state["retrieval_query"]
    facts: list[dict] = state.get("facts", [])
    artifacts: list[dict] = state.get("artifacts", [])

    candidates: list[ContextCandidate] = []

    # --- pgvector search ---
    try:
        embed_model = load_embedding_model(
            cfg.embed_model,
            cfg.embed_device,
            truncate_dim=cfg.embed_dim or None,
            local_files_only=cfg.embed_local_files_only,
        )
        query_vec = embed_model.encode(query, show_progress_bar=False).tolist()
        query_vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

        sql = sa_text(
            """
            SELECT
                tc.id::text,
                tc.text,
                1 - (tc.embedding <=> CAST(:query_vec AS vector)) AS score,
                rs.source_type,
                tc.metadata
            FROM text_chunks tc
            JOIN rag_sources rs ON rs.id = tc.source_id
            ORDER BY tc.embedding <=> CAST(:query_vec AS vector)
            LIMIT :limit
            """
        )
        with make_db_session(cfg.db_dsn) as session:
            rows = session.execute(sql, {"query_vec": query_vec_str, "limit": _MAX_CANDIDATES}).fetchall()

        for row in rows:
            token_est = len(row.text) // _CHARS_PER_TOKEN
            candidates.append(
                ContextCandidate(
                    id=row.id,
                    text=row.text,
                    score=float(row.score),
                    source_type=row.source_type or "chunk",
                    token_estimate=token_est,
                    entities=(row.metadata or {}).get("tags", []) if row.metadata else [],
                )
            )
    except Exception as exc:
        logger.warning("pgvector search failed: %s", exc)

    # --- in-state facts ---
    for i, fact in enumerate(facts):
        text = fact.get("text", "")
        candidates.append(
            ContextCandidate(
                id=f"fact:{i}",
                text=text,
                score=0.9,  # facts are highly trusted
                source_type="fact",
                token_estimate=len(text) // _CHARS_PER_TOKEN,
                entities=fact.get("entity_ids", []),
                freshness_score=1.0,
            )
        )

    # --- in-state artifact summaries ---
    for i, art in enumerate(artifacts):
        text = art.get("key_findings", "")
        candidates.append(
            ContextCandidate(
                id=f"artifact:{i}",
                text=f"[{art.get('tool_name', 'tool')}] {text}",
                score=0.85,
                source_type="artifact",
                token_estimate=art.get("token_estimate", len(text) // _CHARS_PER_TOKEN),
                freshness_score=1.0,
            )
        )

    result = {"context_candidates": [c.model_dump() for c in candidates]}
    return result, state.update(**result)


# ── 3. rank_context_candidates ────────────────────────────────────────────────


@action(reads=["context_candidates", "retrieval_query", "facts"], writes=["context_candidates"])
def rank_context_candidates(state: State) -> tuple[dict, State]:
    """Deterministic re-rank: entity match, freshness, confidence, token cost."""
    candidates: list[dict] = state.get("context_candidates", [])
    query: str = state.get("retrieval_query", "").lower()
    query_tokens = set(query.split())

    def _rank_key(c: dict) -> float:
        score = float(c.get("score", 0.0))

        # Boost for entity/keyword overlap
        entities = [e.lower() for e in c.get("entities", [])]
        text_tokens = set(c.get("text", "").lower().split())
        overlap = len(query_tokens & (set(entities) | text_tokens))
        entity_boost = min(overlap * 0.05, 0.2)

        # Freshness boost (in-state sources already have freshness_score=1.0)
        freshness_boost = float(c.get("freshness_score", 0.0)) * 0.1

        # Token cost penalty (prefer concise chunks)
        token_est = max(c.get("token_estimate", 1), 1)
        cost_penalty = min(token_est / 2000, 0.15)

        return score + entity_boost + freshness_boost - cost_penalty

    ranked = sorted(candidates, key=_rank_key, reverse=True)
    result = {"context_candidates": ranked}
    return result, state.update(**result)


# ── 4. select_context_with_model ──────────────────────────────────────────────

_SELECTOR_SYSTEM = (
    "You are a context selection assistant. Given a list of candidate context "
    "chunks and a user goal, select the minimal set of chunk IDs that cover the "
    "information needed to answer the goal within the token budget. Respond ONLY "
    "with valid JSON matching the schema provided."
)


@action(
    reads=["context_candidates", "user_goal", "token_budget_for_context"],
    writes=["selected_context_ids", "context_candidates", "need_additional_retrieval"],
)
def select_context_with_model(state: State, cfg: AgentConfig) -> tuple[dict, State]:
    """PydanticAI sub-call to select which candidates to include in the prompt."""
    import asyncio

    from pydantic_ai import Agent

    candidates: list[dict] = state.get("context_candidates", [])
    goal: str = state.get("user_goal", "")
    budget: int = state.get("token_budget_for_context", _TOKEN_BUDGET_DEFAULT)

    # Build a compact representation for the model
    candidate_summary = []
    for c in candidates[:20]:  # cap at 20 to stay within prompt budget
        candidate_summary.append(
            {"id": c["id"], "tokens": c.get("token_estimate", 0), "text": c["text"][:200]}
        )

    prompt = (
        f"Goal: {goal}\n\n"
        f"Token budget: {budget}\n\n"
        f"Candidates (JSON):\n{json.dumps(candidate_summary, indent=2)}\n\n"
        "Respond with JSON: "
        '{"selected_ids": [...], "need_additional_retrieval": false, "additional_query": null}'
    )

    async def _call() -> ContextSelection:
        model_str = cfg.selector_model
        logger.info("[context_selector] → %s", model_str)
        api_key = cfg.pydantic_api_key
        base_url = cfg.pydantic_base_url
        import os

        if base_url and model_str.startswith("openai:"):
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            model_name = model_str[len("openai:"):]
            provider = OpenAIProvider(base_url=base_url, api_key=api_key or "no-key")
            model = OpenAIChatModel(model_name, provider=provider)
        else:
            if api_key:
                if model_str.startswith("openai:"):
                    os.environ.setdefault("OPENAI_API_KEY", api_key)
                elif model_str.startswith("anthropic:"):
                    os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            model = model_str

        agent: Agent[None, ContextSelection] = Agent(
            model,
            output_type=ContextSelection,
            system_prompt=_SELECTOR_SYSTEM,
            retries=0,
        )
        result = await agent.run(prompt)
        logger.info("[context_selector] ← done")
        return result.output

    try:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _call())
                selection = future.result(timeout=60)
        except RuntimeError:
            selection = asyncio.run(_call())
    except Exception as exc:
        logger.warning("Context selector model call failed (%s); using top-10 by rank", exc)
        selection = ContextSelection(selected_ids=[c["id"] for c in candidates[:10]])

    result = {
        "selected_context_ids": selection.selected_ids,
        # Persist need_additional_retrieval so Burr transition conditions can read it
        "need_additional_retrieval": selection.need_additional_retrieval,
        # Preserve candidates so assemble_prompt_context can look them up
        "context_candidates": candidates,
    }
    return result, state.update(**result)


# ── 5. assemble_prompt_context ────────────────────────────────────────────────


@action(
    reads=[
        "user_goal",
        "constraints",
        "selected_context_ids",
        "context_candidates",
        "facts",
        "artifacts",
        "open_questions",
        "token_budget_for_context",
    ],
    writes=["assembled_context"],
)
def assemble_prompt_context(state: State) -> tuple[dict, State]:
    """Build the minimal prompt string enforcing the token budget.

    Priority: facts (in-state) > artifact summaries > selected chunk text.
    """
    goal: str = state.get("user_goal", "")
    constraints: list[str] = state.get("constraints", [])
    facts: list[dict] = state.get("facts", [])
    artifacts: list[dict] = state.get("artifacts", [])
    open_qs: list[dict] = state.get("open_questions", [])
    selected_ids: set[str] = set(state.get("selected_context_ids", []))
    candidates: list[dict] = state.get("context_candidates", [])
    budget: int = state.get("token_budget_for_context", _TOKEN_BUDGET_DEFAULT)

    parts: list[str] = [f"## Goal\n{goal}"]

    if constraints:
        parts.append("## Constraints\n" + "\n".join(f"- {c}" for c in constraints))

    if open_qs:
        sorted_qs = sorted(open_qs, key=lambda q: q.get("priority", 0), reverse=True)
        parts.append("## Open Questions\n" + "\n".join(f"- {q['question']}" for q in sorted_qs[:5]))

    tokens_used = sum(len(p) // _CHARS_PER_TOKEN for p in parts)

    # In-state facts (highest priority)
    if facts and tokens_used < budget:
        fact_lines = []
        for f in facts:
            line = f"- {f.get('text', '')}"
            tokens_used += len(line) // _CHARS_PER_TOKEN
            if tokens_used > budget:
                break
            fact_lines.append(line)
        if fact_lines:
            parts.append("## Known Facts\n" + "\n".join(fact_lines))

    # Artifact summaries
    if artifacts and tokens_used < budget:
        art_lines = []
        for a in artifacts:
            line = f"- [{a.get('tool_name', '')}] {a.get('key_findings', '')}"
            tokens_used += len(line) // _CHARS_PER_TOKEN
            if tokens_used > budget:
                break
            art_lines.append(line)
        if art_lines:
            parts.append("## Tool Results (summarised)\n" + "\n".join(art_lines))

    # Selected chunk excerpts (lowest priority)
    candidate_map = {c["id"]: c for c in candidates}
    if tokens_used < budget:
        chunk_lines = []
        for cid in selected_ids:
            c = candidate_map.get(cid)
            if not c:
                continue
            text = c.get("text", "")
            est = c.get("token_estimate", len(text) // _CHARS_PER_TOKEN)
            if tokens_used + est > budget:
                continue
            chunk_lines.append(f"[{c.get('source_type', '')}] {text}")
            tokens_used += est
        if chunk_lines:
            parts.append("## Retrieved Context\n" + "\n\n".join(chunk_lines))

    assembled = "\n\n".join(parts)
    result = {"assembled_context": assembled}
    return result, state.update(**result)

# Plan: Burr-based Dynamic Context Agent Refactoring

## TL;DR

Replace pawn-agent's accumulating chat-log context model with a Burr state machine. State is a
`DynamicContextState` Pydantic model; context is retrieved dynamically per step through a 5-node
pipeline; tool results are compressed into facts/summaries rather than appended raw. PydanticAI is
retained for typed LLM calls (planner, selector, compressor, response generator). Existing tools,
pgvector stores, and session persistence tables are reused. `PydanticAgent` is retired entirely;
`BurrAgent` becomes the sole agent.

---

## Current Architecture (what we're replacing)

- **State**: `list[ModelMessage]` accumulated in memory → full history passed every turn → grows unboundedly
- **Tool dispatch**: PydanticAI native tool loop; tool results go directly into LLM context as raw text
- **Persistence**: `agent_session_turns` (Postgres) — append-only raw message turns
- **No Burr**, no dynamic context retrieval, no compression step

---

## Phase 1 — Schemas & Dependencies

1. Add `burr[tracking]` to `pyproject.toml` dependencies
2. Create `pawn_agent/core/burr_state.py`:
   - `Fact`, `ArtifactSummary`, `OpenQuestion` Pydantic models (per spec)
   - `DynamicContextState` Pydantic model (`user_goal`, `constraints`, `plan`, `open_questions`,
     `facts`, `artifacts`, `recent_messages`, `selected_context_ids`, `token_budget_for_context`,
     `next_action`)
   - `ContextCandidate` and `ContextSelection` models (retrieval pipeline schemas)
   - Helpers: `state_to_burr_dict()` / `burr_dict_to_state()` for Burr ↔ Pydantic conversion

---

## Phase 2 — Dynamic Context Subgraph

Create `pawn_agent/core/burr_context.py` — 5 pure action functions for the reusable context pipeline:

3. `derive_retrieval_query(state: DynamicContextState) -> dict`
   Builds query dict from goal + top-3 open questions + constraints.

4. `retrieve_context_candidates(query, state) -> list[ContextCandidate]`
   Searches existing pgvector `text_chunks` (reusing `search_knowledge` logic) + scans in-state
   facts/artifacts. Returns top-30 sorted by score.

5. `rank_context_candidates(candidates, state) -> list[ContextCandidate]`
   Deterministic re-rank by: exact entity match, freshness, source confidence, conflict status,
   token cost.

6. `select_context_with_model(candidates, state, cfg) -> ContextSelection`
   PydanticAI sub-agent (configurable fast model) returning typed `ContextSelection`. Loops back to
   step 4 if `need_additional_retrieval=True`.

7. `assemble_prompt_context(state, selection) -> str`
   Builds minimal prompt string: goal + constraints + selected facts + selected summaries + current
   question. Enforces token budget (facts > summaries > raw excerpts).

---

## Phase 3 — Main Action Graph

Create `pawn_agent/core/burr_actions.py` — 5 Burr action functions:

8. `planner(state, cfg) -> DynamicContextState`
   PydanticAI call on assembled context. Outputs updated `plan`, `open_questions`, `next_action`.

9. `tool_router(state, cfg) -> DynamicContextState`
   PydanticAI call: inspects plan/open_questions, selects tool name **and typed arguments** via
   a `ToolCall(tool_name: str, arguments: dict)` output schema. Stores selection in state.

10. `tool_executor(state, cfg, tools_registry) -> DynamicContextState`
    Invokes selected tool via `tool.function` (introspected from `pydantic_ai.Tool` objects keyed
    by name). Stores raw result in state for the compressor.

11. `state_compressor(state, cfg) -> DynamicContextState`
    PydanticAI call: extracts normalized `Fact` list, compact `ArtifactSummary`, entities from raw
    tool result. Persists summary to `text_chunks` via existing `vectorize_memory()`. Discards raw
    payload from context.

12. `response_generator(state, cfg) -> str`
    PydanticAI call on assembled context. Outputs final answer — no raw logs, no repeated history.

---

## Phase 4 — Burr Application Wiring

Create `pawn_agent/core/burr_agent.py`:

13. Define Burr `Application`:
    - **Actions**: planner → tool_router → tool_executor → state_compressor → response_generator
    - **Context subgraph** injected before planner and tool_executor
    - **Transitions**: explicit per spec; includes the `need_additional_retrieval` loop in
      `select_context_with_model`
    - **Burr tracker**: Postgres backend on existing `cfg.db_dsn`

    Transition pattern:
    ```
    derive_retrieval_query
      -> retrieve_context_candidates
      -> rank_context_candidates
      -> select_context_with_model
          if need_additional_retrieval:
              -> retrieve_context_candidates
          else:
              -> assemble_prompt_context
      -> [planner | tool_executor]
      -> [tool_router | state_compressor]
      -> response_generator
    ```

14. `BurrAgent` class:
    - Public interface mirrors retired `PydanticAgent`: `run()`, `run_async()`, `chat()`
    - Wraps Burr `Application`
    - Persistence bridge: serializes `DynamicContextState` as JSON into `agent_session_turns`
      (one row per turn) for continuity with existing session history tooling

---

## Phase 5 — CLI Integration

15. `pawn_agent/cli/commands.py`: replace `PydanticAgent` instantiation with `BurrAgent` —
    unconditional, no flag
16. **Delete** `pawn_agent/core/pydantic_agent.py` (retired)
17. `pawn_agent/tools/__init__.py`: add `build_tools_registry(cfg) -> dict[str, pydantic_ai.Tool]`
    returning tools keyed by name (needed by `tool_executor`)

---

## Relevant Files

| File | Role |
|---|---|
| `pawn_agent/core/pydantic_agent.py` | Reference only → DELETE after Phase 5; key patterns: `_resolve_model()`, `chat()` REPL loop, `_coerce_tool_call_content()` |
| `pawn_agent/core/llm_sub.py` | Pattern for cheap PydanticAI sub-calls (selector, compressor, planner) |
| `pawn_agent/core/session_store.py` | `append_turn()`, `load_history()` — reused for state persistence bridge |
| `pawn_agent/tools/__init__.py` | `build_tools()` → extend with `build_tools_registry()` |
| `pawn_agent/utils/vectorize.py` | `vectorize_memory()` — reused by `state_compressor` |
| `pawn_agent/utils/config.py` | `AgentConfig` — passed to all Burr actions |
| `pawn_agent/cli/commands.py` | `run` and `chat` commands — swap agent class |
| `pyproject.toml` | Add `burr[tracking]` dependency |

---

## Verification Checklist

1. `pip install -e ".[dev]"` resolves with `burr` added — no conflicts
2. `pytest tests/test_agent.py` — update or replace tests that reference `PydanticAgent` directly
3. `pawn-agent run "list sessions"` — completes a full planner→router→executor→compressor→response cycle end-to-end
4. Inspect Burr state after a tool call — raw tool output absent from `recent_messages`; fact extracted in `state.facts`
5. `pawn-agent chat` multi-turn — context size stays bounded across turns (does not grow linearly)
6. After session: new rows in `text_chunks` containing compressed summaries/facts from tool results

---

## Decisions

- **Full replacement**: `PydanticAgent` deleted; `BurrAgent` is the only agent
- **No new infra**: reuse existing pgvector (`text_chunks`) as retrieval store, Postgres for Burr tracker on same DSN
- **Reuse existing tools** via `tool.function` introspection — no tool rewrites
- **State bridge**: `DynamicContextState` serialized as JSON to `agent_session_turns` per turn
- **Excluded**: new DB tables, changes to tool module contracts, changes to `pawn_diarize`

---

## Open Questions

1. **Tool argument dispatch**: `tool_router` must output typed arguments alongside the tool name.
   Suggest `tool_router` returns a `ToolCall(tool_name: str, arguments: dict)` output schema;
   `tool_executor` unpacks and passes `**arguments` to `tool.function`.

2. **Dual-write vs single-write persistence**: Burr's Postgres tracker holds Burr-internal state;
   `agent_session_turns` holds human-readable display turns. Recommend dual-write: Burr tracker for
   graph replay, `agent_session_turns` for the chat history display and existing `/history` tooling.

3. **Selector model**: spec suggests `gpt-4.1-mini`; project currently uses Ollama/Gemma locally.
   Recommend adding `selector_model` key to `AgentConfig` defaulting to `cfg.pydantic_model`, so
   a fast cloud model can be plugged in independently when available.

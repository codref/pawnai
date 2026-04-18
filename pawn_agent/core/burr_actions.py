"""Main action graph for the Burr agent.

Five Burr ``@action`` functions implement the core agentic loop:

  planner → tool_router → tool_executor → state_compressor → response_generator

Each action reads from / writes to the shared ``burr.State`` dict (keys mirror
``DynamicContextState`` fields).  Heavy ML imports are deferred to first use.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from burr.core import State, action

from pawn_agent.utils.config import AgentConfig

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_MAX_RECENT_MESSAGES = 10  # keep last N user/assistant turns in state


def _load_anima(cfg: AgentConfig) -> str:
    """Return the anima/persona text, or empty string if not configured."""
    path = getattr(cfg, "anima_path", None)
    if not path:
        return ""
    try:
        from pathlib import Path
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return ""


# ── Shared PydanticAI call helper ─────────────────────────────────────────────


async def _llm_call(
    cfg: AgentConfig,
    prompt: str,
    system_prompt: str,
    output_type: Any = str,
    *,
    label: str = "llm",
) -> Any:
    """Run a single-turn PydanticAI completion and return typed output."""
    import os

    from pydantic_ai import Agent

    model_str = cfg.pydantic_model
    api_key = cfg.pydantic_api_key
    base_url = cfg.pydantic_base_url

    logger.info("[%s] → %s (output_type=%s)", label, model_str, getattr(output_type, "__name__", output_type))

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

    agent: Agent = Agent(model, system_prompt=system_prompt, output_type=output_type, retries=0)
    result = await agent.run(prompt)
    logger.info("[%s] ← done", label)
    return result.output


def _run_async(coro: Any) -> Any:
    """Run *coro* regardless of whether there is already a running event loop."""
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
        # Already inside an event loop — run in a fresh thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=120)
    except RuntimeError:
        return asyncio.run(coro)


# ── Pydantic output schemas for LLM calls ─────────────────────────────────────


from pydantic import BaseModel, Field  # noqa: E402


class _PlannerOutput(BaseModel):
    plan: str = ""
    open_questions: list[dict] = Field(default_factory=list)
    next_action: str = "respond"  # "respond" | tool name


class _ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class _CompressorOutput(BaseModel):
    facts: list[dict] = Field(default_factory=list)
    key_findings: str = ""


# ── 1. planner ────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are a planner for a conversational AI agent.
Given the current context and conversation history, produce an updated plan,
a list of open questions (with integer priority where higher = more urgent),
and the next action.

For simple conversational messages (greetings, small talk, direct questions
that can be answered from general knowledge or history), set next_action to
"respond" — do NOT call tools unnecessarily.
If a tool is needed to retrieve external information, set next_action to the
tool name (e.g. "search_knowledge").

Respond with JSON matching the schema:
{{"plan": "...", "open_questions": [{{"question": "...", "priority": 1}}], "next_action": "respond"}}
"""


@action(
    reads=["assembled_context", "plan", "open_questions", "next_action", "recent_messages"],
    writes=["plan", "open_questions", "next_action"],
)
def planner(state: State, cfg: AgentConfig) -> tuple[dict, State]:
    """Update plan, open questions, and decide next action."""
    context: str = state.get("assembled_context", "")
    current_plan: str = state.get("plan", "")
    recent: list[dict] = state.get("recent_messages", [])

    history_text = ""
    if recent:
        lines = []
        for m in recent[-_MAX_RECENT_MESSAGES:]:
            role = m.get("role", "?")
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        history_text = "\n".join(lines)

    prompt = (
        f"Current plan: {current_plan}\n\n"
        + (f"Recent conversation:\n{history_text}\n\n" if history_text else "")
        + f"Context:\n{context}"
    )

    try:
        output: _PlannerOutput = _run_async(
            _llm_call(cfg, prompt, _PLANNER_SYSTEM, output_type=_PlannerOutput, label="planner")
        )
        result = {
            "plan": output.plan,
            "open_questions": output.open_questions,
            "next_action": output.next_action,
        }
    except Exception as exc:
        logger.warning("planner LLM call failed: %s", exc)
        result = {
            "plan": current_plan,
            "open_questions": state.get("open_questions", []),
            "next_action": "respond",
        }

    return result, state.update(**result)


# ── 2. tool_router ────────────────────────────────────────────────────────────

_ROUTER_SYSTEM = """\
You are a tool selector. Given the current plan and open questions, choose the
best tool and provide its arguments as a flat dict.

Respond with JSON:
{"tool_name": "...", "arguments": {...}}
"""


@action(
    reads=["assembled_context", "plan", "open_questions", "next_action"],
    writes=["pending_tool_call"],
)
def tool_router(state: State, cfg: AgentConfig, tools_registry: dict[str, Any]) -> tuple[dict, State]:
    """Select the tool and arguments to invoke next."""
    context: str = state.get("assembled_context", "")
    plan: str = state.get("plan", "")
    open_qs: list[dict] = state.get("open_questions", [])
    available_tools = list(tools_registry.keys())

    prompt = (
        f"Plan: {plan}\n\n"
        f"Open questions: {json.dumps(open_qs)}\n\n"
        f"Available tools: {json.dumps(available_tools)}\n\n"
        f"Context:\n{context}"
    )

    try:
        output: _ToolCall = _run_async(
            _llm_call(cfg, prompt, _ROUTER_SYSTEM, output_type=_ToolCall, label="tool_router")
        )
        pending = {"tool_name": output.tool_name, "arguments": output.arguments}
    except Exception as exc:
        logger.warning("tool_router LLM call failed: %s", exc)
        # Fallback: use next_action as tool name with no args
        pending = {"tool_name": state.get("next_action", ""), "arguments": {}}

    result = {"pending_tool_call": pending}
    return result, state.update(**result)


# ── 3. tool_executor ──────────────────────────────────────────────────────────


@action(reads=["pending_tool_call"], writes=["raw_tool_result"])
def tool_executor(state: State, tools_registry: dict[str, Any]) -> tuple[dict, State]:
    """Call the selected tool and store its raw result."""
    pending: dict = state.get("pending_tool_call") or {}
    tool_name: str = pending.get("tool_name", "")
    arguments: dict = pending.get("arguments", {})

    raw: str = ""
    if not tool_name:
        raw = "[tool_executor] No tool selected."
    elif tool_name not in tools_registry:
        raw = f"[tool_executor] Unknown tool: {tool_name!r}. Available: {list(tools_registry)}"
    else:
        fn = tools_registry[tool_name]
        try:
            import asyncio
            import inspect

            if inspect.iscoroutinefunction(fn):
                raw = str(_run_async(fn(**arguments)))
            else:
                raw = str(fn(**arguments))
        except Exception as exc:
            raw = f"[tool_executor] Tool {tool_name!r} raised {type(exc).__name__}: {exc}"
            logger.warning(raw)

    result = {"raw_tool_result": raw}
    return result, state.update(**result)


# ── 4. state_compressor ───────────────────────────────────────────────────────

_COMPRESSOR_SYSTEM = """\
You are a fact extractor. Given raw tool output, extract:
1. A list of discrete facts (short declarative sentences).
2. A brief key_findings string summarising the most important finding.

Respond with JSON:
{"facts": [{"text": "...", "source": "...", "confidence": 0.9, "entity_ids": []}], "key_findings": "..."}
"""


@action(
    reads=["raw_tool_result", "pending_tool_call", "facts", "artifacts"],
    writes=["facts", "artifacts", "raw_tool_result"],
)
def state_compressor(state: State, cfg: AgentConfig) -> tuple[dict, State]:
    """Extract facts from raw tool result, persist to RAG store, discard raw payload."""
    raw: str = state.get("raw_tool_result") or ""
    pending: dict = state.get("pending_tool_call") or {}
    tool_name: str = pending.get("tool_name", "tool")
    existing_facts: list[dict] = list(state.get("facts", []))
    existing_artifacts: list[dict] = list(state.get("artifacts", []))

    if not raw or raw.startswith("[tool_executor]"):
        # Nothing useful to compress
        result = {"facts": existing_facts, "artifacts": existing_artifacts, "raw_tool_result": None}
        return result, state.update(**result)

    prompt = f"Tool: {tool_name}\n\nRaw output:\n{raw[:4000]}"

    try:
        output: _CompressorOutput = _run_async(
            _llm_call(cfg, prompt, _COMPRESSOR_SYSTEM, output_type=_CompressorOutput, label="state_compressor")
        )
        new_facts = output.facts
        key_findings = output.key_findings
    except Exception as exc:
        logger.warning("state_compressor LLM call failed: %s", exc)
        new_facts = []
        key_findings = raw[:500]

    # Merge facts (cap at 50 to bound state size)
    merged_facts = (existing_facts + new_facts)[-50:]

    # Build artifact summary
    token_est = len(key_findings) // _CHARS_PER_TOKEN
    new_artifact = {
        "tool_name": tool_name,
        "key_findings": key_findings,
        "token_estimate": token_est,
    }
    merged_artifacts = (existing_artifacts + [new_artifact])[-20:]

    # Persist new facts to the RAG store so future retrieval can find them
    if new_facts and cfg.db_dsn:
        try:
            from pawn_agent.utils.vectorize import vectorize_memory

            for fact in new_facts:
                vectorize_memory(
                    fact.get("text", ""),
                    cfg.db_dsn,
                    embed_model=cfg.embed_model,
                    embed_device=cfg.embed_device,
                    truncate_dim=cfg.embed_dim or None,
                    local_files_only=cfg.embed_local_files_only,
                    tags=fact.get("entity_ids", []),
                )
        except Exception as exc:
            logger.warning("vectorize_memory failed: %s", exc)

    result = {
        "facts": merged_facts,
        "artifacts": merged_artifacts,
        "raw_tool_result": None,  # discard raw payload from context
    }
    return result, state.update(**result)


# ── 5. response_generator ─────────────────────────────────────────────────────

_RESPONSE_SYSTEM = """\
You are a helpful conversational assistant. Answer the user's message naturally
and accurately. Use the provided context when it is relevant; for simple
conversational messages (greetings, small talk, general questions) you may
respond from general knowledge without needing specific context. Be concise.
Do not repeat raw tool outputs verbatim.
"""


@action(
    reads=["assembled_context", "user_goal", "recent_messages", "facts", "artifacts"],
    writes=["recent_messages"],
)
def response_generator(state: State, cfg: AgentConfig) -> tuple[dict, State]:
    """Generate the final response and append it to recent_messages."""
    context: str = state.get("assembled_context", "")
    goal: str = state.get("user_goal", "")
    recent: list[dict] = list(state.get("recent_messages", []))
    facts: list[dict] = state.get("facts", [])
    artifacts: list[dict] = state.get("artifacts", [])

    # Build conversation history for the model
    history_text = ""
    if recent:
        lines = [f"{m.get('role','?')}: {m.get('content','')}" for m in recent[-_MAX_RECENT_MESSAGES:]]
        history_text = "\n".join(lines)

    # Include tool results (most recent artifact key findings + facts)
    tool_section = ""
    if artifacts:
        last = artifacts[-1]
        tool_section += f"Tool result ({last.get('tool_name','tool')}):\n{last.get('key_findings','')}\n"
    if facts:
        fact_lines = [f"- {f.get('text','')}" for f in facts[-10:] if f.get("text")]
        if fact_lines:
            tool_section += "Known facts:\n" + "\n".join(fact_lines) + "\n"

    prompt = (
        (f"Conversation so far:\n{history_text}\n\n" if history_text else "")
        + (f"Relevant context:\n{context}\n\n" if context.strip() else "")
        + (f"{tool_section}\n" if tool_section else "")
        + f"User: {goal}"
    )

    # Load persona/anima and merge with base system prompt
    anima = _load_anima(cfg)
    system = _RESPONSE_SYSTEM + (f"\n\n{anima}" if anima else "")

    try:
        response: str = _run_async(
            _llm_call(cfg, prompt, system, output_type=str, label="response_generator")
        )
    except Exception as exc:
        logger.error("response_generator LLM call failed: %s", exc)
        response = f"I encountered an error generating a response: {exc}"

    recent.append({"role": "user", "content": goal})
    recent.append({"role": "assistant", "content": response})
    # Keep only the last N turns
    recent = recent[-(_MAX_RECENT_MESSAGES * 2):]

    result = {"recent_messages": recent}
    return result, state.update(**result)

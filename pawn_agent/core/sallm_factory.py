"""Build a durable sallm.Agent from AgentConfig.

Owns: model-id mapping, state_dir layout, optional Tempo/Prometheus tracer,
and wiring of pawn CliTools + skills.

Does not own: per-conversation pooling (see sallm_registry.py) or the
async turn façade (see sallm_session.py).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from sallm import Agent, EmbeddingProfile, RetrievalConfig
from sallm.prom import SessionMetrics
from sallm.trace import Tracer, jsonl_sink, multi_sink, otlp_http_sink

from pawn_agent.core.sallm_skills import build_pawn_skills
from pawn_agent.core.sallm_tools import build_pawn_clitools
from pawn_agent.profiles import load_profile_from_config
from pawn_agent.utils.config import AgentConfig

logger = logging.getLogger(__name__)


def resolve_state_dir(cfg: AgentConfig) -> Path:
    """Return the absolute directory that holds ``state.db`` and ``vectors/``."""
    raw = Path(cfg.sallm.state_dir).expanduser()
    if not raw.is_absolute():
        raw = (Path.cwd() / raw).resolve()
    return raw


def _ensure_api_key_env(cfg: AgentConfig) -> None:
    """LiteLLM reads provider keys from the environment.

    If pawnai.yaml supplied an api_key but the matching env var is empty,
    export it so CliTool-less chat completions still authenticate.
    """
    key = cfg.pydantic_api_key
    if not key:
        return
    # Prefer OpenAI-compatible env; LiteLLM also honors provider-specific vars.
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = key
    model = cfg.litellm_model
    if model.startswith("anthropic/") and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = key
    if model.startswith(("gemini/", "google/")) and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = key


def build_optional_tracer(
    *,
    session_id: str,
    cfg: AgentConfig,
    jsonl_path: Optional[Path] = None,
) -> Optional[Tracer]:
    """Build a sallm Tracer when OTLP, metrics, or JSONL tracing is enabled."""
    otlp = (cfg.sallm.otlp_endpoint or "").strip() or None
    metrics_port = int(cfg.sallm.metrics_port or 0)
    sinks = []
    if jsonl_path is not None:
        sinks.append(jsonl_sink(str(jsonl_path)))
    if otlp:
        sinks.append(otlp_http_sink(otlp))
    if not sinks and not metrics_port:
        return None

    emit = sinks[0] if len(sinks) == 1 else (multi_sink(*sinks) if sinks else (lambda _e: None))
    tracer = Tracer(emit, session_id=session_id)
    if metrics_port:
        metrics = SessionMetrics(tracer.session_id)
        metrics.start_server(port=metrics_port)
        tracer.metrics = metrics
    return tracer


def build_sallm_agent(
    cfg: AgentConfig,
    *,
    conversation_id: str,
    trace: Optional[Tracer] = None,
) -> Agent:
    """Construct one durable Agent for *conversation_id*.

    Layout under ``state_dir``::

        state.db          # SQLite: messages, skill stack, chunks, facts
        vectors/          # LanceDB index (rebuildable from SQLite)

    Resume = same ``state_dir`` + same ``conversation_id``.
    """
    state_dir = resolve_state_dir(cfg)
    state_dir.mkdir(parents=True, exist_ok=True)

    _ensure_api_key_env(cfg)

    embedding = EmbeddingProfile(
        model=cfg.sallm.embedding_model,
        api_base=cfg.sallm.embedding_api_base,
    )

    # api_base: OpenAI-compatible providers need the custom base (e.g. Ollama
    # /v1). Pure cloud OpenAI can leave this None and LiteLLM uses defaults.
    api_base = cfg.pydantic_base_url

    compiled = load_profile_from_config(cfg.sallm.profile)

    logger.debug(
        "Building sallm Agent conversation_id=%r model=%s state_dir=%s profile=%s",
        conversation_id,
        cfg.litellm_model,
        state_dir,
        cfg.sallm.profile,
    )

    return Agent(
        model=cfg.litellm_model,
        api_base=api_base,
        tools=build_pawn_clitools(),
        skills=build_pawn_skills(),
        state_path=state_dir / "state.db",
        vector_path=state_dir / "vectors",
        session_id=conversation_id,
        embedding_profile=embedding,
        retrieval=RetrievalConfig(
            memory_gate=True,
            search_mode="dense",
            use_instruct=True,
            use_rewrite=True,
            use_hyde=False,
        ),
        max_steps=int(cfg.sallm.max_steps),
        compiled_profile=compiled,
        trace=trace,
    )


def rebuild_agent_for_config(
    agent: Agent,
    cfg: AgentConfig,
) -> Agent:
    """Rebuild an Agent after a per-turn model override (same session id/paths).

    Used when queue/API passes ``model=`` for one run without changing the
    durable conversation identity.
    """
    return build_sallm_agent(
        cfg,
        conversation_id=agent.session_id,
        trace=agent.trace,
    )

"""Minimal Burr-managed chat path for evaluating dynamic context orchestration."""

from __future__ import annotations

import os
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from prompt_toolkit import PromptSession

from pawn_agent.utils.config import AgentConfig


def new_burr_chat_state() -> dict[str, Any]:
    """Return the default Burr chat state."""
    return {
        "chat_history": [],
        "latest_user_message": "",
        "latest_assistant_message": "",
        "turn_count": 0,
    }


def apply_user_message(state: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Return a copy of *state* updated with the next user message."""
    history = list(state.get("chat_history", []))
    history.append({"role": "user", "content": prompt})
    return {
        **state,
        "chat_history": history,
        "latest_user_message": prompt,
        "turn_count": int(state.get("turn_count", 0)) + 1,
    }


def apply_assistant_message(state: dict[str, Any], reply: str) -> dict[str, Any]:
    """Return a copy of *state* updated with the next assistant message."""
    history = list(state.get("chat_history", []))
    history.append({"role": "assistant", "content": reply})
    return {
        **state,
        "chat_history": history,
        "latest_assistant_message": reply,
    }


def _normalize_output(value: object) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def _normalize_graph_output_path(output_path: str) -> tuple[str, str]:
    """Return ``(graphviz_output_path, format)`` for Burr visualization."""
    path = Path(output_path)
    fmt = path.suffix.lstrip(".") or "png"
    base = path.with_suffix("") if path.suffix else path
    return str(base), fmt


def _import_pydantic_agent_cls():
    try:
        from pydantic_ai import Agent
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "Burr chat requires the optional 'pydantic-ai' dependency to be installed."
        ) from exc
    return Agent


def _import_pydantic_messages():
    try:
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "Burr chat requires the optional 'pydantic-ai' dependency to be installed."
        ) from exc
    return ModelRequest, ModelResponse, TextPart, UserPromptPart


def _import_openai_model_classes():
    try:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "Burr chat requires the optional 'pydantic-ai' OpenAI provider support."
        ) from exc
    return OpenAIChatModel, OpenAIProvider


def _import_burr_core():
    try:
        from burr.core import Action, ApplicationBuilder, default
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "Burr chat requires the optional 'burr[start]' dependency to be installed."
        ) from exc
    return Action, ApplicationBuilder, default


def _import_burr_tracker_cls():
    try:
        from burr.tracking import LocalTrackingClient
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "Burr chat requires the optional 'burr[start]' dependency to be installed."
        ) from exc
    return LocalTrackingClient


def _import_burr_postgres_persister_cls():
    try:
        from burr.integrations.persisters.b_asyncpg import AsyncPostgreSQLPersister
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "Burr chat PostgreSQL persistence requires Burr's asyncpg plugin. "
            "Install or reinstall dependencies with 'burr[start,asyncpg]'."
        ) from exc
    return AsyncPostgreSQLPersister


def parse_postgres_dsn(dsn: str) -> dict[str, Any]:
    """Parse a SQLAlchemy-style PostgreSQL DSN into Burr persister parameters."""
    normalized = dsn
    if normalized.startswith("postgresql+"):
        normalized = "postgresql://" + normalized.split("://", 1)[1]
    parsed = urllib.parse.urlparse(normalized)
    if parsed.scheme not in {"postgresql", "postgres"}:
        raise ValueError(f"Unsupported PostgreSQL DSN scheme: {parsed.scheme!r}")
    db_name = parsed.path.lstrip("/")
    if not db_name:
        raise ValueError("PostgreSQL DSN must include a database name.")
    if parsed.username is None:
        raise ValueError("PostgreSQL DSN must include a username.")
    return {
        "db_name": db_name,
        "user": urllib.parse.unquote(parsed.username),
        "password": urllib.parse.unquote(parsed.password or ""),
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
    }


class PlainPydanticChatAgent:
    """Tool-less PydanticAI chat agent used by the Burr evaluation path."""

    def __init__(
        self,
        cfg: AgentConfig,
        on_thinking: Optional[Callable[[], None]] = None,
    ) -> None:
        self.cfg = cfg
        self._on_thinking = on_thinking
        self._agent = self._build_agent()

    def _load_anima(self) -> str | None:
        if not self.cfg.anima_path:
            return None
        path = Path(self.cfg.anima_path)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8").strip() or None

    def _resolve_model(self):
        model_str = self.cfg.pydantic_model
        api_key = self.cfg.pydantic_api_key
        base_url = self.cfg.pydantic_base_url

        if base_url and model_str.startswith("openai:"):
            OpenAIChatModel, OpenAIProvider = _import_openai_model_classes()

            model_name = model_str[len("openai:"):]
            provider = OpenAIProvider(base_url=base_url, api_key=api_key or "no-key")

            class _CompatModel(OpenAIChatModel):
                def _map_model_response(self, message):
                    param = super()._map_model_response(message)
                    if param.get("content") is None:
                        param["content"] = ""
                    return param

            return _CompatModel(model_name, provider=provider)

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

    def _build_agent(self):
        Agent = _import_pydantic_agent_cls()
        system_prompts: list[str] = []
        anima = self._load_anima()
        if anima:
            system_prompts.append(anima)
        return Agent(
            self._resolve_model(),
            system_prompt=tuple(system_prompts),
            output_retries=2,
        )

    def _history_to_messages(self, chat_history: list[dict[str, str]]) -> list:
        ModelRequest, ModelResponse, TextPart, UserPromptPart = _import_pydantic_messages()

        messages: list = []
        for item in chat_history:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
            elif role == "assistant":
                messages.append(
                    ModelResponse(
                        parts=[TextPart(content=content)],
                        model_name=self.cfg.pydantic_model,
                        timestamp=datetime.now(timezone.utc),
                    )
                )
        return messages

    async def reply(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
        if self._on_thinking is not None:
            self._on_thinking()

        prior_history = list(chat_history)
        if (
            prior_history
            and prior_history[-1].get("role") == "user"
            and prior_history[-1].get("content") == user_prompt
        ):
            prior_history = prior_history[:-1]

        result = await self._agent.run(
            user_prompt,
            message_history=self._history_to_messages(prior_history),
        )
        return _normalize_output(getattr(result, "output", ""))


def make_burr_tracker(cfg: AgentConfig):
    """Build the Burr local tracker used for UI visibility."""
    LocalTrackingClient = _import_burr_tracker_cls()
    kwargs = {"project": cfg.burr.project}
    if cfg.burr.storage_dir:
        kwargs["storage_dir"] = cfg.burr.storage_dir
    return LocalTrackingClient(**kwargs)


async def make_burr_postgres_persister(cfg: AgentConfig):
    """Build the Burr PostgreSQL persister for persisted state snapshots."""
    AsyncPostgreSQLPersister = _import_burr_postgres_persister_cls()
    params = parse_postgres_dsn(cfg.db_dsn)
    persister = await AsyncPostgreSQLPersister.from_values(
        params["db_name"],
        params["user"],
        params["password"],
        params["host"],
        params["port"],
        table_name=cfg.burr.table_name,
    )
    await persister.initialize()
    return persister


async def build_burr_chat_application(
    cfg: AgentConfig,
    chat_agent: PlainPydanticChatAgent,
    *,
    app_id: str,
    partition_key: str = "",
    tracker: object | None = None,
    persister: object | None = None,
):
    """Build one async Burr application instance."""
    Action, ApplicationBuilder, default = _import_burr_core()
    default_state = new_burr_chat_state()

    class HumanInputAction(Action):
        def __init__(self) -> None:
            super().__init__()

        @property
        def reads(self) -> list[str]:
            return ["chat_history", "turn_count"]

        @property
        def writes(self) -> list[str]:
            return ["chat_history", "latest_user_message", "turn_count"]

        @property
        def inputs(self) -> list[str]:
            return ["prompt"]

        def run(self, state, prompt: str) -> dict:
            return {"prompt": prompt}

        def update(self, result: dict, state):
            prompt = _normalize_output(result.get("prompt", ""))
            return (
                state.update(
                    latest_user_message=prompt,
                    turn_count=int(state["turn_count"]) + 1,
                )
                .append(chat_history={"role": "user", "content": prompt})
            )

    class AIResponseAction(Action):
        def __init__(self, conversation_agent: PlainPydanticChatAgent) -> None:
            super().__init__()
            self._conversation_agent = conversation_agent

        @property
        def reads(self) -> list[str]:
            return ["chat_history", "latest_user_message"]

        @property
        def writes(self) -> list[str]:
            return ["chat_history", "latest_assistant_message"]

        def is_async(self) -> bool:
            return True

        async def run(self, state, **run_kwargs) -> dict:
            reply = await self._conversation_agent.reply(
                _normalize_output(state["latest_user_message"]),
                list(state["chat_history"]),
            )
            return {"assistant_message": reply}

        def update(self, result: dict, state):
            reply = _normalize_output(result.get("assistant_message", ""))
            return (
                state.update(latest_assistant_message=reply)
                .append(chat_history={"role": "assistant", "content": reply})
            )

    builder = (
        ApplicationBuilder()
        .with_actions(
            human_input=HumanInputAction(),
            ai_response=AIResponseAction(chat_agent),
        )
        .with_transitions(
            ("human_input", "ai_response"),
            ("ai_response", "human_input", default),
        )
    )

    if tracker is not None or persister is not None:
        builder = builder.with_identifiers(app_id=app_id, partition_key=partition_key)

    if persister is not None:
        builder = (
            builder
            .initialize_from(
                persister,
                resume_at_next_action=True,
                default_state=default_state,
                default_entrypoint="human_input",
            )
            .with_state_persister(persister)
        )
    else:
        builder = builder.with_state(**default_state).with_entrypoint("human_input")

    if tracker is not None:
        builder = builder.with_tracker(tracker)

    return await builder.abuild()


class BurrChatSession:
    """Owns the Burr app, tracker, and optional persister for one REPL session."""

    def __init__(
        self,
        cfg: AgentConfig,
        emit: Callable[[str], None],
        on_thinking: Optional[Callable[[], None]] = None,
    ) -> None:
        self.cfg = cfg
        self._emit = emit
        self._chat_agent = PlainPydanticChatAgent(cfg=cfg, on_thinking=on_thinking)
        self._tracker = make_burr_tracker(cfg)
        self._persister = None
        self._app = None
        self._app_id = ""

    @classmethod
    async def create(
        cls,
        cfg: AgentConfig,
        emit: Callable[[str], None],
        on_thinking: Optional[Callable[[], None]] = None,
    ) -> "BurrChatSession":
        session = cls(cfg=cfg, emit=emit, on_thinking=on_thinking)
        if cfg.burr.backend == "postgres":
            session._persister = await make_burr_postgres_persister(cfg)
        await session.reset()
        return session

    async def reset(self) -> None:
        self._app_id = str(uuid.uuid4())
        self._app = await build_burr_chat_application(
            self.cfg,
            self._chat_agent,
            app_id=self._app_id,
            partition_key="chat",
            tracker=self._tracker,
            persister=self._persister,
        )

    async def close(self) -> None:
        cleanup = getattr(self._persister, "cleanup", None)
        if cleanup is not None:
            await cleanup()

    async def handle_user_input(self, text: str) -> str:
        if self._app is None:
            raise RuntimeError("Burr chat application is not initialized.")
        _, _, state = await self._app.arun(
            halt_after=["ai_response"],
            inputs={"prompt": text},
        )
        return _normalize_output(state["latest_assistant_message"])

    def save_graph_image(self, output_path: str) -> None:
        if self._app is None:
            raise RuntimeError("Burr chat application is not initialized.")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph_path, fmt = _normalize_graph_output_path(output_path)
        self._app.visualize(
            output_file_path=graph_path,
            include_state=True,
            format=fmt,
        )


async def run_burr_chat(
    cfg: AgentConfig,
    emit: Callable[[str], None] = print,
    on_thinking: Optional[Callable[[], None]] = None,
    graph_output_path: str | None = None,
) -> None:
    """Run the minimal Burr-backed chat REPL."""
    session = await BurrChatSession.create(cfg=cfg, emit=emit, on_thinking=on_thinking)
    if graph_output_path:
        session.save_graph_image(graph_output_path)
        emit(f"Burr graph saved to {graph_output_path}")
    prompt_session = PromptSession()
    try:
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
                if graph_output_path:
                    session.save_graph_image(graph_output_path)
                    emit(f"Burr graph saved to {graph_output_path}")
                continue
            if text.startswith("/"):
                emit("Burr chat supports only /reset, /exit, and /quit.")
                continue
            emit(await session.handle_user_input(text))
    finally:
        await session.close()

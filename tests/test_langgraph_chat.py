from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pawn_agent.core.langgraph_chat import (
    LangGraphChatSession,
    build_langgraph_chat_graph,
    new_langgraph_chat_state,
)
from pawn_agent.core.langgraph_runtime import (
    PlainSmolagentsChatAgent,
    build_phoenix_tracer,
    resolve_smolagents_model_config,
)
from pawn_agent.utils.config import load_config


def test_new_langgraph_chat_state_resets_history() -> None:
    assert new_langgraph_chat_state() == {
        "incoming_prompt": "",
        "chat_history": [],
        "latest_user_message": "",
        "latest_assistant_message": "",
        "turn_count": 0,
    }


def test_resolve_smolagents_model_config_supports_openai_compatible_base_url() -> None:
    model_name, kwargs = resolve_smolagents_model_config("openai:gpt-4o", None, None)
    assert model_name == "openai/gpt-4o"
    assert kwargs == {}

    local_name, local_kwargs = resolve_smolagents_model_config(
        "openai:gemma4:26b",
        "http://localhost:11434/v1/",
        "ollama",
    )
    assert local_name == "openai/gemma4:26b"
    assert local_kwargs == {
        "api_base": "http://localhost:11434/v1",
        "api_key": "ollama",
    }

    proxy_name, proxy_kwargs = resolve_smolagents_model_config(
        "openai:gpt-4o",
        "https://proxy.example/v1/",
        "secret",
    )
    assert proxy_name == "openai/gpt-4o"
    assert proxy_kwargs == {
        "api_base": "https://proxy.example/v1",
        "api_key": "secret",
    }


def test_resolve_smolagents_model_config_rejects_non_openai_prefix() -> None:
    try:
        resolve_smolagents_model_config("anthropic:claude-sonnet-4-5", None, None)
    except RuntimeError as exc:
        assert "supports only openai:<model>" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected RuntimeError for unsupported provider prefix.")


def test_build_langgraph_chat_graph_registers_nodes_and_edges() -> None:
    calls: dict[str, object] = {}

    class FakeCompiledGraph:
        async def ainvoke(self, payload):
            return payload

    class FakeStateGraph:
        def __init__(self, state_schema) -> None:
            calls["state_schema"] = state_schema
            calls["nodes"] = {}
            calls["edges"] = []

        def add_node(self, name, fn) -> None:
            calls["nodes"][name] = fn

        def add_edge(self, start, end) -> None:
            calls["edges"].append((start, end))

        def compile(self):
            calls["compiled"] = True
            return FakeCompiledGraph()

    fake_agent = SimpleNamespace(model_name="openai/gpt-4o")
    with patch(
        "pawn_agent.core.langgraph_chat._import_langgraph_core",
        return_value=(FakeStateGraph, "__start__", "__end__"),
    ):
        compiled = asyncio.run(build_langgraph_chat_graph(chat_agent=fake_agent))

    assert isinstance(compiled, FakeCompiledGraph)
    assert set(calls["nodes"]) == {"human_input", "ai_response"}
    assert calls["edges"] == [
        ("__start__", "human_input"),
        ("human_input", "ai_response"),
        ("ai_response", "__end__"),
    ]
    assert calls["compiled"] is True


def test_load_config_parses_phoenix_section(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "phoenix:\n"
        "  enabled: true\n"
        "  endpoint: http://localhost:6006/v1/traces\n"
        "  project_name: parakeet-evals\n"
        "  protocol: grpc\n"
        "  api_key: secret-token\n",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_file))

    assert cfg.phoenix_enabled is True
    assert cfg.phoenix_endpoint == "http://localhost:6006/v1/traces"
    assert cfg.phoenix_project_name == "parakeet-evals"
    assert cfg.phoenix_protocol == "grpc"
    assert cfg.phoenix_api_key == "secret-token"


def test_build_phoenix_tracer_is_disabled_by_default(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text("agent:\n  openai:\n    model: gpt-4o\n", encoding="utf-8")

    cfg = load_config(str(cfg_file))

    assert build_phoenix_tracer(cfg) is None


def test_build_phoenix_tracer_passes_configuration() -> None:
    tracer = object()
    register_calls: dict[str, object] = {}
    trace_calls: list[str] = []

    class FakeTraceApi:
        @staticmethod
        def get_tracer(name: str):
            trace_calls.append(name)
            return tracer

    cfg = SimpleNamespace(
        phoenix_enabled=True,
        phoenix_endpoint="http://localhost:6006/v1/traces",
        phoenix_project_name="parakeet-langgraph",
        phoenix_protocol="http/protobuf",
        phoenix_api_key="secret-token",
    )

    def fake_register(**kwargs):
        register_calls.update(kwargs)

    with (
        patch("pawn_agent.core.langgraph_runtime._import_phoenix_register", return_value=fake_register),
        patch("pawn_agent.core.langgraph_runtime._import_trace_api", return_value=FakeTraceApi),
    ):
        built = build_phoenix_tracer(cfg)

    assert built is tracer
    assert register_calls == {
        "project_name": "parakeet-langgraph",
        "endpoint": "http://localhost:6006/v1/traces",
        "protocol": "http/protobuf",
        "auto_instrument": False,
        "batch": True,
        "headers": {"authorization": "Bearer secret-token"},
    }
    assert trace_calls == ["pawn_agent.core.langgraph_runtime"]


def test_plain_smolagents_chat_agent_uses_anima_and_openai_model(tmp_path: Path) -> None:
    anima = tmp_path / "anima.md"
    anima.write_text("You are calm and concise.", encoding="utf-8")

    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "agent:\n"
        f"  anima: {anima}\n"
        "  openai:\n"
        "    model: gpt-4o\n"
        "    base_url: null\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))

    captured: dict[str, object] = {}

    class FakeLiteLLMModel:
        def __init__(self, model_id, **kwargs) -> None:
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs

        def __call__(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(content="ok")

    with patch(
        "pawn_agent.core.langgraph_runtime._import_smolagents_litellm_model",
        return_value=FakeLiteLLMModel,
    ):
        agent = PlainSmolagentsChatAgent(cfg)

    assert agent.model_name == "openai/gpt-4o"
    assert captured["model_id"] == "openai/gpt-4o"
    assert captured["kwargs"] == {}
    assert agent._system_prompt == "You are calm and concise."


def test_plain_smolagents_chat_agent_reuses_prior_history_on_next_turn(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "agent:\n"
        "  openai:\n"
        "    model: gpt-4o\n"
        "    base_url: null\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))

    captured: dict[str, object] = {}

    class FakeLiteLLMModel:
        def __init__(self, model_id, **kwargs) -> None:
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs

        def __call__(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(content=[{"type": "text", "text": "All good."}])

    with patch(
        "pawn_agent.core.langgraph_runtime._import_smolagents_litellm_model",
        return_value=FakeLiteLLMModel,
    ):
        agent = PlainSmolagentsChatAgent(cfg)
        reply = asyncio.run(
            agent.reply(
                "how are you?",
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                    {"role": "user", "content": "how are you?"},
                ],
            )
        )

    assert reply == "All good."
    assert captured["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
        {"role": "user", "content": [{"type": "text", "text": "how are you?"}]},
    ]


def test_plain_smolagents_chat_agent_passes_openai_compatible_base_url(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "agent:\n"
        "  openai:\n"
        "    model: gemma4:26b\n"
        "    base_url: http://localhost:11434/v1\n"
        "    api_key: ollama\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))

    captured: dict[str, object] = {}

    class FakeLiteLLMModel:
        def __init__(self, model_id, **kwargs) -> None:
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs

        def __call__(self, messages):
            return "ok"

    with patch(
        "pawn_agent.core.langgraph_runtime._import_smolagents_litellm_model",
        return_value=FakeLiteLLMModel,
    ):
        agent = PlainSmolagentsChatAgent(cfg)

    assert agent.model_name == "openai/gemma4:26b"
    assert captured["model_id"] == "openai/gemma4:26b"
    assert captured["kwargs"] == {
        "api_base": "http://localhost:11434/v1",
        "api_key": "ollama",
    }


def test_langgraph_chat_session_handles_turns_and_reset(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "agent:\n"
        "  openai:\n"
        "    model: gpt-4o\n"
        "    base_url: null\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))

    payloads: list[dict[str, object]] = []

    class FakeGraph:
        async def ainvoke(self, payload):
            payloads.append(dict(payload))
            history = list(payload.get("chat_history", []))
            history.append({"role": "user", "content": payload["incoming_prompt"]})
            history.append({"role": "assistant", "content": f"echo:{payload['incoming_prompt']}"})
            return {
                "incoming_prompt": "",
                "chat_history": history,
                "latest_user_message": payload["incoming_prompt"],
                "latest_assistant_message": f"echo:{payload['incoming_prompt']}",
                "turn_count": int(payload.get("turn_count", 0)) + 1,
            }

    with (
        patch("pawn_agent.core.langgraph_chat.PlainSmolagentsChatAgent", return_value=object()),
        patch("pawn_agent.core.langgraph_chat.build_langgraph_chat_graph", return_value=FakeGraph()),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        first = asyncio.run(session.handle_user_input("hello"))
        second = asyncio.run(session.handle_user_input("again"))
        asyncio.run(session.reset())

    assert first == "echo:hello"
    assert second == "echo:again"
    assert payloads[0]["chat_history"] == []
    assert payloads[1]["chat_history"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "echo:hello"},
    ]
    assert session._state == new_langgraph_chat_state()


def test_langgraph_chat_session_emits_phoenix_spans(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "agent:\n"
        "  name: Bob\n"
        "  openai:\n"
        "    model: gpt-4o\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    span_events: list[tuple[str, str, object | None, object | None]] = []

    class FakeSpan:
        def __init__(self, name: str) -> None:
            self._name = name

        def __enter__(self):
            span_events.append(("enter", self._name, None, None))
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            span_events.append(("exit", self._name, None, None))

        def set_attribute(self, key: str, value: object) -> None:
            span_events.append(("attr", self._name, key, value))

    class FakeTracer:
        def start_as_current_span(self, name: str):
            return FakeSpan(name)

    class FakeCompiledGraph:
        def __init__(self, nodes) -> None:
            self._nodes = nodes

        async def ainvoke(self, payload):
            state = dict(payload)
            state = self._nodes["human_input"](state)
            return await self._nodes["ai_response"](state)

    class FakeStateGraph:
        def __init__(self, _state_schema) -> None:
            self._nodes: dict[str, object] = {}

        def add_node(self, name, fn) -> None:
            self._nodes[name] = fn

        def add_edge(self, _start, _end) -> None:
            return None

        def compile(self):
            return FakeCompiledGraph(self._nodes)

    class FakeChatAgent:
        model_name = "openai/gpt-4o"

        async def reply(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            assert user_prompt == "hello"
            assert chat_history == [{"role": "user", "content": "hello"}]
            return "echo:hello"

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch("pawn_agent.core.langgraph_chat.PlainSmolagentsChatAgent", return_value=FakeChatAgent()),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=FakeTracer()),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        reply = asyncio.run(session.handle_user_input("hello"))

    assert reply == "echo:hello"
    span_names = [event[1] for event in span_events if event[0] == "enter"]
    assert span_names == [
        "langgraph-chat-turn",
        "langgraph-human-input",
        "langgraph-ai-response",
    ]
    assert (
        "attr",
        "langgraph-ai-response",
        "llm.model_name",
        "openai/gpt-4o",
    ) in span_events

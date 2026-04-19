from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pawn_agent.core.burr_chat import (
    BurrChatSession,
    PlainPydanticChatAgent,
    _normalize_graph_output_path,
    apply_assistant_message,
    apply_user_message,
    make_burr_postgres_persister,
    make_burr_tracker,
    new_burr_chat_state,
    parse_postgres_dsn,
)
from pawn_agent.utils.config import load_config


def test_load_config_parses_burr_section(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "burr:\n"
        "  enabled: true\n"
        "  project: eval-chat\n"
        "  backend: local\n"
        "  storage_dir: /tmp/burr-demo\n"
        "  table_name: burr_turns\n",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_file))

    assert cfg.burr.enabled is True
    assert cfg.burr.project == "eval-chat"
    assert cfg.burr.backend == "local"
    assert cfg.burr.storage_dir == "/tmp/burr-demo"
    assert cfg.burr.table_name == "burr_turns"


def test_burr_state_helpers_append_messages_in_order() -> None:
    state = new_burr_chat_state()
    state = apply_user_message(state, "hello")
    state = apply_assistant_message(state, "hi there")
    state = apply_user_message(state, "how are you?")

    assert state["turn_count"] == 2
    assert state["latest_user_message"] == "how are you?"
    assert state["latest_assistant_message"] == "hi there"
    assert state["chat_history"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "how are you?"},
    ]


def test_new_burr_chat_state_resets_history() -> None:
    assert new_burr_chat_state() == {
        "chat_history": [],
        "latest_user_message": "",
        "latest_assistant_message": "",
        "turn_count": 0,
    }


def test_normalize_graph_output_path_uses_suffix_as_format() -> None:
    assert _normalize_graph_output_path("/tmp/graph.png") == ("/tmp/graph", "png")
    assert _normalize_graph_output_path("/tmp/graph.svg") == ("/tmp/graph", "svg")
    assert _normalize_graph_output_path("/tmp/graph") == ("/tmp/graph", "png")


def test_plain_pydantic_chat_agent_uses_toolless_agent_and_anima(tmp_path: Path) -> None:
    anima = tmp_path / "anima.md"
    anima.write_text("You are calm and concise.", encoding="utf-8")

    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "agent:\n"
        "  anima: " + str(anima) + "\n"
        "  openai:\n"
        "    model: gpt-4o\n"
        "    base_url: null\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))

    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, model, **kwargs) -> None:
            captured["model"] = model
            captured["kwargs"] = kwargs

        async def run(self, *args, **kwargs):
            return SimpleNamespace(output="ok")

    with patch(
        "pawn_agent.core.burr_chat._import_pydantic_agent_cls",
        return_value=FakeAgent,
    ):
        PlainPydanticChatAgent(cfg)

    assert captured["model"] == "openai:gpt-4o"
    assert captured["kwargs"]["system_prompt"] == ("You are calm and concise.",)
    assert captured["kwargs"]["output_retries"] == 2
    assert "tools" not in captured["kwargs"]


def test_plain_pydantic_chat_agent_reuses_prior_history_on_next_turn(tmp_path: Path) -> None:
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

    class FakeModelRequest:
        def __init__(self, *, parts) -> None:
            self.parts = parts

    class FakeModelResponse:
        def __init__(self, *, parts, model_name, timestamp) -> None:
            self.parts = parts
            self.model_name = model_name
            self.timestamp = timestamp

    class FakeTextPart:
        def __init__(self, *, content) -> None:
            self.content = content

    class FakeUserPromptPart:
        def __init__(self, *, content) -> None:
            self.content = content

    class FakeAgent:
        def __init__(self, model, **kwargs) -> None:
            pass

        async def run(self, prompt, *, message_history):
            captured["prompt"] = prompt
            captured["message_history"] = message_history
            return SimpleNamespace(output="All good.")

    with (
        patch("pawn_agent.core.burr_chat._import_pydantic_agent_cls", return_value=FakeAgent),
        patch(
            "pawn_agent.core.burr_chat._import_pydantic_messages",
            return_value=(FakeModelRequest, FakeModelResponse, FakeTextPart, FakeUserPromptPart),
        ),
    ):
        agent = PlainPydanticChatAgent(cfg)
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
    assert captured["prompt"] == "how are you?"
    assert len(captured["message_history"]) == 2
    assert isinstance(captured["message_history"][0], FakeModelRequest)
    assert isinstance(captured["message_history"][1], FakeModelResponse)


def test_make_burr_tracker_uses_project_and_storage_dir(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "burr:\n"
        "  project: pawn-eval\n"
        "  backend: local\n"
        "  storage_dir: /tmp/custom-burr\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))

    captured: dict[str, object] = {}

    class FakeTracker:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    with patch("pawn_agent.core.burr_chat._import_burr_tracker_cls", return_value=FakeTracker):
        tracker = make_burr_tracker(cfg)

    assert isinstance(tracker, FakeTracker)
    assert captured == {"project": "pawn-eval", "storage_dir": "/tmp/custom-burr"}


def test_parse_postgres_dsn_supports_sqlalchemy_style_driver() -> None:
    parsed = parse_postgres_dsn("postgresql+psycopg://user:pass@db.example:5433/pawnai")

    assert parsed == {
        "db_name": "pawnai",
        "user": "user",
        "password": "pass",
        "host": "db.example",
        "port": 5433,
    }


def test_make_burr_postgres_persister_uses_db_and_table_name(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "db_dsn: postgresql+psycopg://user:pass@localhost:5433/pawnai\n"
        "burr:\n"
        "  backend: postgres\n"
        "  table_name: pawn_burr_state\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))

    captured: dict[str, object] = {}

    class FakePersister:
        def __init__(self) -> None:
            self.initialized = False

        @classmethod
        async def from_values(cls, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            instance = cls()
            captured["instance"] = instance
            return instance

        async def initialize(self) -> None:
            self.initialized = True

    with patch(
        "pawn_agent.core.burr_chat._import_burr_postgres_persister_cls",
        return_value=FakePersister,
    ):
        persister = asyncio.run(make_burr_postgres_persister(cfg))

    assert persister is captured["instance"]
    assert persister.initialized is True
    assert captured["args"] == ("pawnai", "user", "pass", "localhost", 5433)
    assert captured["kwargs"] == {"table_name": "pawn_burr_state"}


def test_burr_chat_session_reset_rebuilds_application(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "burr:\n"
        "  backend: local\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))

    built_apps: list[object] = []
    app_ids: list[str] = []

    partition_keys: list[str] = []

    async def fake_build_app(cfg, chat_agent, *, app_id, partition_key, tracker, persister):
        built_apps.append(object())
        app_ids.append(app_id)
        partition_keys.append(partition_key)
        return built_apps[-1]

    with (
        patch("pawn_agent.core.burr_chat.make_burr_tracker", return_value=object()),
        patch("pawn_agent.core.burr_chat.PlainPydanticChatAgent", return_value=object()),
        patch("pawn_agent.core.burr_chat.build_burr_chat_application", side_effect=fake_build_app),
    ):
        session = BurrChatSession(cfg=cfg, emit=lambda _text: None)
        asyncio.run(session.reset())
        first_app = session._app
        first_app_id = session._app_id
        asyncio.run(session.reset())

    assert first_app is not None
    assert session._app is built_apps[-1]
    assert session._app is not first_app
    assert session._app_id != first_app_id
    assert len(app_ids) == 2
    assert partition_keys == ["chat", "chat"]


def test_burr_chat_session_writes_local_tracking_files(tmp_path: Path) -> None:
    storage_dir = tmp_path / "burr-storage"
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "burr:\n"
        "  project: pawn-agent\n"
        "  backend: local\n"
        f"  storage_dir: {storage_dir}\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))

    class FakeChatAgent:
        def __init__(self, cfg=None, on_thinking=None) -> None:
            pass

        async def reply(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            return f"echo:{user_prompt}"

    with patch("pawn_agent.core.burr_chat.PlainPydanticChatAgent", FakeChatAgent):
        session = asyncio.run(BurrChatSession.create(cfg=cfg, emit=lambda _text: None))
        app_id = session._app_id
        reply = asyncio.run(session.handle_user_input("hello"))
        asyncio.run(session.close())

    assert reply == "echo:hello"
    app_dir = storage_dir / cfg.burr.project / app_id
    assert app_dir.exists()
    assert (app_dir / "metadata.json").exists()
    assert (app_dir / "graph.json").exists()
    log_file = app_dir / "log.jsonl"
    assert log_file.exists()
    log_text = log_file.read_text(encoding="utf-8")
    assert "human_input" in log_text
    assert "ai_response" in log_text


def test_burr_chat_session_can_save_graph_image(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "burr:\n"
        "  backend: local\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))

    class FakeApp:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def visualize(self, **kwargs) -> None:
            self.calls.append(kwargs)

    with (
        patch("pawn_agent.core.burr_chat.make_burr_tracker", return_value=object()),
        patch("pawn_agent.core.burr_chat.PlainPydanticChatAgent", return_value=object()),
        patch("pawn_agent.core.burr_chat.build_burr_chat_application", return_value=FakeApp()),
    ):
        session = BurrChatSession(cfg=cfg, emit=lambda _text: None)
        asyncio.run(session.reset())

    output_path = tmp_path / "graphs" / "chat.png"
    session.save_graph_image(str(output_path))

    assert output_path.parent.exists()
    assert session._app.calls == [
        {
            "output_file_path": str(output_path.with_suffix("")),
            "include_state": True,
            "format": "png",
        }
    ]

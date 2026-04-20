from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pawn_agent.core.chat_primitives import (
    PlainPydanticChatAgent,
    apply_assistant_message,
    apply_user_message,
)
from pawn_agent.utils.config import load_config


def test_chat_message_helpers_append_messages_in_order() -> None:
    state = {
        "chat_history": [],
        "latest_user_message": "",
        "latest_assistant_message": "",
        "turn_count": 0,
    }
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


def test_plain_pydantic_chat_agent_uses_toolless_agent_and_anima(tmp_path: Path) -> None:
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

    class FakeAgent:
        def __init__(self, model, **kwargs) -> None:
            captured["model"] = model
            captured["kwargs"] = kwargs

        async def run(self, *args, **kwargs):
            return SimpleNamespace(output="ok")

    with patch(
        "pawn_agent.core.chat_primitives._import_pydantic_agent_cls",
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
        patch("pawn_agent.core.chat_primitives._import_pydantic_agent_cls", return_value=FakeAgent),
        patch(
            "pawn_agent.core.chat_primitives._import_pydantic_messages",
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

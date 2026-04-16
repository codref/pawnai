from __future__ import annotations

from pawn_agent.core.pydantic_agent import (
    LISTEN_ONLY_BYPASS_MARKER,
    _coerce_tool_call_content,
    _make_listen_only_bypass_result,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart


def test_coerce_adds_empty_text_for_tool_call_only_response() -> None:
    msg = ModelResponse(parts=[ToolCallPart(tool_name="session_vars", args={"key": "listen_only"})])

    out = _coerce_tool_call_content([msg])
    coerced = out[0]

    assert isinstance(coerced.parts[0], TextPart)
    assert coerced.parts[0].content == ""
    assert isinstance(coerced.parts[1], ToolCallPart)


def test_coerce_adds_empty_text_for_empty_response_parts() -> None:
    # Can happen after strip_thinking removes a thinking-only response.
    msg = ModelResponse(parts=[])

    out = _coerce_tool_call_content([msg])
    coerced = out[0]

    assert len(coerced.parts) == 1
    assert isinstance(coerced.parts[0], TextPart)
    assert coerced.parts[0].content == ""


def test_coerce_does_not_modify_response_with_text() -> None:
    msg = ModelResponse(parts=[TextPart(content="hello")])

    out = _coerce_tool_call_content([msg])
    coerced = out[0]

    assert len(coerced.parts) == 1
    assert isinstance(coerced.parts[0], TextPart)
    assert coerced.parts[0].content == "hello"


def test_listen_only_bypass_result_has_fixed_marker() -> None:
    result = _make_listen_only_bypass_result(
        text="thinking out loud",
        model_name="openai:gpt-oss:20b",
        message_history=[],
    )

    assert result.output == LISTEN_ONLY_BYPASS_MARKER
    new_messages = result.new_messages()
    assert len(new_messages) == 2
    assert isinstance(new_messages[0], ModelRequest)
    assert isinstance(new_messages[0].parts[0], UserPromptPart)
    assert new_messages[0].parts[0].content == "thinking out loud"
    assert isinstance(new_messages[1], ModelResponse)
    assert isinstance(new_messages[1].parts[0], TextPart)
    assert new_messages[1].parts[0].content == "[listening]"


def test_listen_only_bypass_result_extends_existing_history() -> None:
    prior = ModelResponse(parts=[TextPart(content="already here")])
    result = _make_listen_only_bypass_result(
        text="next overheard line",
        model_name="openai:gpt-oss:20b",
        message_history=[prior],
    )

    all_messages = result.all_messages()
    assert len(all_messages) == 3
    assert all_messages[0] is prior

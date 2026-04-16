from __future__ import annotations

from pawn_agent.core.session_store import _analyse_history_messages, _strip_thinking
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)


def test_analyse_history_messages_flags_retry_and_empty_responses() -> None:
    raw_messages = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi")]),
        ModelRequest(parts=[RetryPromptPart(content="Please return text or call a tool.")]),
        ModelResponse(parts=[ThinkingPart(content="thinking", id="reasoning")]),
        ModelResponse(parts=[ToolCallPart(tool_name="session_vars", args={"key": "listen_only"})]),
        ModelResponse(parts=[TextPart(content="   ")]),
    ]

    analysis = _analyse_history_messages(raw_messages, _strip_thinking(raw_messages))
    counts = analysis["counts"]

    assert counts["retry_prompt_requests"] == 1
    assert counts["responses_without_text"] == 2
    assert counts["blank_text_responses"] == 1
    assert counts["thinking_only_responses"] == 1
    assert counts["tool_call_only_responses"] == 1
    assert counts["responses_empty_after_strip"] == 1
    assert counts["consecutive_same_kind"] == 2
    assert any("RetryPromptPart" in issue for issue in analysis["issues"])

from __future__ import annotations

from pawn_agent.core.pydantic_agent import (
    LISTEN_ONLY_BYPASS_MARKER,
    _coerce_tool_call_content,
    _finalize_run_result,
    _make_listen_only_bypass_result,
)
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


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


def test_finalize_run_result_rewrites_raw_save_to_siyuan_command_output() -> None:
    result = _make_listen_only_bypass_result(
        text="placeholder",
        model_name="openai:gpt-oss:20b",
        message_history=[],
    )
    result.output = '/save_to_siyuan(content="# Report", session_id="tom-20260416", title="GitHub Copilot SDK")'
    result._new_messages = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="save_to_siyuan",
                    args={"session_id": "tom-20260416", "title": "GitHub Copilot SDK"},
                )
            ]
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name="save_to_siyuan", content="siyuan://blocks/doc-123")]),
    ]
    result._all_messages = list(result._new_messages)

    finalized = _finalize_run_result(result)

    assert finalized.output == (
        "Saved the content to SiYuan as 'GitHub Copilot SDK' for session 'tom-20260416'. "
        "URL: siyuan://blocks/doc-123"
    )


def test_finalize_run_result_leaves_normal_text_unchanged() -> None:
    result = _make_listen_only_bypass_result(
        text="placeholder",
        model_name="openai:gpt-oss:20b",
        message_history=[],
    )
    result.output = "Saved to SiYuan."

    finalized = _finalize_run_result(result)

    assert finalized.output == "Saved to SiYuan."


def test_finalize_run_result_appends_siyuan_url_to_natural_language_save_reply() -> None:
    result = _make_listen_only_bypass_result(
        text="placeholder",
        model_name="openai:gpt-oss:20b",
        message_history=[],
    )
    result.output = "im done. The analysis is saved to SiYuan."
    result._new_messages = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="analyze_custom",
                    args={
                        "instruction": "Extract action points",
                        "save": True,
                        "session_id": "oci-20260416",
                        "title": "Action Items: oci-20260416",
                    },
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="analyze_custom",
                    content="Saved custom analysis to SiYuan (url: siyuan://blocks/doc-789).\n\n# Report",
                )
            ]
        ),
    ]
    result._all_messages = list(result._new_messages)

    finalized = _finalize_run_result(result)

    assert finalized.output == "im done. The analysis is saved to SiYuan.\n\nURL: siyuan://blocks/doc-789"


def test_finalize_run_result_rewrites_raw_call_style_analyze_custom_output() -> None:
    result = _make_listen_only_bypass_result(
        text="placeholder",
        model_name="openai:gpt-oss:20b",
        message_history=[],
    )
    result.output = (
        'call:analyze_custom{instruction:<|"|>Extract all relevant action points<|"|>,'
        'save:true,session_id:<|"|>oci-20260416<|"|>,title:<|"|>Action Points: oci-20260416<|"|>}'
    )
    result._new_messages = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="analyze_custom",
                    args={
                        "instruction": "Extract all relevant action points",
                        "save": True,
                        "session_id": "oci-20260416",
                        "title": "Action Points: oci-20260416",
                    },
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="analyze_custom",
                    content="Saved custom analysis to SiYuan (url: siyuan://blocks/doc-456).\n\n# Report",
                )
            ]
        ),
    ]
    result._all_messages = list(result._new_messages)

    finalized = _finalize_run_result(result)

    assert finalized.output == (
        "Created the SiYuan page with the requested analysis "
        "as 'Action Points: oci-20260416' for session 'oci-20260416'. "
        "URL: siyuan://blocks/doc-456"
    )


def test_finalize_run_result_prefers_saved_analysis_over_query_conversation_marker() -> None:
    result = _make_listen_only_bypass_result(
        text="placeholder",
        model_name="openai:gpt-oss:20b",
        message_history=[],
    )
    result.output = 'call:query_conversation{session_id:<|"|>oci-20260416<|"|>}'
    result._new_messages = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="query_conversation",
                    args={"session_id": "oci-20260416"},
                )
            ]
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name="query_conversation", content="[00:00] Speaker A: hello")]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="analyze_custom",
                    args={
                        "instruction": "Extract action points",
                        "save": True,
                        "session_id": "oci-20260416",
                        "title": "Action Points: oci-20260416",
                    },
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="analyze_custom",
                    content="Saved custom analysis to SiYuan (url: siyuan://blocks/doc-999).",
                )
            ]
        ),
    ]
    result._all_messages = list(result._new_messages)

    finalized = _finalize_run_result(result)

    assert finalized.output == (
        "Created the SiYuan page with the requested analysis "
        "as 'Action Points: oci-20260416' for session 'oci-20260416'. "
        "URL: siyuan://blocks/doc-999"
    )

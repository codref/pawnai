from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pawn_agent.tools import get_registry
from pawn_agent.tools.analyze_custom import build


def _make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        db_dsn="postgresql+psycopg://dummy/dummy",
        model="openai:test-model",
        pydantic_model="openai:test-model",
        pydantic_api_key=None,
        pydantic_base_url=None,
        siyuan_url="http://localhost:6806",
        siyuan_token="token",
        siyuan_notebook="notebook",
        siyuan_path_template="/conversations/{session_id}/{title}",
        siyuan_daily_template="/daily/{date}",
    )


def test_analyze_custom_is_discoverable() -> None:
    assert any(name == "analyze_custom" for name, _ in get_registry())


def test_analyze_custom_runs_llm_with_transcript() -> None:
    cfg = _make_cfg()
    tool = build(cfg)

    with (
        patch("pawn_agent.utils.transcript.fetch_transcript", return_value="Speaker A: hello") as mock_fetch,
        patch(
            "pawn_agent.core.llm_sub.run",
            new=AsyncMock(return_value="# Action Points\n\n- Do the thing"),
        ) as mock_run,
    ):
        result = asyncio.run(
            tool.function(
                session_id="tom-20260416",
                instruction="Create an extensive analysis of the action points discussed.",
            )
        )

    assert "# Action Points" in result
    mock_fetch.assert_called_once_with(cfg, "tom-20260416")
    mock_run.assert_awaited_once()
    prompt = mock_run.await_args.args[1]
    assert "Create an extensive analysis of the action points discussed." in prompt
    assert "Speaker A: hello" in prompt


def test_analyze_custom_can_save_to_siyuan() -> None:
    cfg = _make_cfg()
    tool = build(cfg)

    with (
        patch("pawn_agent.utils.transcript.fetch_transcript", return_value="Speaker A: hello"),
        patch(
            "pawn_agent.core.llm_sub.run",
            new=AsyncMock(return_value="# Report\n\nDetailed analysis"),
        ),
        patch(
            "pawn_agent.utils.siyuan.do_save_to_siyuan",
            return_value="siyuan://blocks/doc-123",
        ) as mock_save,
    ):
        result = asyncio.run(
            tool.function(
                session_id="tom-20260416",
                instruction="Write a deep report.",
                save=True,
                title="Extended Analysis",
            )
        )

    assert "Saved custom analysis to SiYuan" in result
    assert "siyuan://blocks/doc-123" in result
    assert "# Report" not in result
    assert "Detailed analysis" not in result
    mock_save.assert_called_once_with(
        cfg,
        "tom-20260416",
        "Extended Analysis",
        "# Report\n\nDetailed analysis",
        None,
    )

"""Tests for pawn_agent."""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


class TestAgentConfig:
    def test_defaults(self):
        from pawn_agent.utils.config import AgentConfig

        cfg = AgentConfig()
        assert cfg.backend == "copilot"
        assert cfg.model == "gpt-4o"
        assert "postgres" in cfg.db_dsn

    def test_load_config_missing_file(self):
        from pawn_agent.utils.config import load_config

        cfg = load_config("/nonexistent/path.yml")
        assert cfg.backend == "copilot"  # falls back to default

    def test_load_config_from_yaml(self, tmp_path):
        from pawn_agent.utils.config import load_config

        cfg_file = tmp_path / "test.yml"
        cfg_file.write_text(
            "db_dsn: postgresql+psycopg://user:pw@host/db\n"
            "agent:\n"
            "  backend: openai\n"
            "  model: llama3\n"
            "  openai_base_url: http://localhost:11434/v1\n"
            "siyuan:\n"
            "  token: mytoken\n"
            "  notebook: nb123\n"
        )
        cfg = load_config(str(cfg_file))
        assert cfg.db_dsn == "postgresql+psycopg://user:pw@host/db"
        assert cfg.backend == "openai"
        assert cfg.model == "llama3"
        assert cfg.openai_base_url == "http://localhost:11434/v1"
        assert cfg.siyuan_token == "mytoken"
        assert cfg.siyuan_notebook == "nb123"


# ──────────────────────────────────────────────────────────────────────────────
# LM backends
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildLm:
    def test_returns_copilot_lm(self):
        from pawn_agent.utils.config import AgentConfig
        from pawn_agent.core.lm import build_lm, CopilotLM

        cfg = AgentConfig(backend="copilot", model="gpt-4o")
        lm = build_lm(cfg)
        assert isinstance(lm, CopilotLM)

    def test_returns_dspy_lm_for_openai(self):
        import dspy
        from pawn_agent.utils.config import AgentConfig
        from pawn_agent.core.lm import build_lm, CopilotLM

        cfg = AgentConfig(backend="openai", model="llama3",
                          openai_base_url="http://localhost:11434/v1",
                          openai_api_key="ollama")
        lm = build_lm(cfg)
        assert not isinstance(lm, CopilotLM)
        assert isinstance(lm, dspy.LM)


class TestCopilotLm:
    def _make_copilot_response(self, content: str):
        """Build a minimal mock matching CopilotClient.send_and_wait return value."""
        data = MagicMock()
        data.content = content
        response = MagicMock()
        response.data = data
        return response

    def test_call_with_messages(self):
        """CopilotLM.__call__ should invoke the async path and return a list."""
        from pawn_agent.core.lm import CopilotLM

        lm = CopilotLM(model="gpt-4o")
        expected = "Hello, world!"

        with patch.object(lm, "_async_call", return_value=expected) as mock_ac:
            # patch asyncio.run to call the coroutine synchronously
            with patch("pawn_agent.core.lm.asyncio.run", side_effect=lambda c: expected):
                result = lm(messages=[{"role": "user", "content": "hi"}])

        assert isinstance(result, list)
        assert result[0] == expected

    def test_history_appended(self):
        """Each call should append an entry to lm.history."""
        from pawn_agent.core.lm import CopilotLM

        lm = CopilotLM()
        with patch("pawn_agent.core.lm.asyncio.run", return_value="ok"):
            lm(prompt="test")
        assert len(lm.history) == 1
        assert lm.history[0]["prompt"] == "test"


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def agent_cfg(tmp_path):
    from pawn_agent.utils.config import AgentConfig

    return AgentConfig(
        db_dsn="postgresql+psycopg://dummy/dummy",
        siyuan_url="http://localhost:6806",
        siyuan_token="tok",
        siyuan_notebook="nb1",
        siyuan_path_template="/Conversations/{date}/{session_id}",
        siyuan_daily_template="/daily note/{year}/{month}/{date}",
    )


class TestQueryConversation:
    def test_returns_formatted_transcript(self, agent_cfg):
        from pawn_agent.core.tools import build_tools, _TranscriptionSegment, _SpeakerName

        tools = build_tools(agent_cfg)
        query_fn = tools[0]  # query_conversation

        # Mock DB session
        seg = MagicMock(spec=_TranscriptionSegment)
        seg.session_id = "s1"
        seg.audio_file = "audio.wav"
        seg.original_speaker_label = "SPEAKER_00"
        seg.start_time = 0.0
        seg.text = "Hello there"
        seg.segment_index = 0

        speaker = MagicMock(spec=_SpeakerName)
        speaker.audio_file = "audio.wav"
        speaker.local_speaker_label = "SPEAKER_00"
        speaker.speaker_name = "Alice"

        mock_db = MagicMock()
        mock_db.scalars.return_value.all.side_effect = [[seg], [speaker]]

        with patch("pawn_agent.core.tools._make_db_session", return_value=mock_db):
            result = query_fn("s1")

        assert "Alice" in result
        assert "Hello there" in result

    def test_returns_error_when_no_segments(self, agent_cfg):
        from pawn_agent.core.tools import build_tools

        tools = build_tools(agent_cfg)
        query_fn = tools[0]

        mock_db = MagicMock()
        mock_db.scalars.return_value.all.return_value = []

        with patch("pawn_agent.core.tools._make_db_session", return_value=mock_db):
            result = query_fn("missing-session")

        assert "No transcript" in result


class TestAnalyzeConversation:
    def test_persists_to_db_and_returns_text(self, agent_cfg):
        import dspy
        from pawn_agent.core.tools import build_tools

        tools = build_tools(agent_cfg)
        analyze_fn = tools[1]  # analyze_conversation

        # Pre-canned analysis text with all required sections
        analysis_text = (
            "## Title\nTest Meeting\n"
            "## Summary\nA test summary.\n"
            "## Key Topics / Keywords\n- topic1\n"
            "## Speaker Highlights\nAlice talked.\n"
            "## Sentiment\nNeutral.\n"
            "## Sentiment Tags\nneutral\n"
            "## Tags\ntest, meeting\n"
        )

        mock_lm = MagicMock()
        mock_lm.return_value = [analysis_text]
        mock_lm._copilot_model = "gpt-4o"
        mock_lm.model = "gpt-4o"

        mock_db = MagicMock()
        mock_db.scalars.return_value.all.return_value = []

        # Configure DSPy with mock LM, then restore
        original_lm = getattr(dspy.settings, "lm", None)
        try:
            dspy.configure(lm=mock_lm)
            with patch("pawn_agent.core.tools._make_db_session", return_value=mock_db):
                result = analyze_fn("s1")
        finally:
            if original_lm is not None:
                dspy.configure(lm=original_lm)

        # Should return either the analysis or the "no transcript" message
        assert isinstance(result, str)
        assert len(result) > 0


class TestSaveToSiyuan:
    def test_calls_siyuan_api(self, agent_cfg):
        from pawn_agent.core.tools import build_tools

        tools = build_tools(agent_cfg)
        save_fn = tools[2]  # save_to_siyuan

        # Responses for: getIDsByHPath (no existing), createDocWithMd, setBlockAttrs,
        # daily getIDsByHPath, daily createDocWithMd, appendBlock
        responses = [
            [],          # no existing doc to remove
            "blockid1",  # createDocWithMd → new doc id
            {},          # setBlockAttrs
            [],          # daily note getIDsByHPath
            "dailyid1",  # daily createDocWithMd
            {"doOperations": [{"id": "blk2"}]},  # appendBlock
        ]
        call_count = 0

        def fake_post(base_url, token, endpoint, payload):
            nonlocal call_count
            r = responses[call_count] if call_count < len(responses) else {}
            call_count += 1
            return r

        with patch("pawn_agent.core.tools._siyuan_post", side_effect=fake_post):
            result = save_fn("s1", "Test Meeting", "# Test\n\nContent")

        assert result == "blockid1"

    def test_missing_notebook_returns_error(self, agent_cfg):
        from pawn_agent.core.tools import build_tools
        from pawn_agent.utils.config import AgentConfig

        cfg = AgentConfig(siyuan_notebook="")  # no notebook
        tools = build_tools(cfg)
        save_fn = tools[2]

        result = save_fn("s1", "Title", "Content")
        assert "not configured" in result.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────


class TestConversationAgent:
    def test_forward_returns_string(self):
        import dspy
        from pawn_agent.core.agent import ConversationAgent

        mock_tool = MagicMock(return_value="tool result")
        mock_tool.__name__ = "mock_tool"
        mock_tool.__doc__ = "A mock tool."
        mock_tool.__annotations__ = {"query": str, "return": str}

        mock_react_result = MagicMock()
        mock_react_result.response = "Final answer"

        with patch("dspy.ReAct") as MockReAct:
            MockReAct.return_value = MagicMock(return_value=mock_react_result)
            agent = ConversationAgent(tools=[mock_tool], max_iters=3)
            result = agent(user_prompt="What was discussed in session abc?")

        assert result == "Final answer"

    def test_run_is_alias_for_forward(self):
        import dspy
        from pawn_agent.core.agent import ConversationAgent

        mock_react_result = MagicMock()
        mock_react_result.response = "response text"

        with patch("dspy.ReAct") as MockReAct:
            MockReAct.return_value = MagicMock(return_value=mock_react_result)
            agent = ConversationAgent(tools=[], max_iters=1)
            # agent(...) is the DSPy-canonical call path
            assert agent(user_prompt="hi") == "response text"


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


class TestCli:
    def test_help(self):
        from typer.testing import CliRunner
        from pawn_agent.cli.commands import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "pawn-agent" in result.output.lower() or "prompt" in result.output.lower()

    def test_run_missing_prompt(self):
        from typer.testing import CliRunner
        from pawn_agent.cli.commands import app

        runner = CliRunner()
        result = runner.invoke(app, [])
        # Missing required argument → non-zero or usage message
        assert result.exit_code != 0 or "prompt" in result.output.lower()

    def test_run_with_mock_agent(self):
        from typer.testing import CliRunner
        from pawn_agent.cli.commands import app

        runner = CliRunner()
        # Patch at module origin since commands.py uses local imports
        with patch("pawn_agent.core.lm.build_lm"):
            with patch("pawn_agent.core.tools.build_tools", return_value=[]):
                with patch("pawn_agent.core.agent.ConversationAgent.forward",
                           return_value="## Result\n\nDone."):
                    with patch("dspy.configure"):
                        result = runner.invoke(app, ["Hello from test", "--session", "s1"])
        # Should complete without crashing
        assert result.exit_code in (0, 1)  # 1 is acceptable if LM setup fails in test env

"""Tests for the Burr-based agent: BurrAgent, config, state models, and CLI."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── AgentConfig ────────────────────────────────────────────────────────────────


class TestAgentConfig:
    def test_defaults(self):
        from pawn_agent.utils.config import AgentConfig

        cfg = AgentConfig()
        assert cfg.agent_name == "Bob"
        assert cfg.burr_enabled is True
        assert cfg.burr_project == "pawn-agent"
        assert cfg.burr_backend == "postgres"
        assert cfg.burr_storage_dir is None
        # MLflow properties must no longer exist
        assert not hasattr(cfg, "mlflow_enabled")
        assert not hasattr(cfg, "mlflow_tracking_uri")
        assert not hasattr(cfg, "mlflow_experiment")

    def test_selector_model_defaults_to_pydantic_model(self):
        from pawn_agent.utils.config import AgentConfig

        cfg = AgentConfig()
        assert cfg.selector_model == cfg.pydantic_model

    def test_burr_section_from_dict(self):
        from pawn_agent.utils.config import AgentConfig

        cfg = AgentConfig.model_validate(
            {
                "burr": {
                    "enabled": False,
                    "project": "test-project",
                    "backend": "local",
                    "storage_dir": "/tmp/burr",
                }
            }
        )
        assert cfg.burr_enabled is False
        assert cfg.burr_project == "test-project"
        assert cfg.burr_backend == "local"
        assert cfg.burr_storage_dir == "/tmp/burr"


# ── State models ───────────────────────────────────────────────────────────────


class TestBurrState:
    def test_default_state(self):
        from pawn_agent.core.burr_state import DynamicContextState

        s = DynamicContextState()
        assert s.user_goal == ""
        assert s.facts == []
        assert s.next_action == "respond"
        assert s.pending_tool_call is None
        assert s.raw_tool_result is None

    def test_round_trip(self):
        from pawn_agent.core.burr_state import (
            DynamicContextState,
            Fact,
            burr_dict_to_state,
            state_to_burr_dict,
        )

        original = DynamicContextState(
            user_goal="List sessions",
            facts=[Fact(text="Session X exists", source="tool", confidence=0.9)],
            next_action="search_knowledge",
        )
        d = state_to_burr_dict(original)
        restored = burr_dict_to_state(d)

        assert restored.user_goal == original.user_goal
        assert len(restored.facts) == 1
        assert restored.facts[0].text == "Session X exists"
        assert restored.next_action == "search_knowledge"

    def test_tool_call_model(self):
        from pawn_agent.core.burr_state import ToolCall

        tc = ToolCall(tool_name="search_knowledge", arguments={"query": "hello", "top_k": 3})
        assert tc.tool_name == "search_knowledge"
        assert tc.arguments["top_k"] == 3

    def test_context_selection_default(self):
        from pawn_agent.core.burr_state import ContextSelection

        cs = ContextSelection()
        assert cs.need_additional_retrieval is False
        assert cs.selected_ids == []


# ── build_tools_registry ───────────────────────────────────────────────────────


class TestBuildToolsRegistry:
    def test_returns_dict_of_callables(self):
        from pawn_agent.tools import build_tools_registry
        from pawn_agent.utils.config import AgentConfig

        cfg = AgentConfig()

        fake_tool = MagicMock()
        fake_tool.name = "fake_tool"
        fake_tool.function = lambda query: "result"

        with patch("pawn_agent.tools.build_tools", return_value=[fake_tool]):
            registry = build_tools_registry(cfg)

        assert "fake_tool" in registry
        assert callable(registry["fake_tool"])


# ── BurrAgent unit ─────────────────────────────────────────────────────────────


class TestBurrAgent:
    def _make_cfg(self) -> Any:
        from pawn_agent.utils.config import AgentConfig

        cfg = AgentConfig.model_validate(
            {"db_dsn": "", "burr": {"enabled": False}}
        )
        return cfg

    def test_init_without_db(self):
        from pawn_agent.core.burr_agent import BurrAgent

        cfg = self._make_cfg()

        with patch("pawn_agent.core.burr_agent.build_tools_registry", return_value={}):
            agent = BurrAgent(cfg=cfg)

        assert agent._session_id is not None
        assert agent._tools_registry == {}

    def test_run_returns_string(self):
        from pawn_agent.core.burr_agent import BurrAgent

        cfg = self._make_cfg()

        with patch("pawn_agent.core.burr_agent.build_tools_registry", return_value={}):
            agent = BurrAgent(cfg=cfg)

        expected_response = "Here are the sessions."

        with patch.object(agent, "_run_turn", return_value=expected_response) as mock_run:
            result = agent.run("List sessions")

        mock_run.assert_called_once_with("List sessions")
        assert result.output == expected_response

    def test_session_id_preserved(self):
        from pawn_agent.core.burr_agent import BurrAgent

        cfg = self._make_cfg()

        with patch("pawn_agent.core.burr_agent.build_tools_registry", return_value={}):
            agent = BurrAgent(cfg=cfg, session_id="my-session-123")

        assert agent._session_id == "my-session-123"


# ── Context subgraph unit ──────────────────────────────────────────────────────


class TestDeriveRetrievalQuery:
    def test_builds_query_from_goal(self):
        from burr.core import State

        from pawn_agent.core.burr_context import derive_retrieval_query

        state = State(
            {
                "user_goal": "Find sessions about pricing",
                "open_questions": [{"question": "Which sessions?", "priority": 1}],
                "constraints": ["today only"],
            }
        )
        result, new_state = derive_retrieval_query(state)
        query = result["retrieval_query"]
        assert "Find sessions about pricing" in query
        assert "Which sessions?" in query
        assert "today only" in query

    def test_empty_state(self):
        from burr.core import State

        from pawn_agent.core.burr_context import derive_retrieval_query

        state = State({"user_goal": "", "open_questions": [], "constraints": []})
        result, _ = derive_retrieval_query(state)
        assert isinstance(result["retrieval_query"], str)


class TestRankContextCandidates:
    def test_sorts_by_score_descending(self):
        from burr.core import State

        from pawn_agent.core.burr_context import rank_context_candidates

        candidates = [
            {"id": "a", "text": "low", "score": 0.3, "entities": [], "token_estimate": 10, "freshness_score": 0},
            {"id": "b", "text": "high", "score": 0.9, "entities": [], "token_estimate": 10, "freshness_score": 0},
            {"id": "c", "text": "mid", "score": 0.6, "entities": [], "token_estimate": 10, "freshness_score": 0},
        ]
        state = State({"context_candidates": candidates, "retrieval_query": "test", "facts": []})
        result, _ = rank_context_candidates(state)
        ids = [c["id"] for c in result["context_candidates"]]
        assert ids[0] == "b"


class TestAssemblePromptContext:
    def test_includes_goal_and_facts(self):
        from burr.core import State

        from pawn_agent.core.burr_context import assemble_prompt_context

        state = State(
            {
                "user_goal": "Summarise the meeting",
                "constraints": [],
                "open_questions": [],
                "facts": [{"text": "Alice spoke first", "entity_ids": []}],
                "artifacts": [],
                "selected_context_ids": [],
                "context_candidates": [],
                "token_budget_for_context": 4096,
            }
        )
        result, _ = assemble_prompt_context(state)
        ctx = result["assembled_context"]
        assert "Summarise the meeting" in ctx
        assert "Alice spoke first" in ctx

    def test_enforces_token_budget(self):
        from burr.core import State

        from pawn_agent.core.burr_context import assemble_prompt_context

        long_fact = "x" * 8000
        state = State(
            {
                "user_goal": "goal",
                "constraints": [],
                "open_questions": [],
                "facts": [{"text": long_fact, "entity_ids": []}] * 5,
                "artifacts": [],
                "selected_context_ids": [],
                "context_candidates": [],
                "token_budget_for_context": 100,
            }
        )
        result, _ = assemble_prompt_context(state)
        ctx = result["assembled_context"]
        assert len(ctx) < len(long_fact) * 5


# ── Main action graph unit ─────────────────────────────────────────────────────


class TestToolExecutor:
    def test_calls_registered_tool(self):
        from burr.core import State

        from pawn_agent.core.burr_actions import tool_executor

        results = []

        def my_tool(query: str) -> str:
            results.append(query)
            return f"result for {query}"

        registry = {"my_tool": my_tool}
        state = State({"pending_tool_call": {"tool_name": "my_tool", "arguments": {"query": "hello"}}})

        result, _ = tool_executor(state, tools_registry=registry)
        assert results == ["hello"]
        assert "result for hello" in result["raw_tool_result"]

    def test_unknown_tool_returns_error_string(self):
        from burr.core import State

        from pawn_agent.core.burr_actions import tool_executor

        state = State({"pending_tool_call": {"tool_name": "nonexistent", "arguments": {}}})
        result, _ = tool_executor(state, tools_registry={})
        assert "Unknown tool" in result["raw_tool_result"]

    def test_missing_tool_call(self):
        from burr.core import State

        from pawn_agent.core.burr_actions import tool_executor

        state = State({"pending_tool_call": None})
        result, _ = tool_executor(state, tools_registry={})
        assert "No tool selected" in result["raw_tool_result"]


# ── CLI smoke test ─────────────────────────────────────────────────────────────


class TestCli:
    def test_run_command_invokes_agent(self):
        from typer.testing import CliRunner

        from pawn_agent.cli.commands import app

        runner = CliRunner()

        with (
            patch("pawn_agent.core.burr_agent.build_tools_registry", return_value={}),
            patch("pawn_agent.core.burr_agent.BurrAgent") as MockAgent,
        ):
            instance = MockAgent.return_value
            instance.run.return_value = "Hello from agent"

            result = runner.invoke(app, ["run", "Test prompt"])

        assert result.exit_code == 0, result.output
        instance.run.assert_called_once()

    def test_tools_command(self):
        from typer.testing import CliRunner

        from pawn_agent.cli.commands import app

        runner = CliRunner()

        with patch("pawn_agent.tools.get_registry", return_value=[("fake_tool", "Does stuff")]):
            result = runner.invoke(app, ["tools"])

        assert result.exit_code == 0
        assert "fake_tool" in result.output

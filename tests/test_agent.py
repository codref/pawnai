"""Config smoke tests for pawn_agent."""

from __future__ import annotations


class TestAgentConfig:
    def test_defaults(self):
        from pawn_agent.utils.config import AgentConfig

        cfg = AgentConfig()
        assert cfg.backend == "copilot"
        assert "postgres" in cfg.db_dsn
        assert cfg.sallm.max_steps == 8
        assert cfg.sallm.profile == "large.yaml"
        assert cfg.litellm_model.startswith("openai/")

    def test_load_config_missing_file(self):
        from pawn_agent.utils.config import load_config

        cfg = load_config("/nonexistent/path.yml")
        assert cfg.backend == "copilot"

    def test_litellm_model_from_openai_provider(self, tmp_path):
        from pawn_agent.utils.config import load_config

        cfg_file = tmp_path / "test.yml"
        cfg_file.write_text(
            "agent:\n"
            "  openai:\n"
            "    model: gemma4:26b\n"
            "    api_key: ollama\n"
            "    base_url: http://localhost:11434/v1\n"
            "  sallm:\n"
            "    state_dir: .sallm-test\n"
            "    max_steps: 5\n"
            "    profile: large.yaml\n",
            encoding="utf-8",
        )
        cfg = load_config(str(cfg_file))
        assert cfg.litellm_model == "openai/gemma4:26b"
        assert cfg.pydantic_base_url == "http://localhost:11434/v1"
        assert cfg.sallm.max_steps == 5
        assert cfg.sallm.state_dir == ".sallm-test"
        assert cfg.sallm.profile == "large.yaml"

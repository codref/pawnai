"""Tests for sallm CompiledProfile YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from sallm.models import ModelProfile

from pawn_agent.profiles import (
    bundled_profiles_dir,
    load_compiled_profile,
    load_profile_from_config,
    resolve_profile_path,
)


def test_openrouter_route_and_reasoning_override() -> None:
    from pawn_agent.utils.model_catalog import (
        ModelSelection,
        routed_litellm_model,
        think_override,
    )

    selection = ModelSelection(
        catalog_id="openrouter@deepseek/deepseek-v4.1-flash",
        litellm_model="openai/deepseek/deepseek-v4.1-flash",
        api_base="https://openrouter.ai/api/v1",
        api_key="sk-test",
        user_agent="pawn/1.0",
        profile=None,
        provider="openrouter",
        reasoning="low",
        route="nitro",
    )
    assert routed_litellm_model(selection) == "openai/deepseek/deepseek-v4.1-flash:nitro"
    assert think_override(selection) == "low"
    balanced = ModelSelection(
        catalog_id=selection.catalog_id,
        litellm_model=selection.litellm_model,
        api_base=selection.api_base,
        api_key=None,
        user_agent=None,
        profile=None,
        provider="openrouter",
        route="balanced",
    )
    assert routed_litellm_model(balanced) == selection.litellm_model
    assert think_override(balanced) is None


def test_openai_compatible_think_false_disables_reasoning() -> None:
    from sallm.llm import prepare_completion_kwargs

    kwargs = prepare_completion_kwargs(
        "openai/deepseek/deepseek-v4.1-flash",
        {"max_tokens": 1024, "think": False},
    )
    assert "think" not in kwargs
    assert kwargs["extra_body"]["reasoning"] == {"effort": "none", "enabled": False}


def test_ollama_think_flag_is_preserved() -> None:
    from sallm.llm import prepare_completion_kwargs

    kwargs = prepare_completion_kwargs("ollama/gemma4:26b", {"think": False})
    assert kwargs["think"] is False
    assert "extra_body" not in kwargs


def test_bundled_large_profile_is_10x_defaults() -> None:
    path = bundled_profiles_dir() / "large.yaml"
    compiled = load_compiled_profile(path)
    base = ModelProfile()
    assert compiled.budgets["max_output_tokens"] == base.max_output_tokens * 10
    assert compiled.budgets["prompt_budget"] == base.prompt_budget * 10
    assert compiled.budgets["recent_history_tokens"] == base.recent_history_tokens * 10
    overlay = compiled.apply_budgets(base)
    assert overlay.max_output_tokens == 10_240
    assert overlay.prompt_budget == 40_960


def test_deepseek_v4_1_flash_profile_fits_the_model() -> None:
    path = bundled_profiles_dir() / "deepseek-v4.1-flash.yaml"
    compiled = load_compiled_profile(path)
    assert compiled.target_model == "deepseek/deepseek-v4.1-flash"
    overlay = compiled.apply_budgets(ModelProfile())
    assert overlay.prompt_budget == 262_144
    assert overlay.prompt_budget < compiled.metadata["context_window"]
    assert overlay.max_output_tokens == 8_192
    assert overlay.max_output_tokens < compiled.metadata["max_output"]
    assert overlay.temperature == 0.4
    assert overlay.think == "low"
    assert "```run" in compiled.instructions["converse"]


def test_resolve_profile_path_bundled() -> None:
    path = resolve_profile_path("large.yaml")
    assert path is not None
    assert path.name == "large.yaml"
    assert path.is_file()


def test_resolve_profile_path_empty() -> None:
    assert resolve_profile_path(None) is None
    assert resolve_profile_path("") is None
    assert resolve_profile_path("   ") is None


def test_resolve_profile_path_missing() -> None:
    with pytest.raises(FileNotFoundError):
        resolve_profile_path("does-not-exist.yaml")


def test_load_profile_from_config_default_name() -> None:
    compiled = load_profile_from_config("large.yaml")
    assert compiled is not None
    assert compiled.budgets["max_output_tokens"] == 10_240


def test_load_json_profile(tmp_path: Path) -> None:
    p = tmp_path / "tiny.json"
    p.write_text(
        '{"target_model":"","instructions":{},"demonstrations":{},'
        '"budgets":{"max_output_tokens":128},"metadata":{}}',
        encoding="utf-8",
    )
    compiled = load_compiled_profile(p)
    assert compiled.budgets["max_output_tokens"] == 128


def test_agent_config_default_profile() -> None:
    from pawn_agent.utils.config import AgentConfig

    cfg = AgentConfig()
    assert cfg.sallm.profile == "large.yaml"

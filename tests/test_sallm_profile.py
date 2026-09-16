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

"""Catalog ids bind a provider, credentials, and a compiled profile."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest

from pawn_agent.core.coworker.slash import resolve_chat_message
from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.model_catalog import (
    ModelSelection,
    apply_model_selection,
    completion_headers,
    get_background_model,
    model_is_vision,
    normalize_openai_base_url,
    public_catalog,
    reset_background_model,
    set_background_model,
)

_PROFILE = """\
target_model: gemma4:4b
instructions: {}
demonstrations: {}
budgets: {}
metadata: {}
"""


def _write_profile(directory: Path, name: str = "gemma.yaml", body: str = _PROFILE) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(body, encoding="utf-8")


def _cfg(tmp_path: Path, **agent: object) -> AgentConfig:
    base = {
        "profiles_dir": str(tmp_path / "profiles"),
        "sallm": {"state_dir": str(tmp_path / "state"), "profile": "large.yaml"},
    }
    base.update(agent)
    return AgentConfig(agent=base)


def test_full_endpoint_url_is_trimmed_to_v1_root() -> None:
    assert (
        normalize_openai_base_url("https://opencode.ai/zen/v1/chat/completions")
        == "https://opencode.ai/zen/v1"
    )
    assert (
        normalize_openai_base_url("https://opencode.ai/zen/go/v1/responses/")
        == "https://opencode.ai/zen/go/v1"
    )
    assert normalize_openai_base_url("https://opencode.ai/zen/go/v1") == (
        "https://opencode.ai/zen/go/v1"
    )
    assert normalize_openai_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"


def test_opencode_calls_send_user_agent_and_session() -> None:
    selection = ModelSelection(
        catalog_id="opencode@glm-5.3-flash",
        litellm_model="openai/glm-5.3-flash",
        api_base="https://opencode.ai/zen/v1",
        api_key="sk-test",
        user_agent="pawn/1.0",
        profile=None,
        provider="opencode",
    )
    headers = completion_headers(selection, "note:Pawn/Today.md")
    assert headers["User-Agent"] == "pawn/1.0"
    assert headers["x-opencode-session"] == "note:Pawn/Today.md"

    local = ModelSelection(
        catalog_id="ollama@gemma4:4b",
        litellm_model="openai/gemma4:4b",
        api_base="http://localhost:11434/v1",
        api_key=None,
        user_agent="pawn/1.0",
        profile=None,
        provider="ollama",
    )
    assert "x-opencode-session" not in completion_headers(local, "cli")


def test_legacy_openai_block_becomes_one_catalog_entry(tmp_path: Path) -> None:
    cfg = AgentConfig(
        agent={
            "openai": {
                "model": "gemma4:26b",
                "api_key": "ollama",
                "base_url": "http://localhost:11434/v1",
            },
            "sallm": {"state_dir": str(tmp_path / "state"), "profile": "large.yaml"},
        }
    )
    selection = cfg.model_selection
    assert selection.catalog_id == "openai@gemma4:26b"
    assert selection.litellm_model == "openai/gemma4:26b"
    assert selection.api_base == "http://localhost:11434/v1"
    assert selection.api_key == "ollama"
    assert selection.profile
    assert selection.profile.endswith("large.yaml")


def test_provider_profiles_use_target_model(tmp_path: Path) -> None:
    _write_profile(tmp_path / "profiles")
    cfg = _cfg(
        tmp_path,
        default="ollama@gemma4:4b",
        providers={
            "ollama": {
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
                "models": [{"profile": "gemma.yaml"}],
            },
            "opencode": {
                "base_url": "https://opencode.example/v1",
                "api_key": "sk-test",
                "models": [{"model": "gpt-5.6-sol", "profile": "gemma.yaml"}],
            },
        },
    )
    assert cfg.chat_model_id == "ollama@gemma4:4b"
    assert cfg.litellm_model == "openai/gemma4:4b"
    assert cfg.model_selection.api_base == "http://127.0.0.1:11434/v1"

    clone = copy.copy(cfg)
    apply_model_selection(clone, "opencode@gpt-5.6-sol")
    assert cfg.chat_model_id == "ollama@gemma4:4b"
    chosen = clone.model_selection
    assert chosen.catalog_id == "opencode@gpt-5.6-sol"
    assert chosen.litellm_model == "openai/gpt-5.6-sol"
    assert chosen.api_base == "https://opencode.example/v1"
    assert chosen.api_key == "sk-test"


def test_vision_flag_is_public_and_off_by_default(tmp_path: Path) -> None:
    _write_profile(tmp_path / "profiles")
    cfg = _cfg(
        tmp_path,
        default="ollama@gemma4:4b",
        providers={
            "ollama": {
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
                "models": [
                    {"profile": "gemma.yaml"},
                    {"model": "gemma4:e4b", "profile": "gemma.yaml", "vision": True},
                ],
            }
        },
    )
    listed = {item["id"]: item for item in public_catalog(cfg)["models"]}
    assert listed["ollama@gemma4:4b"]["vision"] is False
    assert listed["ollama@gemma4:e4b"]["vision"] is True
    assert model_is_vision(cfg, "ollama@gemma4:e4b") is True
    assert model_is_vision(cfg, "ollama@gemma4:4b") is False
    assert "api_key" not in listed["ollama@gemma4:e4b"]


def test_duplicate_catalog_ids_fail(tmp_path: Path) -> None:
    _write_profile(tmp_path / "profiles")
    cfg = _cfg(
        tmp_path,
        providers={
            "ollama": {
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
                "models": [
                    {"profile": "gemma.yaml"},
                    {"model": "gemma4:4b", "profile": "gemma.yaml"},
                ],
            }
        },
    )
    with pytest.raises(ValueError, match="duplicate model id"):
        cfg.model_selection


def test_legacy_model_string_keeps_profile_and_credentials(tmp_path: Path) -> None:
    cfg = AgentConfig(
        agent={
            "openai": {
                "model": "gemma4:26b",
                "api_key": "ollama",
                "base_url": "http://localhost:11434/v1",
            },
            "sallm": {"state_dir": str(tmp_path / "state"), "profile": "large.yaml"},
        }
    )
    original = cfg.model_selection
    apply_model_selection(cfg, "qwen3.5:9b")
    overridden = cfg.model_selection
    assert overridden.litellm_model == "openai/qwen3.5:9b"
    assert overridden.api_base == original.api_base
    assert overridden.api_key == original.api_key
    assert overridden.profile == original.profile


def test_background_model_round_trip(tmp_path: Path) -> None:
    _write_profile(tmp_path / "profiles")
    cfg = _cfg(
        tmp_path,
        default="ollama@gemma4:4b",
        providers={
            "ollama": {
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
                "models": [
                    {"profile": "gemma.yaml"},
                    {"model": "gpt-5.6-sol", "profile": "gemma.yaml"},
                ],
            }
        },
    )
    assert get_background_model(cfg) == "ollama@gemma4:4b"
    assert set_background_model(cfg, "ollama@gpt-5.6-sol") == "ollama@gpt-5.6-sol"
    assert (tmp_path / "state" / "background_model").read_text(encoding="utf-8").strip() == (
        "ollama@gpt-5.6-sol"
    )
    assert reset_background_model(cfg) == "ollama@gemma4:4b"
    assert not (tmp_path / "state" / "background_model").exists()


def test_model_slash_lists_and_sets(tmp_path: Path) -> None:
    _write_profile(tmp_path / "profiles")
    cfg = _cfg(
        tmp_path,
        providers={
            "ollama": {
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
                "models": [{"profile": "gemma.yaml"}],
            }
        },
    )
    listed = asyncio.run(resolve_chat_message(cfg, "/model"))
    assert listed.mode == "reply"
    assert "ollama@gemma4:4b" in listed.text

    updated = asyncio.run(resolve_chat_message(cfg, "/model ollama@gemma4:4b"))
    assert "Background model set to ollama@gemma4:4b" in updated.text

    unknown = asyncio.run(resolve_chat_message(cfg, "/model nope@nope"))
    assert unknown.mode == "reply"
    assert "Unknown model" in unknown.text

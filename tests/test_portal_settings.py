"""Portal settings: override layer in load_config and /v1/portal/settings."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from pawn_agent.utils import config as config_mod
from pawn_agent.utils.config import load_config, override_path
from pawn_server.core import api_server, settings_api

AUTH = {"Authorization": "Bearer tok"}

_PROFILE = "target_model: gemma4:4b\ninstructions: {}\ndemonstrations: {}\nbudgets: {}\nmetadata: {}\n"


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key in list(__import__("os").environ):
        if key.startswith("PAWN_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "gemma.yaml").write_text(_PROFILE, encoding="utf-8")
    (tmp_path / "profiles" / "big.yaml").write_text(
        _PROFILE.replace("gemma4:4b", "qwen3:32b"), encoding="utf-8"
    )
    data = {
        "db_dsn": f"sqlite:///{tmp_path / 'db.sqlite'}",
        "api": {"token": "tok", "portal_dir": ""},
        "agent": {
            "name": "Yamlbob",
            "profiles_dir": str(tmp_path / "profiles"),
            "default": "ollama@gemma4:4b",
            "sallm": {"state_dir": str(tmp_path / "state")},
            "providers": {
                "ollama": {
                    "base_url": "http://127.0.0.1:11434/v1",
                    "api_key": "ollama",
                    "models": [{"profile": "gemma.yaml"}, {"profile": "big.yaml"}],
                }
            },
        },
        "coworker": {"timezone": "Europe/Rome", "briefing_cron": "0 8 * * *", "notify": {"token": "s3cret"}},
    }
    (tmp_path / "pawnai.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return tmp_path


def _write_override(cfg, data: dict) -> None:
    path = override_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _field(payload: dict, path: str) -> dict:
    return next(f for f in payload["fields"] if f["path"] == path)


@pytest.mark.parametrize("explicit", [False, True])
def test_override_file_wins(workdir: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool) -> None:
    arg = str(workdir / "pawnai.yaml") if explicit else None
    cfg = load_config(arg)
    assert cfg.coworker.timezone == "Europe/Rome"
    _write_override(cfg, {"coworker": {"timezone": "Asia/Tokyo"}})
    monkeypatch.setenv("PAWN_COWORKER__STALE_DAYS", "4")
    cfg = load_config(arg)
    assert cfg.coworker.timezone == "Asia/Tokyo"
    assert cfg.coworker.briefing_cron == "0 8 * * *"  # sibling yaml keys survive
    assert cfg.coworker.notify.token == "s3cret"
    assert cfg.agent.name == "Yamlbob"
    assert cfg.coworker.stale_days == 4
    assert config_mod.loaded_config_file() == str((workdir / "pawnai.yaml").resolve())


def test_payload_sources_secrets_and_cron(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAWN_COWORKER__STALE_DAYS", "4")
    cfg = load_config()
    payload = settings_api.settings_payload(cfg)
    assert _field(payload, "coworker.timezone")["source"] == "yaml"
    assert _field(payload, "coworker.stale_days")["source"] == "env"
    assert _field(payload, "coworker.commitment_days")["source"] == "default"
    token = _field(payload, "coworker.notify.token")
    assert token["value"] is None and token["set"] is True and token["yaml_value"] is None
    assert len(_field(payload, "coworker.briefing_cron")["next"]) == 3
    bg = _field(payload, "agent.default")
    assert bg["value"] == "ollama@gemma4:4b" and bg["source"] == "yaml"
    assert {m["id"] for m in payload["catalog"]["models"]} == {"ollama@gemma4:4b", "ollama@qwen3:32b"}
    assert payload["restart_required"] == []


def test_patch_validates_writes_and_applies(workdir: Path) -> None:
    cfg = load_config()
    yaml_before = (workdir / "pawnai.yaml").read_text(encoding="utf-8")
    with pytest.raises(settings_api.SettingsError) as err:
        settings_api.apply_patch(
            cfg,
            {
                "coworker.briefing_cron": "not a cron",
                "coworker.autonomy.mode": "yolo",
                "api.token": "x",
                "coworker.timezone": "Mars/Base",
                "coworker.stale_days": 0,
            },
        )
    assert set(err.value.errors) == {
        "coworker.briefing_cron",
        "coworker.autonomy.mode",
        "api.token",
        "coworker.timezone",
        "coworker.stale_days",
    }
    assert not override_path(cfg).exists()

    payload = settings_api.apply_patch(
        cfg,
        {
            "coworker.autonomy.mode": "approve_writes",
            "coworker.me": "Davide, SPEAKER_00",
            "coworker.timezone": "Asia/Tokyo",
            "coworker.notify.token": "",
            "agent.anima": "",
            "agent.default": "ollama@qwen3:32b",
        },
    )
    # live field patched into the running config; restart field is pending
    assert cfg.coworker.autonomy.mode == "approve_writes"
    assert cfg.coworker.me == ["Davide", "SPEAKER_00"]
    assert cfg.coworker.timezone == "Europe/Rome"
    assert payload["restart_required"] == ["coworker.timezone"]
    tz = _field(payload, "coworker.timezone")
    assert (tz["value"], tz["running"], tz["source"]) == ("Asia/Tokyo", "Europe/Rome", "override")
    assert cfg.agent.anima is None
    assert _field(payload, "agent.default")["source"] == "override"

    from pawn_agent.utils.model_catalog import get_background_model

    assert get_background_model(cfg) == "ollama@qwen3:32b"
    saved = yaml.safe_load(override_path(cfg).read_text(encoding="utf-8"))
    assert saved["coworker"]["autonomy"] == {"mode": "approve_writes"}
    assert "notify" not in saved["coworker"]  # empty secret keeps the yaml token
    assert "default" not in saved.get("agent", {})  # background model has its own file
    assert (workdir / "pawnai.yaml").read_text(encoding="utf-8") == yaml_before

    reloaded = load_config()
    assert reloaded.coworker.autonomy.mode == "approve_writes"
    assert reloaded.coworker.notify.token == "s3cret"


def test_reset_reverts(workdir: Path) -> None:
    cfg = load_config()
    settings_api.apply_patch(cfg, {"coworker.autonomy.mode": "off", "agent.default": "ollama@qwen3:32b"})
    payload = settings_api.reset_field(cfg, "coworker.autonomy.mode")
    assert cfg.coworker.autonomy.mode == "suggest_only"
    assert _field(payload, "coworker.autonomy.mode")["source"] == "default"
    assert not override_path(cfg).exists()  # emptied file is removed
    payload = settings_api.reset_field(cfg, "agent.default")
    assert _field(payload, "agent.default")["value"] == "ollama@gemma4:4b"
    with pytest.raises(settings_api.SettingsError):
        settings_api.reset_field(cfg, "api.token")


def test_http_routes(workdir: Path) -> None:
    cfg = load_config()
    with TestClient(api_server.create_app(cfg)) as client:
        assert client.get("/v1/portal/settings").status_code == 401
        body = client.get("/v1/portal/settings", headers=AUTH).json()
        assert [g["id"] for g in body["groups"]][0] == "models"
        bad = client.patch("/v1/portal/settings", headers=AUTH, json={"coworker.weekly_cron": "nope"})
        assert bad.status_code == 422
        assert "coworker.weekly_cron" in bad.json()["detail"]["errors"]
        ok = client.patch("/v1/portal/settings", headers=AUTH, json={"portal.auto_name_every": 3})
        assert ok.status_code == 200 and cfg.portal.auto_name_every == 3
        reset = client.delete("/v1/portal/settings/portal.auto_name_every", headers=AUTH)
        assert reset.status_code == 200 and cfg.portal.auto_name_every == 6
        status = client.get("/v1/portal/settings/status", headers=AUTH).json()
        assert status["db"]["reachable"] is True
        assert status["db"]["head"]
        assert status["config_file"].endswith("pawnai.yaml")


def test_emergence_fields_resolve(workdir: Path) -> None:
    cfg = load_config()
    payload = settings_api.settings_payload(cfg)
    assert "emergence" in [g["id"] for g in payload["groups"]]
    rows = [f for f in payload["fields"] if f["group"] == "emergence"]
    assert {r["path"] for r in rows} >= {"emergence.enabled", "emergence.night_cron", "emergence.fast_model"}
    assert _field(payload, "emergence.night_cron")["next"]
    out = settings_api.apply_patch(cfg, {"emergence.triage_threshold": 0.5, "emergence.fast_model": ""})
    assert cfg.emergence.triage_threshold == 0.5
    assert _field(out, "emergence.triage_threshold")["source"] == "override"
    with pytest.raises(settings_api.SettingsError):
        settings_api.apply_patch(cfg, {"emergence.fast_model": "nope@nothing"})


def test_config_view_redacts(workdir: Path) -> None:
    from pawn_server.core import diagnostics

    cfg = load_config()
    body = diagnostics.config_payload(cfg)
    assert body["config"]["api"]["token"] == diagnostics.MASK
    assert body["config"]["coworker"]["notify"]["token"] == diagnostics.MASK
    assert body["config"]["agent"]["name"] == "Yamlbob"
    assert body["editable"]["coworker.timezone"] == "coworker"
    assert diagnostics.redact("postgresql://pawn:hunter2@db:5433/pawn") == (
        f"postgresql://pawn:{diagnostics.MASK}@db:5433/pawn"
    )
    assert diagnostics.redact({"max_tokens": 5, "keep": "x"}) == {"max_tokens": 5, "keep": "x"}


def test_scheduler_view(workdir: Path) -> None:
    from datetime import datetime, timedelta, timezone

    from pawn_agent.core import worker_status
    from pawn_agent.utils.db import (
        AgentRun,
        AgentSchedule,
        AgentScheduleFire,
        AgentScheduleProposal,
        _get_session,
    )
    from pawn_core.database import Base, get_engine
    from pawn_server.core import diagnostics

    cfg = load_config()
    tables = [t.__table__ for t in (AgentRun, AgentSchedule, AgentScheduleFire, AgentScheduleProposal)]
    Base.metadata.create_all(get_engine(cfg.db_dsn), tables=tables)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with _get_session(cfg.db_dsn) as db:
        db.add(
            AgentSchedule(
                id="s1", name="Daily digest", prompt="digest", session_id="x", status="active",
                schedule_kind="cron", timezone="UTC", cron_expression="0 9 * * *",
                next_run_at=now + timedelta(hours=2), created_at=now, updated_at=now,
            )
        )
        db.add(
            AgentRun(
                id="r1", source="schedule", command="run", model="m", status="failed",
                error="boom", created_at=now, started_at=now, completed_at=now + timedelta(seconds=3),
            )
        )
        db.add(AgentRun(id="r2", source="portal", command="chat", model="m", status="done", created_at=now))
        db.flush()
        db.add(
            AgentScheduleFire(
                id="f1", schedule_id="s1", scheduled_for=now, status="failed", agent_run_id="r1",
                claimed_at=now, started_at=now, completed_at=now + timedelta(seconds=3), created_at=now,
            )
        )
    worker_status.reset()
    worker_status.started("scheduler", poll_interval_seconds=30)
    worker_status.tick("scheduler", last_claimed=1)
    try:
        body = diagnostics.scheduler_payload(cfg)
    finally:
        worker_status.reset()
    assert "db_error" not in body
    assert body["config"]["poll_interval_seconds"] == 30.0
    assert body["workers"]["scheduler"]["ticks"] == 1
    assert [s["name"] for s in body["schedules"]] == ["Daily digest"]
    fire = body["fires"][0]
    assert fire["schedule_name"] == "Daily digest" and fire["error"] == "boom"
    assert fire["duration_s"] == 3.0
    assert body["fire_counts_24h"] == {"failed": 1}
    assert [r["id"] for r in body["runs"]] == ["r1"]  # interactive chat is left out
    crons = {c["id"]: c for c in body["crons"]}
    assert crons["briefing"]["enabled"] is False and crons["briefing"]["gate"] == "coworker.enabled"

    with TestClient(api_server.create_app(cfg)) as client:
        assert client.get("/v1/portal/settings/scheduler").status_code == 401
        assert client.get("/v1/portal/settings/scheduler", headers=AUTH).status_code == 200
        cfgv = client.get("/v1/portal/settings/config", headers=AUTH).json()
        assert cfgv["config"]["api"]["token"] == diagnostics.MASK

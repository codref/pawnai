"""capture_update keeps Obsidian-valid YAML frontmatter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pawn_agent.tools.captures_impl import capture_update_impl, slugify
from pawn_core.config import VaultConfig
from pawn_core.vault import VaultStore, parse_frontmatter
from pawn_server.core.captures import _render_research_atom
from tests.test_vault_store import FakeS3Client


@pytest.fixture
def store_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    store = VaultStore(bucket="b", agent_root="Pawn", client=FakeS3Client())
    cfg = SimpleNamespace(vault=VaultConfig(agent_root="Pawn"))
    monkeypatch.setattr(
        "pawn_agent.tools.captures_impl.vault_store_from_config",
        lambda _c: store,
    )
    return SimpleNamespace(cfg=cfg, store=store)


def test_slugify() -> None:
    assert slugify("Cyberpunk Edgerunners") == "cyberpunk-edgerunners"
    assert slugify("Movies!") == "movies"


def test_capture_update_quotes_colon_in_caption(store_env: SimpleNamespace) -> None:
    path = "Pawn/Captures/Inbox/2026-10-09 capture-ae1.md"
    body = _render_research_atom(
        title="capture",
        snippet_id="ae1",
        kind="image",
        text="",
        image_key="Pawn/Captures/assets/ae1.png",
        source_url="https://example.com",
    )
    store_env.store.write(path, body, skip_guards=True)

    out = capture_update_impl(
        store_env.cfg,
        path,
        collection="Anime",
        entity="Cyberpunk Edgerunners",
        type="still",
        caption="Anime still of a character from Cyberpunk: Edgerunners.",
        proposed_tags="anime, cyberpunk",
        status="proposed",
    )
    assert "Updated" in out
    assert "collection='anime'" in out
    assert "entity='cyberpunk-edgerunners'" in out

    text = store_env.store.read(path)
    assert text.startswith("---\n")
    meta, rest = parse_frontmatter(text)
    assert meta.get("pawn") == "capture"
    assert meta.get("status") == "proposed"
    assert meta.get("collection") == "anime"
    assert meta.get("entity") == "cyberpunk-edgerunners"
    assert "Cyberpunk: Edgerunners" in str(meta.get("caption") or "")
    assert "<!-- pawn-snippet:ae1 -->" in rest
    # Round-trip: Obsidian-style parse still sees frontmatter at byte 0
    assert parse_frontmatter(text)[0].get("entity") == "cyberpunk-edgerunners"


def test_research_capture_skill_has_capture_update() -> None:
    from pawn_agent.core.sallm_skills import build_pawn_skills

    skill = build_pawn_skills().get("research_capture")
    assert skill.tools is not None
    assert "capture_update" in skill.tools
    assert "note_read" in skill.tools
    # note_write allowed only for new entity stubs; capture atoms use capture_update
    assert "capture_update" in (skill.prompt or "")

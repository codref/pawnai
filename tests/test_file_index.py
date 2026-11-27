"""File index extraction and reconcile (portal search)."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from pawn_core.database import Base
from pawn_core.vault import VaultStore
from pawn_server.core import file_index
from pawn_server.core.portal_models import PORTAL_TABLES, PortalFileIndex, PortalFileTags
from tests.test_vault_store import FakeS3Client


@pytest.fixture
def env(tmp_path: Path) -> SimpleNamespace:
    dsn = f"sqlite:///{tmp_path / 'idx.db'}"
    engine = sa.create_engine(dsn)
    Base.metadata.create_all(engine, tables=PORTAL_TABLES)
    engine.dispose()
    cfg = SimpleNamespace(db_dsn=dsn, portal=SimpleNamespace(index_max_bytes=1000))
    store = VaultStore(bucket="b", agent_root="Pawn", client=FakeS3Client())
    return SimpleNamespace(cfg=cfg, store=store, dsn=dsn)


def _rows(dsn: str) -> dict[str, PortalFileIndex]:
    engine = sa.create_engine(dsn)
    with sa.orm.Session(engine) as db:
        rows = {r.key: r for r in db.scalars(sa.select(PortalFileIndex))}
        for r in rows.values():
            db.expunge(r)
    engine.dispose()
    return rows


def test_extract_markdown_frontmatter() -> None:
    out = file_index.extract("a/Note.md", b"---\ntags: one, two/x\n---\n# Heading\nbody")
    assert out["kind"] == "markdown" and out["title"] == "Heading"
    assert out["tags"] == ["one", "two/x"] and "body" in out["content"]
    assert "tags:" not in out["content"]


def test_extract_docx_and_pdf() -> None:
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Wombat quarterly plan")
    buf = io.BytesIO()
    doc.save(buf)
    assert "Wombat" in file_index.extract("x.docx", buf.getvalue())["content"]

    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    pdf = io.BytesIO()
    writer.write(pdf)
    out = file_index.extract("x.pdf", pdf.getvalue())
    assert out["kind"] == "pdf" and out["error"] is None

    bad = file_index.extract("broken.pdf", b"not a pdf")
    assert bad["error"]


def test_image_without_vision_model_is_name_only() -> None:
    out = file_index.extract("p.png", b"\x89PNG", SimpleNamespace(portal=None))
    assert out["kind"] == "image" and out["content"] == ""


def test_reconcile_add_change_delete(env: SimpleNamespace) -> None:
    env.store.write("Pawn/a.md", "# A\nalpha", skip_guards=True)
    env.store.write("Pawn/big.txt", "x" * 2000, skip_guards=True)
    env.store.write(".obsidian/app.json", "{}", skip_guards=True)
    counts = file_index.reconcile(env.cfg, env.store)
    assert counts == {"files": 2, "indexed": 2, "removed": 0}
    rows = _rows(env.dsn)
    assert "alpha" in rows["Pawn/a.md"].content
    assert rows["Pawn/big.txt"].error == "too large"

    assert file_index.reconcile(env.cfg, env.store)["indexed"] == 0
    env.store.write("Pawn/a.md", "# A\nbeta", skip_guards=True)
    env.store.delete("Pawn/big.txt")
    counts = file_index.reconcile(env.cfg, env.store)
    assert counts == {"files": 1, "indexed": 1, "removed": 1}
    assert "beta" in _rows(env.dsn)["Pawn/a.md"].content


def test_index_merges_stored_tags(env: SimpleNamespace) -> None:
    env.store.write("Pawn/t.md", "---\ntags: [fm]\n---\nx", skip_guards=True)
    engine = sa.create_engine(env.dsn)
    with sa.orm.Session(engine) as db:
        db.add(PortalFileTags(key="Pawn/t.md", tags=["db"]))
        db.commit()
    engine.dispose()
    file_index.index_keys(env.cfg, ["Pawn/t.md"], env.store)
    assert _rows(env.dsn)["Pawn/t.md"].tags == ["fm", "db"]

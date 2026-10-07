"""Item note rendering and triage command parsing."""

from pawn_agent.core.coworker.actions import parse_coworker_command
from pawn_agent.core.coworker.notes import parse_item_note, render_item_note, render_today


def test_item_note_roundtrip():
    raw = render_item_note(
        item_id="uuid",
        short_id="abcd1234",
        status="new",
        kind="decision",
        text="Keep it on our S3.",
        thread="Project X",
        quote="we stay on S3",
        source_link="[[Pawn/Transcripts/s.md]]",
        reason="New commitment.",
        interrupt=True,
    )
    parsed = parse_item_note(raw)
    assert parsed["short_id"] == "abcd1234"
    assert parsed["action"] == ""
    assert "Keep it on our S3." in raw
    assert "Why you were notified" in raw


def test_today_groups():
    body = render_today(
        {
            "attention": [
                {"note_key": "Pawn/Items/a.md", "kind": "decision", "text": "Ship", "thread": "X"}
            ],
            "filed": [{"text": "Mentioned tools", "thread": "X"}],
        }
    )
    assert "[[Pawn/Items/a.md]]" in body
    assert "### X" in body


def test_parse_coworker_command():
    assert parse_coworker_command("file abcd1234") == ("file", "abcd1234", None)
    assert parse_coworker_command("later abcd1234 2026-10-01") == (
        "later",
        "abcd1234",
        "2026-10-01",
    )
    assert parse_coworker_command("todo abcd1234") == ("todo", "abcd1234", None)
    assert parse_coworker_command("delete abcd1234") == ("delete", "abcd1234", None)
    assert parse_coworker_command("hello") is None


def test_item_note_key_readable():
    from types import SimpleNamespace

    from pawn_agent.core.coworker.actions import item_note_key

    cfg = SimpleNamespace(coworker=SimpleNamespace(items_dir="Pawn/Items"))
    key = item_note_key(
        cfg,  # type: ignore[arg-type]
        short_id="abcd1234",
        text="Keep the recordings on our S3 through Friday.",
        created_at=__import__("datetime").datetime(
            2026, 10, 7, tzinfo=__import__("datetime").timezone.utc
        ),
    )
    assert key.startswith("Pawn/Items/2026-10-07-")
    assert key.endswith("-abcd1234.md")
    assert "keep-the-recordings" in key

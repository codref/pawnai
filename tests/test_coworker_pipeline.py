"""JSON item parsing used by extract and score."""

from pawn_agent.core.coworker.jsonutil import normalize_kind, parse_items_payload


def test_parse_fenced_items():
    raw = """```json
{"items": [{"kind": "decision", "text": "Ship Friday"}]}
```"""
    items = parse_items_payload(raw)
    assert items[0]["text"] == "Ship Friday"
    assert normalize_kind("open question") == "open_question"


def test_bad_json_is_empty():
    assert parse_items_payload("not json") == []

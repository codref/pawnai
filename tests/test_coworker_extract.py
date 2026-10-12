"""Extract normalization: context/quote soft caps and claim required."""

from pawn_agent.core.coworker.extract import (
    _CONTEXT_MAX,
    _QUOTE_MAX,
    normalize_extracted_entry,
)


def test_normalize_requires_text():
    empty = normalize_extracted_entry({"kind": "decision", "text": "  "})
    assert empty is None


def test_normalize_keeps_context_and_quote():
    item = normalize_extracted_entry(
        {
            "kind": "decision",
            "text": "Keep recordings on S3 through Friday.",
            "context": (
                "Davide and Mira agreed not to migrate before the deploy."
            ),
            "owner": "Davide",
            "due": "2026-10-17",
            "quote": "We should not move the bucket before the deploy.",
        }
    )
    assert item is not None
    assert item["kind"] == "decision"
    assert item["text"].startswith("Keep recordings")
    assert "Davide and Mira" in item["context"]
    assert "bucket" in item["quote"]
    assert item["owner"] == "Davide"
    assert item["due"] == "2026-10-17"


def test_normalize_soft_trims_context_and_quote():
    long_context = "word " * 200
    long_quote = "said " * 400
    item = normalize_extracted_entry(
        {
            "kind": "commitment",
            "text": "Ship the cutover.",
            "context": long_context,
            "quote": long_quote,
        }
    )
    assert item is not None
    assert len(item["context"]) <= _CONTEXT_MAX
    assert len(item["quote"]) <= _QUOTE_MAX
    assert not item["context"].endswith(" ")

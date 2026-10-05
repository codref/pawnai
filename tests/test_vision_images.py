"""Note-image selection: caption once, then skip until the user asks again."""

from __future__ import annotations

import hashlib

import pytest

from pawn_agent.core.vision import (
    TurnImage,
    decode_chat_image,
    embed_keys,
    load_note_images,
    note_image_targets,
    select_turn_images,
    vision_refusal,
)


def _png(name: str, payload: bytes) -> TurnImage:
    return TurnImage(filename=name, media_type="image/png", data=payload, role="context")


def test_note_embeds_keep_document_order() -> None:
    body = "\n".join(
        [
            "See ![second](folder/second.jpg) then ![[first.png]].",
            "![[first.png]]",
            "![web](https://example.com/skip.png)",
            "![[notes/page]]",
        ]
    )
    assert note_image_targets(body) == ["folder/second.jpg", "first.png"]
    assert embed_keys("Notes/Page.md", "first.png") == ["Notes/first.png", "first.png"]
    assert embed_keys("Notes/Page.md", "folder/second.jpg")[0] == "folder/second.jpg"


def test_select_skips_known_context_and_caps_at_four() -> None:
    images = [
        TurnImage(filename="q.png", media_type="image/png", data=b"question", role="question"),
        _png("a.png", b"a"),
        _png("b.png", b"b"),
        _png("c.png", b"c"),
        _png("d.png", b"d"),
        _png("e.png", b"e"),
    ]
    known = {hashlib.sha256(b"a").hexdigest(), hashlib.sha256(b"b").hexdigest()}
    chosen = select_turn_images(images, known=known)
    assert [image.filename for image in chosen] == ["q.png", "c.png", "d.png", "e.png"]

    again = select_turn_images(images, known=known, force=True)
    assert [image.filename for image in again] == ["q.png", "a.png", "b.png", "c.png"]


def test_load_note_images_reads_vault_bytes() -> None:
    class Store:
        def read_bytes(self, key: str) -> bytes:
            if key != "Notes/shot.png":
                raise FileNotFoundError(key)
            return b"png-bytes"

    found = load_note_images(Store(), [("Notes/Page.md", "Look ![[shot.png]]")])
    assert len(found) == 1
    assert found[0].filename == "shot.png"
    assert found[0].role == "context"
    assert found[0].data == b"png-bytes"


def test_decode_rejects_non_images() -> None:
    with pytest.raises(Exception, match="not an image"):
        decode_chat_image(
            filename="notes.txt",
            media_type="text/plain",
            data_base64="aGVsbG8=",
            role="question",
        )


def test_vision_refusal_names_the_surface() -> None:
    assert "/model" in vision_refusal("matrix")
    assert "model menu" in vision_refusal("obsidian")

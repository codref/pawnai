"""Unit tests for pawn_core.vault."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from botocore.exceptions import ClientError

from pawn_core.vault import (
    VaultConflict,
    VaultStore,
    VaultWriteDenied,
    dump_frontmatter,
    is_obsidian_meta,
    is_under_agent_root,
    normalize_vault_key,
    parse_frontmatter,
)


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3Client:
    """Minimal in-memory S3 stand-in for VaultStore tests."""

    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        obj = self.objects.get(Key)
        if obj is None:
            raise _no_such_key(Key)
        return {"Body": _FakeBody(obj["body"]), "ETag": obj["etag"]}

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        obj = self.objects.get(Key)
        if obj is None:
            raise _no_such_key(Key)
        return {
            "ETag": obj["etag"],
            "ContentLength": len(obj["body"]),
            "LastModified": obj.get("last_modified"),
        }

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kwargs) -> dict:
        etag = f"etag-{len(self.objects) + 1}"
        self.objects[Key] = {
            "body": Body if isinstance(Body, bytes) else Body.encode("utf-8"),
            "etag": f'"{etag}"',
            "last_modified": datetime.now(timezone.utc),
        }
        return {"ETag": f'"{etag}"'}

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        self.objects.pop(Key, None)

    def list_objects_v2(self, *, Bucket: str, Prefix: str = "", **kwargs) -> dict:
        contents = []
        for key, obj in sorted(self.objects.items()):
            if not key.startswith(Prefix):
                continue
            contents.append(
                {
                    "Key": key,
                    "Size": len(obj["body"]),
                    "ETag": obj["etag"],
                }
            )
        return {"Contents": contents, "IsTruncated": False}

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return _FakePaginator(self)


def _no_such_key(key: str) -> ClientError:
    return ClientError({"Error": {"Code": "NoSuchKey", "Message": key}}, "HeadObject")


class _FakePaginator:
    def __init__(self, client: FakeS3Client) -> None:
        self._client = client

    def paginate(self, *, Bucket: str, Prefix: str = ""):
        contents = []
        for key, obj in sorted(self._client.objects.items()):
            if not key.startswith(Prefix):
                continue
            contents.append(
                {
                    "Key": key,
                    "Size": len(obj["body"]),
                }
            )
        yield {"Contents": contents}


@pytest.fixture
def store() -> VaultStore:
    return VaultStore(bucket="vault", client=FakeS3Client(), agent_root="Pawn")


def test_normalize_vault_key() -> None:
    assert normalize_vault_key("\\Pawn\\Note.md") == "Pawn/Note.md"
    assert normalize_vault_key("/Pawn/Note.md") == "Pawn/Note.md"


def test_parse_dump_frontmatter() -> None:
    text = "---\ntitle: Hello\npawn: editable\n---\n\nBody here\n"
    meta, body = parse_frontmatter(text)
    assert meta["title"] == "Hello"
    assert meta["pawn"] == "editable"
    assert "Body here" in body

    roundtrip = dump_frontmatter(meta, body)
    meta2, body2 = parse_frontmatter(roundtrip)
    assert meta2["title"] == "Hello"
    assert body2.strip() == "Body here"


def test_assert_writable_denies_obsidian(store: VaultStore) -> None:
    with pytest.raises(VaultWriteDenied):
        store.assert_writable(".obsidian/app.json")
    assert is_obsidian_meta(".obsidian/plugins/foo.json")


def test_assert_writable_allows_agent_root(store: VaultStore) -> None:
    store.assert_writable("Pawn/Tasks/new.md")
    assert is_under_agent_root("Pawn/Tasks/x.md", "Pawn")


def test_assert_writable_outside_root_requires_pawn_editable(store: VaultStore) -> None:
    with pytest.raises(VaultWriteDenied):
        store.assert_writable("Inbox/locked.md", existing_body="No frontmatter")

    store.assert_writable(
        "Inbox/open.md",
        existing_body="---\npawn: editable\n---\n",
    )


def test_write_and_write_if_unchanged(store: VaultStore) -> None:
    stat = store.write("Pawn/Tasks/a.md", "# Task\n")
    assert stat.key == "Pawn/Tasks/a.md"
    assert store.read("Pawn/Tasks/a.md") == "# Task\n"
    assert store.exists("Pawn/")

    with pytest.raises(VaultConflict):
        store.write_if_unchanged("Pawn/Tasks/a.md", "# Task v2\n", "wrong-etag")

    stat2 = store.write_if_unchanged("Pawn/Tasks/a.md", "# Task v2\n", stat.etag)
    assert "v2" in store.read("Pawn/Tasks/a.md")
    assert stat2.etag != stat.etag


def test_delete_only_under_agent_root(store: VaultStore) -> None:
    store.write("Pawn/tmp.md", "x")
    store.delete("Pawn/tmp.md")
    with pytest.raises(VaultWriteDenied):
        store.delete("Inbox/other.md")

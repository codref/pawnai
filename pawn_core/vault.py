"""S3-backed Markdown vault store.

Pawn treats the vault as a flat keyspace of Markdown files under an optional
prefix. Obsidian Sync Engine (or any sync) keeps device copies in sync; this
module is the only way Pawn reads and writes the bucket.

Write guards (enforced here, not left to the model):
- Never touch ``.obsidian/``.
- Free write under ``agent_root`` (default ``Pawn/``).
- Outside the agent root, require frontmatter ``pawn: editable``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import boto3
import yaml
from botocore.client import BaseClient
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class VaultError(Exception):
    """Base error for vault operations."""


class VaultWriteDenied(VaultError):
    """Raised when a write violates vault guards."""


class VaultConflict(VaultError):
    """Raised when an ETag-checked write sees a mismatched remote object."""


class VaultNotFound(VaultError):
    """Raised when a key does not exist."""


@dataclass
class VaultStat:
    """Metadata for one vault object."""

    key: str
    etag: str
    size: int
    last_modified: Optional[datetime] = None


def normalize_vault_key(path: str) -> str:
    """Normalize to a forward-slash key without a leading slash."""
    key = (path or "").replace("\\", "/").strip()
    while "//" in key:
        key = key.replace("//", "/")
    return key.lstrip("/")


def is_obsidian_meta(key: str) -> bool:
    """True when *key* is under ``.obsidian/``."""
    k = normalize_vault_key(key)
    return k == ".obsidian" or k.startswith(".obsidian/")


def is_under_agent_root(key: str, agent_root: str = "Pawn") -> bool:
    """True when *key* is the agent root or a descendant."""
    root = normalize_vault_key(agent_root).rstrip("/")
    k = normalize_vault_key(key)
    if not root:
        return True
    return k == root or k.startswith(root + "/")


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(frontmatter_dict, body)``. Empty dict when no YAML fence."""
    if not text:
        return {}, ""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}, text
    raw = match.group(1)
    body = text[match.end() :]
    try:
        meta = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, body


def dump_frontmatter(meta: dict[str, Any], body: str) -> str:
    """Serialize YAML frontmatter plus *body*."""
    if not meta:
        return body if body.endswith("\n") or not body else body + "\n"
    dumped = yaml.safe_dump(
        meta,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).rstrip()
    body_text = body if not body or body.startswith("\n") else "\n" + body
    if body_text and not body_text.endswith("\n"):
        body_text += "\n"
    return f"---\n{dumped}\n---\n{body_text}"


def _strip_etag(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.strip().strip('"')


class VaultStore:
    """Read/write Markdown notes in an S3 bucket prefix."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
        path_style: bool = True,
        verify_ssl: bool = True,
        agent_root: str = "Pawn",
        client: Optional[BaseClient] = None,
    ) -> None:
        if not bucket and client is None:
            raise VaultError("vault.s3.bucket is required")
        self.bucket = bucket
        self.prefix = normalize_vault_key(prefix)
        if self.prefix and not self.prefix.endswith("/"):
            # Keep prefix as a directory-like string without forcing trailing
            # slash into relative keys; applied only in _full_key.
            pass
        self.agent_root = normalize_vault_key(agent_root).rstrip("/") or "Pawn"
        if client is not None:
            self._client = client
        else:
            session_kwargs: dict[str, Any] = {}
            if access_key and secret_key:
                session_kwargs["aws_access_key_id"] = access_key
                session_kwargs["aws_secret_access_key"] = secret_key
            if region:
                session_kwargs["region_name"] = region
            session = boto3.session.Session(**session_kwargs)
            self._client = session.client(
                "s3",
                endpoint_url=endpoint_url,
                verify=verify_ssl,
                config=BotoConfig(s3={"addressing_style": "path" if path_style else "virtual"}),
            )

    def _full_key(self, key: str) -> str:
        rel = normalize_vault_key(key)
        if not self.prefix:
            return rel
        prefix = self.prefix.rstrip("/")
        return f"{prefix}/{rel}" if rel else prefix

    def assert_writable(
        self,
        key: str,
        *,
        existing_body: Optional[str] = None,
        new_body: Optional[str] = None,
    ) -> None:
        """Raise :class:`VaultWriteDenied` when *key* may not be written."""
        k = normalize_vault_key(key)
        if not k:
            raise VaultWriteDenied("empty vault key")
        if is_obsidian_meta(k):
            raise VaultWriteDenied("writes under .obsidian/ are forbidden")
        if is_under_agent_root(k, self.agent_root):
            return
        body = existing_body
        if body is None and new_body is not None:
            body = new_body
        if body is None:
            try:
                body = self.read(k)
            except VaultNotFound:
                body = ""
            except VaultError:
                body = ""
        meta, _ = parse_frontmatter(body or "")
        pawn = meta.get("pawn")
        if pawn == "editable" or pawn is True:
            return
        raise VaultWriteDenied(f"{k!r} is outside {self.agent_root}/ and lacks pawn: editable")

    def read(self, key: str) -> str:
        """Return object body as UTF-8 text."""
        full = self._full_key(key)
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=full)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise VaultNotFound(normalize_vault_key(key)) from exc
            raise VaultError(f"read failed for {key!r}: {exc}") from exc
        raw = resp["Body"].read()
        return raw.decode("utf-8")

    def exists(self, key: str) -> bool:
        return self.stat(key) is not None

    def stat(self, key: str) -> Optional[VaultStat]:
        full = self._full_key(key)
        try:
            resp = self._client.head_object(Bucket=self.bucket, Key=full)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound", "405", "403"}:
                # Some providers return 403/405 for missing keys on HEAD.
                if code in {"404", "NoSuchKey", "NotFound"}:
                    return None
                # Fall through: treat ambiguous as missing only when listed.
                return None
            raise VaultError(f"stat failed for {key!r}: {exc}") from exc
        return VaultStat(
            key=normalize_vault_key(key),
            etag=_strip_etag(resp.get("ETag")),
            size=int(resp.get("ContentLength") or 0),
            last_modified=resp.get("LastModified"),
        )

    def _ensure_folder_markers(self, key: str) -> None:
        """Best-effort zero-byte ``folder/`` markers for Sync Engine."""
        parts = normalize_vault_key(key).split("/")
        if len(parts) < 2:
            return
        for i in range(1, len(parts)):
            folder = "/".join(parts[:i]) + "/"
            full = self._full_key(folder)
            try:
                self._client.head_object(Bucket=self.bucket, Key=full)
            except ClientError:
                try:
                    self._client.put_object(
                        Bucket=self.bucket,
                        Key=full,
                        Body=b"",
                        ContentType="application/x-directory",
                    )
                except ClientError as exc:
                    logger.debug("folder marker %s skipped: %s", folder, exc)

    def write(
        self,
        key: str,
        body: str,
        *,
        content_type: str = "text/markdown; charset=utf-8",
        skip_guards: bool = False,
    ) -> VaultStat:
        """Put *body* at *key*. Returns the resulting :class:`VaultStat`."""
        k = normalize_vault_key(key)
        if not skip_guards:
            existing: Optional[str] = None
            if self.exists(k):
                try:
                    existing = self.read(k)
                except VaultError:
                    existing = None
            self.assert_writable(k, existing_body=existing, new_body=body)
        self._ensure_folder_markers(k)
        full = self._full_key(k)
        data = body.encode("utf-8")
        try:
            resp = self._client.put_object(
                Bucket=self.bucket,
                Key=full,
                Body=data,
                ContentType=content_type,
            )
        except ClientError as exc:
            raise VaultError(f"write failed for {k!r}: {exc}") from exc
        etag = _strip_etag(resp.get("ETag"))
        if not etag:
            st = self.stat(k)
            if st is not None:
                return st
        return VaultStat(key=k, etag=etag, size=len(data))

    def write_if_unchanged(
        self,
        key: str,
        body: str,
        expected_etag: str,
        *,
        content_type: str = "text/markdown; charset=utf-8",
    ) -> VaultStat:
        """Write only when the current ETag matches *expected_etag*."""
        k = normalize_vault_key(key)
        current = self.stat(k)
        want = _strip_etag(expected_etag)
        have = current.etag if current else ""
        if want and have and want != have:
            raise VaultConflict(f"{k!r}: etag mismatch (expected {want!r}, have {have!r})")
        if want and current is None:
            raise VaultConflict(f"{k!r}: expected etag {want!r} but object missing")
        return self.write(k, body, content_type=content_type)

    def append(self, key: str, suffix: str) -> VaultStat:
        """Append *suffix* to an existing note (or create it)."""
        k = normalize_vault_key(key)
        try:
            existing = self.read(k)
        except VaultNotFound:
            existing = ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        return self.write(k, existing + suffix)

    def list(self, prefix: str = "", *, suffix: str = ".md") -> list[str]:
        """List relative vault keys under *prefix* (optional *suffix* filter)."""
        rel_prefix = normalize_vault_key(prefix)
        full_prefix = self._full_key(rel_prefix)
        if rel_prefix and not full_prefix.endswith("/"):
            # Allow listing a folder by path without trailing slash.
            full_prefix = full_prefix + "/"
        keys: list[str] = []
        token: Optional[str] = None
        vault_prefix = self.prefix.rstrip("/")
        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": full_prefix,
            }
            if token:
                kwargs["ContinuationToken"] = token
            try:
                resp = self._client.list_objects_v2(**kwargs)
            except ClientError as exc:
                raise VaultError(f"list failed for {prefix!r}: {exc}") from exc
            for item in resp.get("Contents") or []:
                full = item.get("Key") or ""
                if full.endswith("/"):
                    continue
                if vault_prefix and full.startswith(vault_prefix + "/"):
                    rel = full[len(vault_prefix) + 1 :]
                elif vault_prefix and full == vault_prefix:
                    continue
                else:
                    rel = full
                if suffix and not rel.endswith(suffix):
                    continue
                keys.append(rel)
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return sorted(keys)

    def delete(self, key: str) -> None:
        """Delete a key (agent_root only)."""
        k = normalize_vault_key(key)
        if not is_under_agent_root(k, self.agent_root):
            raise VaultWriteDenied(f"delete denied outside {self.agent_root}/: {k!r}")
        if is_obsidian_meta(k):
            raise VaultWriteDenied("deletes under .obsidian/ are forbidden")
        full = self._full_key(k)
        try:
            self._client.delete_object(Bucket=self.bucket, Key=full)
        except ClientError as exc:
            raise VaultError(f"delete failed for {k!r}: {exc}") from exc


def resolve_path_template(
    template: str,
    *,
    agent_root: str = "Pawn",
    session_id: str = "",
    title: str = "",
    date: Optional[str] = None,
    task_id: str = "",
    now: Optional[datetime] = None,
) -> str:
    """Format a vault path template and normalize the result."""
    from datetime import timezone

    when = now or datetime.now(timezone.utc)
    date_str = date or when.strftime("%Y-%m-%d")
    rendered = template.format(
        agent_root=agent_root.rstrip("/"),
        session_id=session_id,
        title=title or session_id,
        date=date_str,
        year=when.strftime("%Y"),
        month=when.strftime("%m"),
        id=task_id,
    )
    return normalize_vault_key(rendered)

"""SiYuan Note integration for PawnAI.

Provides :class:`SiyuanClient` for pushing conversation analyses to a locally
running SiYuan instance (``http://127.0.0.1:6806`` by default).

Each session is created as a standalone SiYuan **document** with:
- Full structured analysis (title, summary, topics, highlights, sentiment)
- Full transcript
- Block attributes for tags and metadata (enabling SiYuan queries)
- A backlink inserted into today's daily note

Usage::

    from pawnai.core.siyuan import SiyuanClient, format_session_markdown

    client = SiyuanClient(
        url="http://127.0.0.1:6806",
        token="your_token",
        notebook_id="20210817205410-2kvfpfn",
    )
    md = format_session_markdown(analysis_row, transcript)
    doc_id = client.upsert_session_doc(
        notebook="20210817205410-2kvfpfn",
        path="/Conversations/2026-02/myconv",
        markdown=md,
        attrs={"custom-session-id": "myconv", "custom-tags": "oci, migration"},
    )
    client.append_daily_note_link(
        notebook="20210817205410-2kvfpfn",
        daily_path="/daily note/2026/02/2026-02-28",
        doc_id=doc_id,
        title="Internal Team Meeting on OCI Cloud Migration Challenges",
    )
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


# ──────────────────────────────────────────────────────────────────────────────
# Default path templates
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PATH_TEMPLATE = "/Conversations/{date}/{title}"
DEFAULT_DAILY_PATH_TEMPLATE = "/daily note/{year}/{month}/{date}"


# ──────────────────────────────────────────────────────────────────────────────
# Markdown formatter
# ──────────────────────────────────────────────────────────────────────────────


def format_session_markdown(
    title: Optional[str],
    summary: Optional[str],
    key_topics: Optional[str],
    speaker_highlights: Optional[str],
    sentiment: Optional[str],
    sentiment_tags: Optional[List[str]],
    tags: Optional[List[str]],
    session_id: Optional[str],
    source: str,
    analyzed_at: Optional[datetime],
    transcript: str,
    model: str = "",
) -> str:
    """Build a SiYuan-compatible Markdown document from analysis fields.

    The resulting document has the shape::

        # {title}
        **Session:** `{session_id}` · **Date:** {date} · **Model:** {model}
        **Tags:** tag1, tag2
        **Sentiment:** label1, label2
        ---
        ## Summary
        …
        ## Key Topics / Keywords
        …
        ## Speaker Highlights
        …
        ## Sentiment
        …
        ---
        ## Transcript
        …

    Args:
        title: Short descriptive title.
        summary: Summary paragraph.
        key_topics: Bullet list of key topics.
        speaker_highlights: Per-speaker highlights.
        sentiment: Overall sentiment prose.
        sentiment_tags: Up to 3 short sentiment labels.
        tags: 5–10 topic/entity tags.
        session_id: Session identifier (shown in header).
        source: Raw source label from the DB.
        analyzed_at: UTC datetime the analysis was produced.
        transcript: Full transcript text (``[MM:SS.ss] Speaker: text``).
        model: Copilot model name (shown in header).

    Returns:
        Markdown string ready to pass to ``createDocWithMd``.
    """
    date_str = (
        analyzed_at.strftime("%Y-%m-%d %H:%M UTC")
        if analyzed_at
        else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )

    header_parts = []
    if session_id:
        header_parts.append(f"**Session:** `{session_id}`")
    else:
        header_parts.append(f"**Source:** `{source}`")
    header_parts.append(f"**Date:** {date_str}")
    if model:
        header_parts.append(f"**Model:** {model}")

    tag_line = ""
    if tags:
        tag_line = "\n**Tags:** " + ", ".join(tags)
    if sentiment_tags:
        tag_line += "\n**Sentiment tags:** " + ", ".join(sentiment_tags)

    sections: List[str] = []
    sections.append(f"# {title or 'Conversation Analysis'}")
    sections.append(" · ".join(header_parts) + tag_line)
    sections.append("---")

    if summary:
        sections.append("## Summary\n\n" + summary)
    if key_topics:
        sections.append("## Key Topics / Keywords\n\n" + key_topics)
    if speaker_highlights:
        sections.append("## Speaker Highlights\n\n" + speaker_highlights)
    if sentiment:
        sections.append("## Sentiment\n\n" + sentiment)

    sections.append("---")
    sections.append("## Transcript\n\n" + (transcript or "_No transcript available._"))

    return "\n\n".join(sections)


def resolve_path_template(
    template: str,
    session_id: str,
    title: Optional[str] = None,
    now: Optional[datetime] = None,
) -> str:
    """Fill placeholders in a path template.

    Supported placeholders: ``{session_id}``, ``{title}``, ``{date}``,
    ``{year}``, ``{month}``, ``{day}``.

    ``{title}`` is slugified (lowercased, whitespace → hyphens, special chars
    stripped); falls back to *session_id* when title is absent.

    Args:
        template: Path template string, e.g. ``"/Conversations/{date}/{session_id}"``.
        session_id: Session identifier.
        title: Optional document title for ``{title}`` placeholder.
        now: Datetime to use for date placeholders (defaults to UTC now).

    Returns:
        Resolved path string.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    return template.format(
        session_id=session_id,
        title=title if title else session_id,
        date=now.strftime("%Y-%m-%d"),
        year=now.strftime("%Y"),
        month=now.strftime("%m"),
        day=now.strftime("%d"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# SiYuan API client
# ──────────────────────────────────────────────────────────────────────────────


class SiyuanClient:
    """HTTP client for the SiYuan Note local API.

    Args:
        url: Base URL of the SiYuan instance, e.g. ``"http://127.0.0.1:6806"``.
        token: API token from SiYuan Settings → About.
        notebook_id: Default notebook ID used when none is supplied to methods.
        timeout: Request timeout in seconds (default 30).

    Raises:
        :class:`SiyuanError`: When the API returns a non-zero ``code``.
        :class:`requests.RequestException`: On network / HTTP failures.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:6806",
        token: str = "",
        notebook_id: str = "",
        timeout: int = 30,
    ) -> None:
        self._base = url.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {token}",
        }
        self.notebook_id = notebook_id
        self._timeout = timeout

    # ── low-level ─────────────────────────────────────────────────────────────

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Any:
        """POST *payload* to *endpoint* and return ``data`` on success.

        Args:
            endpoint: API path, e.g. ``"/api/notebook/lsNotebooks"``.
            payload: JSON-serialisable request body.

        Returns:
            The ``data`` field of the SiYuan response object.

        Raises:
            SiyuanError: If ``code != 0``.
            requests.RequestException: On network failure.
        """
        resp = requests.post(
            f"{self._base}{endpoint}",
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("code", 0) != 0:
            raise SiyuanError(
                f"SiYuan API error [{body.get('code')}]: {body.get('msg')} "
                f"(endpoint={endpoint})"
            )
        return body.get("data")

    # ── notebooks ─────────────────────────────────────────────────────────────

    def list_notebooks(self) -> List[Dict[str, Any]]:
        """Return all open notebooks.

        Returns:
            List of notebook dicts (``id``, ``name``, ``icon``, ``sort``, ``closed``).
        """
        data = self._post("/api/notebook/lsNotebooks", {})
        return data.get("notebooks", []) if data else []

    # ── documents ─────────────────────────────────────────────────────────────

    def get_doc_ids_by_path(self, notebook: str, path: str) -> List[str]:
        """Resolve a human-readable document path to a list of block IDs.

        Args:
            notebook: Notebook ID.
            path: Human-readable path, e.g. ``"/Conversations/2026-02/myconv"``.

        Returns:
            List of document block IDs (usually zero or one element).
        """
        try:
            data = self._post(
                "/api/filetree/getIDsByHPath",
                {"path": path, "notebook": notebook},
            )
            return data if isinstance(data, list) else []
        except SiyuanError:
            return []

    def remove_doc(self, doc_id: str) -> None:
        """Delete a document by its block ID.

        Args:
            doc_id: Document block ID to delete.
        """
        self._post("/api/filetree/removeDocByID", {"id": doc_id})

    def create_doc(self, notebook: str, path: str, markdown: str) -> str:
        """Create a document from Markdown content.

        If a document already exists at *path* it will **not** be overwritten
        by this call alone — use :meth:`upsert_session_doc` instead.

        Args:
            notebook: Notebook ID.
            path: Target human-readable path (must start with ``/``).
            markdown: GFM Markdown content for the new document.

        Returns:
            Block ID of the newly created document.
        """
        return self._post(
            "/api/filetree/createDocWithMd",
            {"notebook": notebook, "path": path, "markdown": markdown},
        )

    # ── blocks ────────────────────────────────────────────────────────────────

    def append_block(self, parent_id: str, markdown: str) -> str:
        """Append a Markdown block to a parent block (document or container).

        Args:
            parent_id: Block ID of the parent document or container.
            markdown: Markdown content of the block to append.

        Returns:
            Block ID of the newly created block.
        """
        data = self._post(
            "/api/block/appendBlock",
            {"dataType": "markdown", "data": markdown, "parentID": parent_id},
        )
        ops = data[0]["doOperations"] if data else []
        return ops[0]["id"] if ops else ""

    # ── attributes ────────────────────────────────────────────────────────────

    def set_block_attrs(self, block_id: str, attrs: Dict[str, str]) -> None:
        """Set custom attributes on a block (document-level or any block).

        Custom attribute keys **must** be prefixed with ``custom-``.

        Args:
            block_id: Target block ID.
            attrs: Mapping of attribute name → value.
        """
        self._post(
            "/api/attr/setBlockAttrs",
            {"id": block_id, "attrs": attrs},
        )

    # ── high-level helpers ────────────────────────────────────────────────────

    def upsert_session_doc(
        self,
        notebook: str,
        path: str,
        markdown: str,
        attrs: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create (or re-create) a session document at *path*.

        If a document already exists at *path* it is deleted first, then a
        fresh document is created from *markdown*.

        Args:
            notebook: Notebook ID.
            path: Target human-readable path (must start with ``/``).
            markdown: Full document content.
            attrs: Optional block attributes to set after creation
                (e.g. ``custom-session-id``, ``custom-tags``).

        Returns:
            Block ID of the newly created document.
        """
        existing = self.get_doc_ids_by_path(notebook, path)
        for doc_id in existing:
            self.remove_doc(doc_id)

        new_doc_id = self.create_doc(notebook, path, markdown)

        if attrs and new_doc_id:
            self.set_block_attrs(new_doc_id, attrs)

        return new_doc_id

    # ── inbox (shorthands) ──────────────────────────────────────────────────
    # NOTE: These endpoints are internal (not in the public API docs) and may
    # change in future SiYuan releases.  Auth header format is identical to the
    # documented APIs.

    def create_shorthand(self, content: str) -> str:
        """Send a message to the SiYuan Inbox (shorthands).

        Args:
            content: Markdown content of the inbox item.

        Returns:
            ID of the newly created inbox item.
        """
        data = self._post("/api/inbox/createShorthand", {"content": content})
        return (data or {}).get("id", "")

    def get_shorthands(self, page: int = 1) -> List[Dict[str, Any]]:
        """List inbox items (shorthands), paginated.

        Args:
            page: 1-based page number (default 1).

        Returns:
            List of shorthand dicts.
        """
        data = self._post("/api/inbox/getShorthands", {"page": page})
        return (data or {}).get("shorthands", [])

    def get_shorthand(self, shorthand_id: str) -> Dict[str, Any]:
        """Fetch a single inbox item by ID.

        Args:
            shorthand_id: Inbox item ID.

        Returns:
            Shorthand dict, or empty dict if not found.
        """
        data = self._post("/api/inbox/getShorthand", {"id": shorthand_id})
        return data or {}

    def remove_shorthands(self, ids: List[str]) -> None:
        """Delete one or more inbox items by ID.

        Args:
            ids: List of inbox item IDs to delete.
        """
        self._post("/api/inbox/removeShorthands", {"ids": ids})

    # ── notifications ─────────────────────────────────────────────────────────

    def push_msg(self, msg: str, timeout: int = 7000) -> str:
        """Push a toast notification to the SiYuan UI.

        Args:
            msg: Message text to display.
            timeout: Display duration in milliseconds (default 7000).

        Returns:
            Message ID string.
        """
        data = self._post("/api/notification/pushMsg", {"msg": msg, "timeout": timeout})
        return (data or {}).get("id", "")

    def push_err_msg(self, msg: str, timeout: int = 7000) -> str:
        """Push an error toast notification to the SiYuan UI.

        Args:
            msg: Error message text to display.
            timeout: Display duration in milliseconds (default 7000).

        Returns:
            Message ID string.
        """
        data = self._post("/api/notification/pushErrMsg", {"msg": msg, "timeout": timeout})
        return (data or {}).get("id", "")

    def append_daily_note_link(
        self,
        notebook: str,
        daily_path: str,
        doc_id: str,
        title: str,
    ) -> None:
        """Append a content-block reference link to the daily note.

        Locates (or creates) the daily note at *daily_path* and appends a
        SiYuan block-ref ``((doc_id "title"))`` as a new paragraph.

        Args:
            notebook: Notebook ID.
            daily_path: Human-readable path of the daily note (e.g.
                ``"/daily note/2026/02/2026-02-28"``).
            doc_id: Block ID of the session document to reference.
            title: Display text for the block reference.
        """
        ids = self.get_doc_ids_by_path(notebook, daily_path)
        if ids:
            daily_doc_id = ids[0]
        else:
            # Create an empty daily note so we can append to it
            daily_doc_id = self.create_doc(notebook, daily_path, "")

        link_md = f'(({doc_id} "{title}"))'
        self.append_block(daily_doc_id, link_md)


# ──────────────────────────────────────────────────────────────────────────────
# Exception
# ──────────────────────────────────────────────────────────────────────────────


class SiyuanError(Exception):
    """Raised when the SiYuan API returns a non-zero response code."""

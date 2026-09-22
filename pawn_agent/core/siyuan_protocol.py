"""SiYuan @pawn protocol helpers (pure logic + client orchestration).

Owns: mention matching, instruction hashing, discovery SQL templates,
attribute keys, result markdown stubs, approval parsing.
Does not own: the watcher loop or queue dispatch.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Attribute keys written onto the @pawn instruction block.
ATTR_STATUS = "custom-agent-status"
ATTR_REQUEST_ID = "custom-agent-request-id"
ATTR_OUTPUT_ID = "custom-agent-output-id"
ATTR_SOURCE_HASH = "custom-agent-source-hash"
ATTR_RUN_ID = "custom-agent-run-id"

STATUSES_TERMINAL = frozenset({"done", "cancelled"})
STATUSES_ACTIVE = frozenset({"queued", "claimed", "running", "review", "blocked"})
STATUSES_SKIP_REDISCOVER = frozenset(
    {"claimed", "running", "review", "done", "blocked", "cancelled"}
)

_MENTION_RE = re.compile(r"^\s*@pawn\b", re.IGNORECASE)
_BLOCK_REF_RE = re.compile(r"\(\(([0-9a-z]{14}-[0-9a-z]{7})(?:\s+\"[^\"]*\")?\)\)")
_APPROVE_CHECKED_RE = re.compile(r"(?im)^\s*[-*]\s*\[[xX]\]\s*Approve\s+for\s+Pawn\s+memory\b")
_TIP_CALLOUT_RE = re.compile(r"(?im)^\s*>\s*\[!TIP\]")
# Statuses that mean "do not start another run for this trigger+hash".
STATUSES_NO_RETRIGGER = frozenset({"queued", "claimed", "running", "review"})


@dataclass(frozen=True)
class DiscoveredPawnBlock:
    """One candidate @pawn block from SiYuan SQL."""

    block_id: str
    parent_id: str
    root_id: str
    notebook_id: str
    content: str
    markdown: str
    updated: str

    @property
    def instruction_text(self) -> str:
        return (self.markdown or self.content or "").strip()


@dataclass(frozen=True)
class ResolvedPawnTrigger:
    """Callout (or plain mention) block resolved for a plugin/API trigger."""

    trigger_block_id: str
    parent_block_id: str
    root_id: str
    notebook_id: str
    instruction_text: str
    source_updated: str


def fetch_block_row(client: Any, block_id: str) -> dict[str, Any] | None:
    """Load one blocks-table row by id via SiYuan SQL."""
    bid = _sql_escape(block_id or "")
    if not bid:
        return None
    stmt = (
        "SELECT id, parent_id, root_id, box, content, markdown, updated "
        f"FROM blocks WHERE id = '{bid}' LIMIT 1"
    )
    try:
        rows = client.query_sql(stmt)
    except Exception:
        return None
    if not rows:
        return None
    row = rows[0]
    return row if isinstance(row, dict) else None


def resolve_pawn_trigger(
    client: Any,
    block_id: str,
    *,
    mention_token: str = "@pawn",
    require_mention: bool = True,
) -> ResolvedPawnTrigger | None:
    """Walk from *block_id* up to the nearest pawn TIP callout (or plain block).

    Prefers a TIP callout ancestor. When *require_mention* is True (watcher /
    legacy), the callout or plain block must contain/start with the mention
    token. When False (plugin Send), any TIP callout is accepted, else the
    starting block itself if it has non-empty content.
    """
    if not (block_id or "").strip():
        return None
    chain: list[tuple[str, dict[str, Any], str]] = []
    seen: set[str] = set()
    current = block_id.strip()
    while current and current not in seen:
        seen.add(current)
        row = fetch_block_row(client, current)
        if row is None:
            break
        kramdown = ""
        try:
            kramdown = client.get_block_kramdown(current) or ""
        except Exception:
            kramdown = ""
        if not kramdown:
            kramdown = str(row.get("markdown") or row.get("content") or "")
        chain.append((current, row, kramdown))
        parent = str(row.get("parent_id") or "").strip()
        root = str(row.get("root_id") or "").strip()
        if not parent or parent == current or current == root:
            break
        current = parent

    def _to_resolved(tid: str, row: dict[str, Any], instruction: str) -> ResolvedPawnTrigger:
        parent = str(row.get("parent_id") or tid).strip() or tid
        root = str(row.get("root_id") or tid).strip() or tid
        notebook = str(row.get("box") or "").strip()
        updated = str(row.get("updated") or "").strip()
        return ResolvedPawnTrigger(
            trigger_block_id=tid,
            parent_block_id=parent,
            root_id=root,
            notebook_id=notebook,
            instruction_text=(instruction or "").strip(),
            source_updated=updated,
        )

    for tid, row, kd in chain:
        if not looks_like_tip_callout(kd):
            continue
        if require_mention and not contains_mention_token(kd, mention_token):
            continue
        return _to_resolved(tid, row, kd)

    if require_mention:
        for tid, row, kd in chain:
            content = str(row.get("content") or "")
            if is_pawn_mention(kd, mention_token) or is_pawn_mention(
                content, mention_token
            ):
                return _to_resolved(tid, row, kd or content)
        return None

    # Plugin Send: use the starting block if it has any non-empty text.
    if chain:
        tid, row, kd = chain[0]
        content = str(row.get("content") or "")
        text = (kd or content).strip()
        if text:
            return _to_resolved(tid, row, kd or content)
    return None


def is_pawn_mention(text: str, mention_token: str = "@pawn") -> bool:
    """True when *text* starts with the mention token (default ``@pawn``)."""
    token = (mention_token or "@pawn").strip()
    if not token:
        return False
    if token.lower() == "@pawn":
        return bool(_MENTION_RE.match(text or ""))
    return bool(re.match(rf"^\s*{re.escape(token)}\b", text or "", re.IGNORECASE))


def contains_mention_token(text: str, mention_token: str = "@pawn") -> bool:
    """True when *text* contains the mention token as a whole word."""
    token = (mention_token or "@pawn").strip()
    if not token:
        return False
    return bool(re.search(rf"(?i)(?:^|[\s>]){re.escape(token)}\b", text or ""))


def looks_like_tip_callout(kramdown: str) -> bool:
    """True when *kramdown* looks like a SiYuan 3.5+ TIP callout."""
    return bool(_TIP_CALLOUT_RE.search(kramdown or ""))


def instruction_hash(text: str) -> str:
    """Stable hash of normalized instruction text."""
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def conversation_id_for_root(root_id: str) -> str:
    """Sallm conversation key for a SiYuan document root."""
    return f"siyuan:{root_id}"


def new_request_id() -> str:
    return str(uuid.uuid4())


def extract_block_refs(text: str) -> list[str]:
    """Return unique SiYuan block ids referenced via ``((id …))`` syntax."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _BLOCK_REF_RE.finditer(text or ""):
        bid = match.group(1)
        if bid not in seen:
            seen.add(bid)
            out.append(bid)
    return out


def build_discovery_sql(
    notebook_ids: Sequence[str],
    *,
    mention_token: str = "@pawn",
    watermark: str = "",
    limit: int = 50,
) -> str:
    """Fixed SELECT template for discovering @pawn blocks.

    SiYuan SQL does not support bound parameters; values are escaped as
    single-quoted string literals.
    """
    if not notebook_ids:
        raise ValueError("notebook_ids must be non-empty")
    boxes = ", ".join(f"'{_sql_escape(b)}'" for b in notebook_ids)
    token = _sql_escape(mention_token or "@pawn")
    lim = max(1, min(int(limit), 200))
    where = [
        f"box IN ({boxes})",
        f"content LIKE '%{token}%'",
    ]
    if watermark:
        where.append(f"updated > '{_sql_escape(watermark)}'")
    return (
        "SELECT id, parent_id, root_id, box, content, markdown, updated "
        "FROM blocks WHERE " + " AND ".join(where) + f" ORDER BY updated ASC LIMIT {lim}"
    )


def parse_discovered_rows(
    rows: Iterable[dict[str, Any]],
    *,
    mention_token: str = "@pawn",
) -> list[DiscoveredPawnBlock]:
    """Filter SQL rows to blocks whose content starts with the mention."""
    out: list[DiscoveredPawnBlock] = []
    for row in rows:
        content = str(row.get("content") or "")
        markdown = str(row.get("markdown") or content)
        if not is_pawn_mention(markdown, mention_token) and not is_pawn_mention(
            content, mention_token
        ):
            continue
        block_id = str(row.get("id") or "")
        if not block_id:
            continue
        parent_id = str(row.get("parent_id") or block_id)
        root_id = str(row.get("root_id") or block_id)
        notebook_id = str(row.get("box") or "")
        updated = str(row.get("updated") or "")
        out.append(
            DiscoveredPawnBlock(
                block_id=block_id,
                parent_id=parent_id or block_id,
                root_id=root_id,
                notebook_id=notebook_id,
                content=content,
                markdown=markdown,
                updated=updated,
            )
        )
    return out


def build_result_markdown(body: str, *, request_id: str) -> str:
    """Append-ready Markdown for a reviewable Pawn result."""
    text = (body or "").strip() or "_No content produced._"
    return (
        f"### Pawn result — ready for review\n\n"
        f"_request: `{request_id}`_\n\n"
        f"{text}\n\n"
        f"- [ ] Approve for Pawn memory\n"
        f"- [ ] Request changes (reply with @pawn …)\n"
    )


def approval_checked(kramdown_or_markdown: str) -> bool:
    """True when the Approve-for-Pawn-memory checkbox is checked."""
    return bool(_APPROVE_CHECKED_RE.search(kramdown_or_markdown or ""))


def strip_mention_prefix(text: str, mention_token: str = "@pawn") -> str:
    """Remove the leading mention token from instruction text."""
    token = (mention_token or "@pawn").strip()
    return re.sub(
        rf"^\s*{re.escape(token)}\b\s*",
        "",
        text or "",
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def strip_mention_from_instruction(text: str, mention_token: str = "@pawn") -> str:
    """Remove mention tokens from instruction text (plain or callout lines).

    Strips a leading mention on the first line, and also ``@pawn`` that appears
    after blockquote markers so multi-block TIP callout kramdown stays readable
    without the trigger token.
    """
    token = (mention_token or "@pawn").strip()
    if not token:
        return (text or "").strip()
    lines: list[str] = []
    for line in (text or "").splitlines():
        lines.append(
            re.sub(
                rf"^(\s*(?:>\s*)*){re.escape(token)}\b\s*",
                r"\1",
                line,
                count=1,
                flags=re.IGNORECASE,
            )
        )
    return "\n".join(lines).strip()


def extract_pawn_callout_title(
    instruction: str,
    *,
    mention_token: str = "@pawn",
    max_len: int = 60,
) -> str:
    """Derive a short TIP callout title from an @pawn instruction.

    Heuristic only (no LLM): first clause after the mention, cut at ``--`` /
    newline, truncated. Falls back to ``Pawn task``.
    """
    body = strip_mention_prefix(instruction, mention_token)
    if not body:
        return "Pawn task"
    body = re.sub(r"\(\([^)]*$", "", body).strip()
    body = re.split(r"\s+--\s+|\n", body, maxsplit=1)[0].strip()
    body = re.sub(r"\s+", " ", body)
    if not body:
        return "Pawn task"
    if len(body) > max_len:
        cut = body[: max_len - 1]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        body = cut.rstrip(".,;:") + "…"
    return body[0].upper() + body[1:]


def build_pawn_tip_callout_markdown(
    instruction: str,
    *,
    title: str | None = None,
    mention_token: str = "@pawn",
    icon: str = "🤖",
) -> str:
    """GitHub-alert Markdown that SiYuan 3.5+ spins into a TIP callout.

    Keeps the instruction body as-is (``@pawn`` optional — plugin Send does
    not require it; the watcher still discovers leading mentions).
    """
    body = (instruction or "").strip() or "Pawn task"
    body_lines = body.splitlines() or [body]
    title_text = (title or extract_pawn_callout_title(body, mention_token=mention_token)).strip()
    title_text = title_text.replace("\n", " ")
    head = f"> [!TIP] {icon} {title_text}".rstrip()
    quoted = "\n".join(f"> {line}" if line else ">" for line in body_lines)
    return f"{head}\n{quoted}\n"


def build_agent_prompt(
    *,
    request_id: str,
    instruction: str,
    parent_excerpt: str,
    ref_excerpts: Sequence[tuple[str, str]],
    parent_block_id: str,
    trigger_block_id: str,
) -> str:
    """Compose the queue prompt for a ``siyuan_run`` turn."""
    lines = [
        "SiYuan @pawn task — fulfill the instruction below.",
        f"request_id: {request_id}",
        f"trigger_block_id: {trigger_block_id}",
        f"parent_block_id: {parent_block_id}",
        "",
        "Rules:",
        "- Use siyuan_read to fetch context and follow explicit ((block refs)).",
        "- Append your draft under parent_block_id with siyuan_append "
        "(include the Approve for Pawn memory checklist).",
        "- Set status to review with siyuan_set_status when done writing.",
        "- For diarization sessions named in the note, use sessions_list / "
        "session_transcript / session_analyze — never invent session ids.",
        "- Do not delete or overwrite human blocks.",
        "- Do not write tool errors into SiYuan.",
        "",
        "## Instruction",
        instruction.strip() or "(empty)",
    ]
    if parent_excerpt.strip():
        lines.extend(["", "## Parent context (excerpt)", parent_excerpt.strip()[:4000]])
    if ref_excerpts:
        lines.append("")
        lines.append("## Linked blocks")
        for bid, excerpt in ref_excerpts:
            lines.append(f"### (({bid}))")
            lines.append((excerpt or "")[:3000])
    return "\n".join(lines)


def _sql_escape(value: str) -> str:
    return (value or "").replace("'", "''")


class InstructionSettleTracker:
    """Debounce @pawn instructions until their text hash is stable.

    SiYuan autosaves while the user types. We only treat an instruction as
    ready after the same content hash has been observed for ``settle_seconds``.
    """

    def __init__(self) -> None:
        # block_id → (instruction_hash, first_seen_monotonic)
        self._seen: dict[str, tuple[str, float]] = {}

    def clear(self) -> None:
        self._seen.clear()

    def is_settled(
        self,
        block_id: str,
        content_hash: str,
        *,
        settle_seconds: float,
        now: float | None = None,
    ) -> bool:
        """Return True when *content_hash* has been stable long enough."""
        import time

        t = time.monotonic() if now is None else now
        settle = max(0.0, float(settle_seconds))
        prev = self._seen.get(block_id)
        if prev is None or prev[0] != content_hash:
            self._seen[block_id] = (content_hash, t)
            return settle <= 0.0
        return (t - prev[1]) >= settle

    def forget(self, block_id: str) -> None:
        self._seen.pop(block_id, None)


def client_from_agent_config(cfg: Any) -> Any:
    """Build a :class:`SiyuanClient` from agent config flat attrs."""
    from pawn_diarize.core.siyuan import SiyuanClient

    return SiyuanClient(
        url=cfg.siyuan_url,
        token=cfg.siyuan_token,
        notebook_id=cfg.siyuan_notebook or "",
    )


def resolve_notebook_allowlist(cfg: Any) -> list[str]:
    """Notebook ids the watcher may scan."""
    watcher = getattr(cfg, "siyuan_watcher", None)
    allowlist: list[str] = []
    if watcher is not None:
        allowlist = list(getattr(watcher, "notebook_allowlist", None) or [])
    if allowlist:
        return [str(x) for x in allowlist if str(x).strip()]
    notebook = getattr(cfg, "siyuan_notebook", "") or ""
    return [notebook] if notebook else []

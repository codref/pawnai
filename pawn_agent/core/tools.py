"""Agent tools for pawn-agent.

Each tool is a plain Python function with typed parameters and a docstring
so DSPy can describe it to the LLM during the ReAct loop.  Tools are wired
into a closure over :class:`~pawn_agent.utils.config.AgentConfig` via
:func:`build_tools` so their signatures stay clean.

Tools
-----
- ``query_conversation`` – fetch and format a transcript from the DB
- ``analyze_conversation`` – LLM analysis of a transcript, stored in DB
- ``save_to_siyuan`` – create/update a document in SiYuan
"""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from typing import Callable, List, Optional

import requests
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy import String, Float, Integer, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB

from pawn_agent.utils.config import AgentConfig


# ──────────────────────────────────────────────────────────────────────────────
# Minimal ORM models (inline, no pawn_diarize dependency)
# ──────────────────────────────────────────────────────────────────────────────


class _Base(DeclarativeBase):
    pass


class _TranscriptionSegment(_Base):
    __tablename__ = "transcription_segments"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    audio_file: Mapped[str] = mapped_column(String, nullable=False)
    original_speaker_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class _SpeakerName(_Base):
    __tablename__ = "speaker_names"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    audio_file: Mapped[str] = mapped_column(String, nullable=False)
    local_speaker_label: Mapped[str] = mapped_column(String, nullable=False)
    speaker_name: Mapped[str] = mapped_column(String, nullable=False)


class _SessionAnalysis(_Base):
    __tablename__ = "session_analysis"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_topics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    speaker_highlights: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment_tags: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    analyzed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: DB session factory
# ──────────────────────────────────────────────────────────────────────────────


def _make_db_session(dsn: str) -> Session:
    engine = create_engine(dsn)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: analysis prompt
# ──────────────────────────────────────────────────────────────────────────────

_ANALYSIS_PROMPT = """\
You are an expert conversation analyst. Below is a speaker-diarized transcript.
Provide a structured analysis with exactly these sections:

## Title
A short, descriptive title (5–10 words).

## Summary
A concise paragraph (10–20 sentences) summarising the main discussion.

## Key Topics / Keywords
Bullet list of the most important topics and concepts.

## Speaker Highlights
For each speaker, one or two sentences on their main contributions.

## Sentiment
Overall tone — one or two descriptive sentences.

## Sentiment Tags
Up to 3 lowercase sentiment labels (comma-separated, single line).

## Tags
5–10 lowercase topic/entity tags (comma-separated, single line).

---
TRANSCRIPT:
{transcript}
---
Respond only with the structured analysis above. Do not repeat the transcript."""


def _parse_sections(text_: str) -> dict:
    """Split LLM analysis output into named sections."""
    section_map = {
        "Title": "title",
        "Summary": "summary",
        "Key Topics / Keywords": "key_topics",
        "Speaker Highlights": "speaker_highlights",
        "Sentiment Tags": "sentiment_tags",
        "Sentiment": "sentiment",
        "Tags": "tags",
    }
    result: dict = {v: None for v in section_map.values()}
    current: Optional[str] = None
    buf: list[str] = []

    def flush():
        if current and buf:
            content = "\n".join(buf).strip()
            key = section_map[current]
            if key in ("sentiment_tags", "tags"):
                # Parse comma-separated tags; strip markdown bold/italic
                raw = re.sub(r"[*_`]", "", content)
                result[key] = [t.strip().lower() for t in raw.split(",") if t.strip()]
            else:
                result[key] = content

    for line in text_.splitlines():
        m = re.match(r"##\s+(.+)", line)
        if m:
            flush()
            buf = []
            current = m.group(1).strip()
        elif current:
            buf.append(line)

    flush()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Helper: SiYuan HTTP calls
# ──────────────────────────────────────────────────────────────────────────────


def _siyuan_post(base_url: str, token: str, endpoint: str, payload: dict) -> dict:
    resp = requests.post(
        f"{base_url.rstrip('/')}{endpoint}",
        json=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Token {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("code", 0) != 0:
        raise RuntimeError(f"SiYuan error [{body.get('code')}]: {body.get('msg')}")
    return body.get("data") or {}


def _resolve_path(template: str, session_id: str, title: Optional[str]) -> str:
    now = datetime.now(timezone.utc)
    slug = re.sub(r"[^\w\-]", "-", (title or session_id).lower()).strip("-")
    return template.format(
        session_id=session_id,
        title=slug,
        date=now.strftime("%Y-%m-%d"),
        year=now.strftime("%Y"),
        month=now.strftime("%m"),
        day=now.strftime("%d"),
    )


def _build_siyuan_markdown(sections: dict, session_id: str, transcript: str, model: str) -> str:
    title = sections.get("title") or "Conversation Analysis"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tags_str = ", ".join(sections.get("tags") or [])
    sentiment_str = ", ".join(sections.get("sentiment_tags") or [])

    header = f"**Session:** `{session_id}` · **Date:** {now_str} · **Model:** {model}"
    if tags_str:
        header += f"\n**Tags:** {tags_str}"
    if sentiment_str:
        header += f"\n**Sentiment tags:** {sentiment_str}"

    parts = [f"# {title}", header, "---"]
    for heading, key in [
        ("Summary", "summary"),
        ("Key Topics / Keywords", "key_topics"),
        ("Speaker Highlights", "speaker_highlights"),
        ("Sentiment", "sentiment"),
    ]:
        val = sections.get(key)
        if val:
            parts.append(f"## {heading}\n\n{val}")
    parts += ["---", f"## Transcript\n\n{transcript or '_No transcript._'}"]
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Tool factory
# ──────────────────────────────────────────────────────────────────────────────


def build_tools(cfg: AgentConfig) -> List[Callable]:
    """Return the list of DSPy-compatible tool functions for the given config.

    Each tool is a plain Python callable with type annotations and a docstring.
    DSPy's ``ReAct`` module uses these attributes to describe the tools to the
    LLM automatically.

    Args:
        cfg: Populated :class:`~pawn_agent.utils.config.AgentConfig`.

    Returns:
        List of callables: ``[query_conversation, analyze_conversation, save_to_siyuan]``.
    """

    # ── Tool 1: query_conversation ────────────────────────────────────────────

    def query_conversation(session_id: str) -> str:
        """Retrieve and return the full transcript for a session from the database.

        Use this tool first when the user asks about the content of a conversation
        or wants to analyse a specific session.

        Args:
            session_id: Unique session identifier stored in the database.

        Returns:
            Formatted transcript text with timestamps and speaker names, or an
            error message if the session is not found.
        """
        try:
            db = _make_db_session(cfg.db_dsn)
            segments = db.scalars(
                select(_TranscriptionSegment)
                .where(_TranscriptionSegment.session_id == session_id)
                .order_by(_TranscriptionSegment.segment_index)
            ).all()

            if not segments:
                return f"No transcript found for session: {session_id!r}"

            # Batch-fetch speaker names
            audio_files = list({s.audio_file for s in segments if s.audio_file})
            labels = list({s.original_speaker_label for s in segments if s.original_speaker_label})
            name_lookup: dict = {}
            if audio_files and labels:
                rows = db.scalars(
                    select(_SpeakerName).where(
                        _SpeakerName.audio_file.in_(audio_files),
                        _SpeakerName.local_speaker_label.in_(labels),
                    )
                ).all()
                name_lookup = {(r.audio_file, r.local_speaker_label): r.speaker_name for r in rows}
            db.close()

            lines: list[str] = []
            current_speaker: Optional[str] = None
            for seg in segments:
                txt = (seg.text or "").strip()
                if not txt:
                    continue
                mm = int(seg.start_time // 60)
                ss = seg.start_time % 60
                raw_label = seg.original_speaker_label
                display = name_lookup.get((seg.audio_file, raw_label), raw_label or "Speaker")
                if display != current_speaker:
                    if current_speaker is not None:
                        lines.append("")
                    lines.append(f"[{mm:02d}:{ss:05.2f}] {display}:")
                    current_speaker = display
                lines.append(f"  {txt}")

            return "\n".join(lines)
        except Exception as exc:
            return f"Error retrieving transcript: {exc}"

    # ── Tool 2: analyze_conversation ─────────────────────────────────────────

    def analyze_conversation(session_id: str) -> str:
        """Analyse a conversation session and save the analysis to the database.

        Fetches the transcript from the database, sends it to the LLM for a
        structured analysis (title, summary, key topics, speaker highlights,
        sentiment, tags), persists the result in the ``session_analysis`` table,
        and returns the full analysis text.

        Args:
            session_id: Unique session identifier stored in the database.

        Returns:
            Full structured analysis in Markdown, or an error message if the
            session cannot be found or analysed.
        """
        try:
            # Fetch transcript
            transcript = query_conversation(session_id)
            if transcript.startswith("Error") or transcript.startswith("No transcript"):
                return transcript

            # Build and call the LLM via the currently configured DSPy LM
            import dspy  # noqa: PLC0415
            lm = dspy.settings.lm
            prompt = _ANALYSIS_PROMPT.format(transcript=transcript)
            completions = lm(messages=[{"role": "user", "content": prompt}])
            analysis_text = completions[0] if isinstance(completions, list) else str(completions)

            sections = _parse_sections(analysis_text)

            # Persist to DB
            db = _make_db_session(cfg.db_dsn)
            row = _SessionAnalysis(
                id=str(uuid.uuid4()),
                session_id=session_id,
                source=f"session:{session_id}",
                model=getattr(dspy.settings.lm, "_copilot_model", getattr(dspy.settings.lm, "model", "unknown")),
                title=sections.get("title"),
                summary=sections.get("summary"),
                key_topics=sections.get("key_topics"),
                speaker_highlights=sections.get("speaker_highlights"),
                sentiment=sections.get("sentiment"),
                sentiment_tags=sections.get("sentiment_tags"),
                tags=sections.get("tags"),
                analyzed_at=datetime.now(timezone.utc),
            )
            db.add(row)
            db.commit()
            db.close()

            return analysis_text
        except Exception as exc:
            return f"Error analysing conversation: {exc}"

    # ── Tool 3: save_to_siyuan ────────────────────────────────────────────────

    def save_to_siyuan(session_id: str, title: str, content: str) -> str:
        """Save a conversation analysis to SiYuan Notes as a structured document.

        Creates or replaces a document in SiYuan under the configured notebook
        and path template.  Also appends a backlink to today's daily note.

        Args:
            session_id: Session identifier used to resolve the document path.
            title: Document title (used in the path and as the page heading).
            content: Full Markdown content to store (e.g. from analyze_conversation).

        Returns:
            The SiYuan block ID of the created document, or an error message.
        """
        try:
            url = cfg.siyuan_url
            token = cfg.siyuan_token
            notebook = cfg.siyuan_notebook

            if not notebook:
                return "SiYuan notebook ID is not configured (set siyuan.notebook in config)."

            path = _resolve_path(cfg.siyuan_path_template, session_id, title)

            # Remove existing doc at same path if any
            try:
                ids_data = _siyuan_post(url, token, "/api/filetree/getIDsByHPath",
                                        {"path": path, "notebook": notebook})
                for doc_id in (ids_data if isinstance(ids_data, list) else []):
                    _siyuan_post(url, token, "/api/filetree/removeDocByID", {"id": doc_id})
            except Exception:
                pass  # path may not exist yet

            doc_id = _siyuan_post(url, token, "/api/filetree/createDocWithMd",
                                  {"notebook": notebook, "path": path, "markdown": content})

            # Set custom block attributes for querying
            if doc_id:
                try:
                    _siyuan_post(url, token, "/api/attr/setBlockAttrs",
                                 {"id": doc_id, "attrs": {"custom-session-id": session_id}})
                except Exception:
                    pass

            # Append daily note backlink
            try:
                daily_path = _resolve_path(cfg.siyuan_daily_template, session_id, None)
                daily_ids = _siyuan_post(url, token, "/api/filetree/getIDsByHPath",
                                         {"path": daily_path, "notebook": notebook})
                if isinstance(daily_ids, list) and daily_ids:
                    daily_doc_id = daily_ids[0]
                else:
                    daily_doc_id = _siyuan_post(url, token, "/api/filetree/createDocWithMd",
                                                {"notebook": notebook, "path": daily_path, "markdown": ""})
                if daily_doc_id and doc_id:
                    _siyuan_post(url, token, "/api/block/appendBlock",
                                 {"dataType": "markdown", "data": f'(({doc_id} "{title}"))',
                                  "parentID": daily_doc_id})
            except Exception:
                pass  # daily note link is best-effort

            return str(doc_id) if doc_id else "Document created (no ID returned)."
        except Exception as exc:
            return f"Error saving to SiYuan: {exc}"

    return [query_conversation, analyze_conversation, save_to_siyuan]

"""Agent tools for pawn-agent.

Tools are registered with the Copilot SDK via :func:`~copilot.define_tool`.
Each tool function takes a Pydantic params model and returns a plain string.
They are wired into a closure over :class:`~pawn_agent.utils.config.AgentConfig`
and a shared :class:`~copilot.CopilotClient` via :func:`build_tools`.

Tools
-----
- ``query_conversation``  – fetch and format a transcript from the DB
- ``analyze_conversation`` – standard structured analysis (title/summary/topics/sentiment), stored in DB
- ``analyze_custom``      – free-form analysis of a transcript grounded exclusively on its content
- ``save_to_siyuan``      – create/update a document in SiYuan
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import requests
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from copilot import CopilotClient, MessageOptions, PermissionHandler, Tool, define_tool

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


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic params models
# ──────────────────────────────────────────────────────────────────────────────


class QueryConversationParams(BaseModel):
    session_id: str = Field(description="Unique session identifier stored in the database.")


class AnalyzeConversationParams(BaseModel):
    session_id: str = Field(description="Unique session identifier stored in the database.")


class AnalyzeCustomParams(BaseModel):
    session_id: str = Field(description="Unique session identifier stored in the database.")
    instruction: str = Field(
        description="The analysis task to perform on the transcript, e.g. "
                    "'extract epics and user stories', 'list action items', "
                    "'identify decisions made'."
    )


class AnalyzeAndSaveCustomParams(BaseModel):
    session_id: str = Field(description="Unique session identifier stored in the database.")
    instruction: str = Field(
        description="The analysis task to perform on the transcript, e.g. "
                    "'extract epics and user stories', 'list action items'."
    )
    title: str = Field(description="Document title used in the SiYuan path and as the page heading.")
    path: Optional[str] = Field(
        default=None,
        description="Optional explicit SiYuan path (e.g. '/Notes/2026-03-13/epics'). "
                    "When the user asks to save to a *new* or *specific* note, derive a "
                    "unique path from the title and today's date.",
    )


class SaveToSiyuanParams(BaseModel):
    session_id: str = Field(description="Session identifier used to resolve the document path.")
    title: str = Field(description="Document title used in the path and as the page heading.")
    content: str = Field(description="Full Markdown content to store (e.g. from analyze_conversation).")
    path: Optional[str] = Field(
        default=None,
        description="Optional explicit SiYuan path (e.g. '/Notes/2026-03-13/epics'). "
                    "When provided it overrides the configured path template so that each "
                    "call creates a distinct document. Use this whenever the user asks to "
                    "save to a *new* note or a specific location.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tool factory
# ──────────────────────────────────────────────────────────────────────────────


def build_tools(cfg: AgentConfig, client: CopilotClient) -> List[Tool]:
    """Return the list of Copilot SDK :class:`~copilot.Tool` objects for the given config.

    Each tool is defined with :func:`~copilot.define_tool` and a Pydantic params
    model so the SDK can auto-generate a JSON schema for the LLM.

    Args:
        cfg: Populated :class:`~pawn_agent.utils.config.AgentConfig`.
        client: Active :class:`~copilot.CopilotClient` (reused for the analysis
            LLM call inside ``analyze_conversation``).

    Returns:
        List of :class:`~copilot.Tool` objects.
    """

    # ── Shared internal helper: siyuan save (not a tool) ─────────────────────

    def _do_save_to_siyuan(session_id: str, title: str, content: str, path: Optional[str]) -> str:
        url = cfg.siyuan_url
        token = cfg.siyuan_token
        notebook = cfg.siyuan_notebook
        if not notebook:
            return "SiYuan notebook ID is not configured (set siyuan.notebook in config)."
        resolved_path = path or _resolve_path(cfg.siyuan_path_template, session_id, title)
        try:
            ids_data = _siyuan_post(url, token, "/api/filetree/getIDsByHPath",
                                    {"path": resolved_path, "notebook": notebook})
            for doc_id in (ids_data if isinstance(ids_data, list) else []):
                _siyuan_post(url, token, "/api/filetree/removeDocByID", {"id": doc_id})
        except Exception:
            pass
        doc_id = _siyuan_post(url, token, "/api/filetree/createDocWithMd",
                              {"notebook": notebook, "path": resolved_path, "markdown": content})
        if doc_id:
            try:
                _siyuan_post(url, token, "/api/attr/setBlockAttrs",
                             {"id": doc_id, "attrs": {"custom-session-id": session_id}})
            except Exception:
                pass
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
            pass
        return str(doc_id) if doc_id else "Document created (no ID returned)."

    # ── Private helper (shared by query and analyze tools) ───────────────────

    def _fetch_transcript(session_id: str) -> str:
        """Fetch and format a transcript from the database (no tool decorator)."""
        try:
            db = _make_db_session(cfg.db_dsn)
            segments = db.scalars(
                select(_TranscriptionSegment)
                .where(_TranscriptionSegment.session_id == session_id)
                .order_by(_TranscriptionSegment.segment_index)
            ).all()

            if not segments:
                return f"No transcript found for session: {session_id!r}"

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

    # ── Tool 1: query_conversation ────────────────────────────────────────────

    @define_tool(
        description=(
            "Retrieve and return the full transcript for a session from the database. "
            "Use this first when the user asks about the content of a conversation "
            "or wants to analyse a specific session."
        )
    )
    def query_conversation(params: QueryConversationParams) -> str:
        return _fetch_transcript(params.session_id)

    # ── Tool 2: analyze_conversation ─────────────────────────────────────────

    @define_tool(
        description=(
            "Run the STANDARD structured analysis on a conversation session and persist it to "
            "the database. Produces exactly these fixed sections: Title, Summary, Key Topics, "
            "Speaker Highlights, Sentiment, Sentiment Tags, Tags. "
            "Use this ONLY when the user explicitly asks for a full conversation analysis or "
            "summary. For any other task (epics, user stories, action items, decisions, risks, "
            "custom extraction, etc.) use analyze_custom instead."
        )
    )
    async def analyze_conversation(params: AnalyzeConversationParams) -> str:
        session_id = params.session_id
        try:
            transcript = _fetch_transcript(session_id)
            if transcript.startswith("Error") or transcript.startswith("No transcript"):
                return transcript

            prompt = _ANALYSIS_PROMPT.format(transcript=transcript)

            # Use a dedicated Copilot session for the analysis LLM call
            analysis_session = await client.create_session(
                {
                    "model": cfg.model,
                    "on_permission_request": PermissionHandler.approve_all,
                    "system_message": {
                        "mode": "replace",
                        "content": (
                            "You are an expert conversation analyst. "
                            "Analyse only the transcript provided in the user message. "
                            "Do not use any knowledge about the user's codebase, tools, "
                            "or environment."
                        ),
                    },
                }
            )
            try:
                response = await analysis_session.send_and_wait(
                    MessageOptions(prompt=prompt),
                    timeout=120,
                )
                analysis_text = response.data.content if (response is not None and response.data.content is not None) else ""
            finally:
                await analysis_session.disconnect()

            analysis_text = analysis_text or ""
            sections = _parse_sections(analysis_text)

            # Persist to DB
            db = _make_db_session(cfg.db_dsn)
            row = _SessionAnalysis(
                id=str(uuid.uuid4()),
                session_id=session_id,
                source=f"session:{session_id}",
                model=cfg.model,
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

    # ── Tool 3: analyze_custom ────────────────────────────────────────────────

    @define_tool(
        description=(
            "Perform a custom analysis of a conversation session's transcript and return "
            "the result. Use this for ANY request that is not the standard structured "
            "analysis (title/summary/topics/sentiment), such as extracting epics, user "
            "stories, action items, decisions, risks, or any other bespoke task. "
            "The analysis is grounded EXCLUSIVELY on the stored transcript — no other "
            "context is used. "
            "IMPORTANT: if the user also wants the result saved to SiYuan, use "
            "analyze_and_save_custom instead of this tool — that avoids passing the "
            "content back through this session."
        )
    )
    async def analyze_custom(params: AnalyzeCustomParams) -> str:
        session_id = params.session_id
        instruction = params.instruction
        try:
            transcript = _fetch_transcript(session_id)
            if transcript.startswith("Error") or transcript.startswith("No transcript"):
                return transcript

            prompt = (
                "You are a helpful assistant. Your task is to analyse the conversation "
                "transcript below and nothing else. Do NOT use any knowledge outside of "
                "this transcript.\n"
                "\n"
                f"Task: {instruction}\n"
                "\n"
                "---\n"
                "TRANSCRIPT:\n"
                f"{transcript}\n"
                "---\n"
                "Respond only with the result of the task above. "
                "Do not repeat the transcript or the instructions."
            )

            analysis_session = await client.create_session(
                {
                    "model": cfg.model,
                    "on_permission_request": PermissionHandler.approve_all,
                    "system_message": {
                        "mode": "replace",
                        "content": (
                            "You are a helpful assistant. "
                            "Analyse only the transcript provided in the user message. "
                            "Do not use any knowledge about the user's codebase, tools, "
                            "or environment."
                        ),
                    },
                }
            )
            try:
                response = await analysis_session.send_and_wait(
                    MessageOptions(prompt=prompt),
                    timeout=120,
                )
                return response.data.content if (response is not None and response.data.content is not None) else ""
            finally:
                await analysis_session.disconnect()
        except Exception as exc:
            return f"Error performing custom analysis: {exc}"

    # ── Tool 4: analyze_and_save_custom ──────────────────────────────────────

    @define_tool(
        description=(
            "Perform a custom analysis of a session transcript AND save the result "
            "directly to SiYuan Notes in one step, WITHOUT passing the content back "
            "through this session. "
            "USE THIS whenever the user asks to both analyse a transcript AND save/store "
            "the result (e.g. 'extract epics and save to a new note', 'list action items "
            "and put them in SiYuan'). "
            "The analysis is grounded EXCLUSIVELY on the stored transcript — no other "
            "context is used. Provide 'path' with a unique value (e.g. "
            "'/Notes/YYYY-MM-DD/title-slug') so a new document is created each time."
        )
    )
    async def analyze_and_save_custom(params: AnalyzeAndSaveCustomParams) -> str:
        session_id = params.session_id
        instruction = params.instruction
        title = params.title
        try:
            transcript = _fetch_transcript(session_id)
            if transcript.startswith("Error") or transcript.startswith("No transcript"):
                return transcript

            prompt = (
                "You are a helpful assistant. Your task is to analyse the conversation "
                "transcript below and nothing else. Do NOT use any knowledge outside of "
                "this transcript.\n"
                "\n"
                f"Task: {instruction}\n"
                "\n"
                "---\n"
                "TRANSCRIPT:\n"
                f"{transcript}\n"
                "---\n"
                "Respond only with the result of the task above. "
                "Do not repeat the transcript or the instructions."
            )

            analysis_session = await client.create_session(
                {
                    "model": cfg.model,
                    "on_permission_request": PermissionHandler.approve_all,
                    "system_message": {
                        "mode": "replace",
                        "content": (
                            "You are a helpful assistant. "
                            "Analyse only the transcript provided in the user message. "
                            "Do not use any knowledge about the user's codebase, tools, "
                            "or environment."
                        ),
                    },
                }
            )
            try:
                response = await analysis_session.send_and_wait(
                    MessageOptions(prompt=prompt),
                    timeout=120,
                )
                content = response.data.content if (response is not None and response.data.content is not None) else ""
            finally:
                await analysis_session.disconnect()

            doc_id = _do_save_to_siyuan(session_id, title, content, params.path)
            return f"Saved to SiYuan (doc id: {doc_id})"
        except Exception as exc:
            return f"Error in analyze_and_save_custom: {exc}"

    # ── Tool 5: save_to_siyuan ────────────────────────────────────────────────

    @define_tool(
        description=(
            "Save already-generated content to SiYuan Notes as a structured document. "
            "Use this ONLY when you already have the final content as a string (e.g. "
            "from analyze_conversation). If you still need to run an analysis first, "
            "use analyze_and_save_custom instead so the content never passes through "
            "this session. "
            "Pass 'path' to specify an explicit document path so a new document is "
            "created each time; otherwise the configured path template is used."
        )
    )
    def save_to_siyuan(params: SaveToSiyuanParams) -> str:
        session_id = params.session_id
        title = params.title
        content = params.content
        try:
            return _do_save_to_siyuan(session_id, title, content, params.path)
        except Exception as exc:
            return f"Error saving to SiYuan: {exc}"

    return [query_conversation, analyze_conversation, analyze_custom, analyze_and_save_custom, save_to_siyuan]  # type: ignore[list-item]

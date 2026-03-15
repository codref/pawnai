"""Transcript fetching and analysis prompt helpers."""

from __future__ import annotations

import re
from typing import Optional

from sqlalchemy import select

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.db import make_db_session, TranscriptionSegment, SpeakerName


ANALYSIS_PROMPT = """\
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


def parse_sections(text_: str) -> dict:
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


def fetch_transcript(cfg: AgentConfig, session_id: str) -> str:
    """Fetch and format a transcript from the database."""
    try:
        db = make_db_session(cfg.db_dsn)
        segments = db.scalars(
            select(TranscriptionSegment)
            .where(TranscriptionSegment.session_id == session_id)
            .order_by(TranscriptionSegment.segment_index)
        ).all()

        if not segments:
            return f"No transcript found for session: {session_id!r}"

        audio_files = list({s.audio_file for s in segments if s.audio_file})
        labels = list({s.original_speaker_label for s in segments if s.original_speaker_label})
        name_lookup: dict = {}
        if audio_files and labels:
            rows = db.scalars(
                select(SpeakerName).where(
                    SpeakerName.audio_file.in_(audio_files),
                    SpeakerName.local_speaker_label.in_(labels),
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

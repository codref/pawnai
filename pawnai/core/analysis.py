"""Analysis engine for transcript summarization and keyword extraction using GitHub Copilot."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from copilot import CopilotClient, SessionConfig, MessageOptions

from .combined import transcribe_with_diarization, format_transcript_with_speakers
from .database import get_engine, get_session, init_db, save_session_analysis, SpeakerName, TranscriptionSegment


class AnalysisEngine:
    """Engine for analyzing transcripts using GitHub Copilot."""

    def __init__(self, model: str = "gpt-4o"):
        """Initialize the analysis engine.

        Args:
            model: Copilot model to use for analysis (default: gpt-4o)
        """
        self.model = model

    def _extract_transcript_from_json(self, data: Dict[str, Any]) -> str:
        """Extract a readable transcript from a JSON data structure.

        Args:
            data: Dict with 'segments' or 'text' keys

        Returns:
            Plain-text transcript string

        Raises:
            ValueError: If JSON structure is unrecognized
        """
        if "segments" in data:
            lines = []
            for seg in data["segments"]:
                speaker = seg.get("speaker", "Speaker")
                text = seg.get("text", "").strip()
                start = seg.get("start", 0)
                mm = int(start // 60)
                ss = start % 60

                if text:
                    lines.append(f"[{mm:02d}:{ss:05.2f}] {speaker}: {text}")
                else:
                    # Diarization-only JSON (no text) – just note the segment
                    duration = seg.get("duration", seg.get("end", 0) - start)
                    lines.append(
                        f"[{mm:02d}:{ss:05.2f}] {speaker} speaks for {duration:.1f}s"
                    )
            return "\n".join(lines)
        elif "text" in data:
            return data["text"]
        else:
            raise ValueError("Unrecognised JSON structure (no 'segments' or 'text' key)")

    def _load_transcript_from_db(
        self,
        session_id: str,
        db_dsn: str,
    ) -> str:
        """Load and format a transcript from the database for a given session.

        Segments are ordered by ``segment_index``.  Speaker labels are resolved
        to human-assigned names from the ``speaker_names`` table where
        available, falling back to the raw pyannote label or ``"Speaker"``.

        Args:
            session_id: Session identifier matching ``TranscriptionSegment.session_id``.
            db_dsn: PostgreSQL DSN used to connect to the database.

        Returns:
            Multi-line transcript string in the form
            ``"[MM:SS.ss] SpeakerName: text"``.

        Raises:
            ValueError: If no segments exist for the given session_id.
        """
        engine = get_engine(db_dsn)
        with get_session(engine) as db:
            # --- Load all segments for the session, ordered by position -------
            orm_segments = db.scalars(
                select(TranscriptionSegment)
                .where(TranscriptionSegment.session_id == session_id)
                .order_by(TranscriptionSegment.segment_index)
            ).all()

            if not orm_segments:
                raise ValueError(f"No segments found for session: {session_id!r}")

            # Snapshot to plain dicts while the session is still open
            segments = [
                {
                    "audio_file": s.audio_file,
                    "label": s.original_speaker_label,
                    "start_time": s.start_time,
                    "text": s.text or "",
                }
                for s in orm_segments
            ]

            # --- Batch-fetch human-assigned names (single query) --------------
            pairs = {
                (s["audio_file"], s["label"])
                for s in segments
                if s["label"] is not None
            }
            name_lookup: Dict[Tuple[str, str], str] = {}
            if pairs:
                audio_files = list({p[0] for p in pairs})
                labels = list({p[1] for p in pairs})
                rows = db.scalars(
                    select(SpeakerName).where(
                        SpeakerName.audio_file.in_(audio_files),
                        SpeakerName.local_speaker_label.in_(labels),
                    )
                ).all()
                name_lookup = {
                    (r.audio_file, r.local_speaker_label): r.speaker_name
                    for r in rows
                }

        # --- Format segments (plain dicts, no live ORM objects) --------------
        lines: List[str] = []
        for seg in segments:
            mm = int(seg["start_time"] // 60)
            ss = seg["start_time"] % 60
            if seg["label"] is not None:
                display = name_lookup.get((seg["audio_file"], seg["label"]), seg["label"])
            else:
                display = "Speaker"
            lines.append(f"[{mm:02d}:{ss:05.2f}] {display}: {seg['text'].strip()}")

        return "\n".join(lines)

    def _load_transcript(
        self,
        input_path: str,
        db_dsn: str = "speakers_db",
        device: str = "cuda",
        *,
        session_id: Optional[str] = None,
    ) -> str:
        """Load or generate a transcript from various input formats.

        Supports:
        - session_id: Load stored segments directly from the PostgreSQL database
        - .json: Output from diarize/transcribe-diarize commands
        - .txt/.md: Plain text transcripts
        - Audio files: Transcribes and diarizes on-the-fly

        Args:
            input_path: Path to input file (ignored when *session_id* is given).
            db_dsn: PostgreSQL DSN for the speaker database.
            device: Device for audio processing ("cuda" or "cpu").
            session_id: When provided, transcripts are loaded from the database
                instead of from a file.  *input_path* is not used.

        Returns:
            Plain-text transcript

        Raises:
            FileNotFoundError: If file does not exist (file-based paths only).
            ValueError: If content cannot be extracted, or *session_id* is given
                but has no stored segments.
        """
        # --- Database path (session_id takes precedence over file) ------------
        if session_id is not None:
            return self._load_transcript_from_db(session_id, db_dsn)
        input_file = Path(input_path)

        if not input_file.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        suffix = input_file.suffix.lower()

        if suffix == ".json":
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._extract_transcript_from_json(data)

        elif suffix in (".txt", ".md"):
            with open(input_file, "r", encoding="utf-8") as f:
                return f.read()

        elif suffix in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"):
            # Audio file – transcribe and diarize
            result = transcribe_with_diarization(
                input_path,
                db_dsn=db_dsn,
                device=device,
            )
            return format_transcript_with_speakers(result, include_timestamps=True)

        else:
            # Fall back: try to read as plain text
            with open(input_file, "r", encoding="utf-8") as f:
                return f.read()

    async def _send_to_copilot(self, prompt: str) -> str:
        """Send a prompt to GitHub Copilot and wait for response.

        Args:
            prompt: The prompt to send

        Returns:
            The Copilot response text

        Raises:
            RuntimeError: If Copilot returns empty response or fails
        """
        client = CopilotClient()
        try:
            await client.start()
            session = await client.create_session(SessionConfig(model=self.model))
            response = await session.send_and_wait(MessageOptions(prompt=prompt))
            await session.destroy()

            if response and response.data.content:
                return response.data.content
            raise RuntimeError("Copilot returned empty response")

        finally:
            await client.stop()

    def analyze(self, transcript_text: str) -> str:
        """Analyze a transcript using GitHub Copilot.

        Provides a structured analysis including:
        - Summary (3–5 sentences)
        - Key Topics / Keywords
        - Speaker Highlights
        - Overall Sentiment

        Args:
            transcript_text: The transcript to analyze

        Returns:
            Structured analysis text from Copilot

        Raises:
            ValueError: If transcript is empty
            RuntimeError: If Copilot call fails
        """
        if not transcript_text.strip():
            raise ValueError("Transcript is empty")

        prompt = f"""You are an expert conversation analyst. Below is a speaker-diarized transcript. \
Please provide a structured analysis with the following sections:

## Title
A short, descriptive title for the conversation (5–10 words).

## Summary
A concise paragraph (10–20 sentences) summarising the main discussion.

## Key Topics / Keywords
A bullet list of the most important topics, concepts, and keywords extracted from the transcript.

## Speaker Highlights
For each speaker, one or two sentences describing their main contributions or talking points.

## Sentiment
Overall tone of the conversation — write one or two descriptive sentences.

## Sentiment Tags
Up to 3 short, lowercase sentiment labels that classify the tone for grouping and filtering
(e.g., collaborative, tense, formal, casual, informative, confrontational).
Provide as a comma-separated list on a single line. Maximum 3 labels.

## Tags
5–10 short, lowercase tags covering the key topics, entities, speaker names, and tone of the
conversation. Provide as a comma-separated list on a single line.

---
TRANSCRIPT:
{transcript_text}
---

Respond only with the structured analysis above. Do not repeat the transcript."""

        return asyncio.run(self._send_to_copilot(prompt))

    def extract_graph(self, transcript_text: str) -> List[Tuple[str, str, str]]:
        """Extract a knowledge graph as (subject, relation, object) tuples.

        Sends the transcript to GitHub Copilot and asks it to identify
        relationships between entities, concepts, and speakers. Returns
        a list of triples suitable for constructing a knowledge graph.

        Args:
            transcript_text: The transcript to analyze

        Returns:
            List of (subject, relation, object) tuples

        Raises:
            ValueError: If transcript is empty or response cannot be parsed
            RuntimeError: If Copilot call fails
        """
        if not transcript_text.strip():
            raise ValueError("Transcript is empty")

        prompt = f"""You are a knowledge graph extraction expert. \
Analyze the speaker-diarized transcript below and extract relationships between \
entities (people, organizations, topics, concepts, places, events, etc.).

Return ONLY a valid JSON array of triples. Each triple must be a 3-element array:
  ["subject", "relation", "object"]

Rules:
- Extract factual, meaningful relationships explicitly stated or strongly implied
- Use short, normalized labels (e.g. "John" not "John said that")
- Use active-voice relation verbs (e.g. "works at", "mentioned", "disagrees with", "is part of")
- Include speaker relationships (e.g. ["Alice", "discussed", "budget cuts"])
- Return 10–40 triples depending on content richness
- Do NOT include any explanation, markdown, or text outside the JSON array

Example output format:
[["Alice", "works at", "Acme Corp"], ["Bob", "disagrees with", "Alice"], ["Acme Corp", "is located in", "Berlin"]]

---
TRANSCRIPT:
{transcript_text}
---"""

        raw = asyncio.run(self._send_to_copilot(prompt))

        # Strip markdown code fences if Copilot wraps the JSON
        cleaned = re.sub(r"^```[^\n]*\n", "", raw.strip())
        cleaned = re.sub(r"\n```$", "", cleaned.strip())

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Copilot returned non-JSON graph data: {exc}\nRaw response: {raw[:300]}"
            ) from exc

        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of triples")

        triples: List[Tuple[str, str, str]] = []
        for item in data:
            if isinstance(item, list) and len(item) == 3:
                triples.append((str(item[0]), str(item[1]), str(item[2])))

        return triples

    def extract_graph_from_file(
        self,
        input_path: str,
        db_dsn: str = "speakers_db",
        device: str = "cuda",
        *,
        session_id: Optional[str] = None,
    ) -> List[Tuple[str, str, str]]:
        """Extract a knowledge graph from a file or database session.

        Convenience method combining transcript loading and graph extraction.

        Args:
            input_path: Path to input (JSON, TXT, or audio file).  Ignored
                when *session_id* is provided.
            db_dsn: PostgreSQL DSN for speaker database.
            device: Device for audio processing ("cuda" or "cpu").
            session_id: When provided, transcripts are loaded from the database
                instead of from a file.

        Returns:
            List of (subject, relation, object) tuples

        Raises:
            FileNotFoundError: If file does not exist (file path only).
            ValueError: If content cannot be extracted or parsed.
            RuntimeError: If Copilot call fails.
        """
        transcript = self._load_transcript(input_path, db_dsn, device, session_id=session_id)
        return self.extract_graph(transcript)

    @staticmethod
    def _split_tags(raw: Optional[str]) -> Optional[List[str]]:
        """Split a comma-separated tag string into a cleaned list.

        Strips markdown formatting (backticks, bold/italic markers, newlines)
        that the model may wrap around the tag list before splitting.

        Returns ``None`` when *raw* is ``None`` or blank.
        """
        if not raw:
            return None
        # Strip enclosing backtick fences (e.g. `tag1, tag2`) and whitespace
        cleaned = raw.strip().strip("`").strip()
        # Remove inline markdown: **bold**, *italic*, __bold__, _italic_
        cleaned = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", cleaned)
        # Collapse newlines/extra whitespace so multi-line lists split cleanly
        cleaned = re.sub(r"\s*\n\s*", ", ", cleaned)
        tags = [
            re.sub(r"[`*_]", "", t).strip().lower()
            for t in cleaned.split(",")
            if t.strip()
        ]
        return tags if tags else None

    def _parse_sections(self, analysis_text: str) -> Dict[str, Any]:
        """Parse structured Copilot analysis text into its constituent sections.

        Expects Markdown headings of the form ``## Title``, ``## Summary``,
        ``## Key Topics / Keywords``, ``## Speaker Highlights``,
        ``## Sentiment``, ``## Sentiment Tags``, and ``## Tags``.
        Missing sections are returned as ``None``.

        Args:
            analysis_text: Raw Markdown string returned by :meth:`analyze`.

        Returns:
            Dict with keys: ``title``, ``summary``, ``key_topics``,
            ``speaker_highlights``, ``sentiment`` (all ``Optional[str]``),
            plus ``sentiment_tags`` and ``tags`` (both ``Optional[List[str]]``).
        """
        result: Dict[str, Any] = {
            "title": None,
            "summary": None,
            "key_topics": None,
            "speaker_highlights": None,
            "sentiment": None,
            "sentiment_tags": None,
            "tags": None,
        }
        # Split on any '## ' heading; each block starts with the heading text
        blocks = re.split(r"(?m)^## ", analysis_text)
        for block in blocks:
            if not block.strip():
                continue
            first_line, _, body = block.partition("\n")
            heading = first_line.strip().lower()
            content = body.strip() or None
            if heading == "title":
                result["title"] = content
            elif heading == "summary":
                result["summary"] = content
            elif "key topics" in heading or "keywords" in heading:
                result["key_topics"] = content
            elif "speaker" in heading:
                result["speaker_highlights"] = content
            elif heading == "sentiment tags" or "sentiment tag" in heading:
                result["sentiment_tags"] = self._split_tags(content)
            elif "sentiment" in heading:
                # Match plain '## Sentiment' after the more specific tag check
                result["sentiment"] = content
            elif heading == "tags":
                result["tags"] = self._split_tags(content)
        return result

    def analyze_from_file(
        self,
        input_path: str,
        db_dsn: str = "speakers_db",
        device: str = "cuda",
        *,
        session_id: Optional[str] = None,
    ) -> str:
        """Analyze content from a file or database session.

        Runs the Copilot analysis, parses the structured response into its
        five sections (Title, Summary, Key Topics, Speaker Highlights,
        Sentiment), persists the result to the ``session_analysis`` database
        table, and returns the raw analysis Markdown string.

        Args:
            input_path: Path to input (JSON, TXT, or audio file).  Ignored
                when *session_id* is provided.
            db_dsn: PostgreSQL DSN for speaker database.
            device: Device for audio processing ("cuda" or "cpu").
            session_id: When provided, transcripts are loaded from the database
                instead of from a file.

        Returns:
            Structured analysis text (raw Markdown from Copilot).

        Raises:
            FileNotFoundError: If file does not exist (file path only).
            ValueError: If content cannot be extracted or is empty.
            RuntimeError: If Copilot call fails.
        """
        transcript = self._load_transcript(input_path, db_dsn, device, session_id=session_id)
        raw = self.analyze(transcript)

        # Parse the five sections and persist to DB
        sections = self._parse_sections(raw)
        source = f"session:{session_id}" if session_id else input_path
        engine = get_engine(db_dsn)
        init_db(engine)
        save_session_analysis(
            session_id=session_id,
            source=source,
            model=self.model,
            title=sections.get("title"),
            summary=sections.get("summary"),
            key_topics=sections.get("key_topics"),
            speaker_highlights=sections.get("speaker_highlights"),
            sentiment=sections.get("sentiment"),
            sentiment_tags=sections.get("sentiment_tags"),
            tags=sections.get("tags"),
            engine=engine,
        )

        return raw

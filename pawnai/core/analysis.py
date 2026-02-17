"""Analysis engine for transcript summarization and keyword extraction using GitHub Copilot."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from copilot import CopilotClient, SessionConfig, MessageOptions

from .combined import transcribe_with_diarization, format_transcript_with_speakers


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

    def _load_transcript(
        self,
        input_path: str,
        db_path: str = "speakers_db",
        device: str = "cuda",
    ) -> str:
        """Load or generate a transcript from various input formats.

        Supports:
        - .json: Output from diarize/transcribe-diarize commands
        - .txt/.md: Plain text transcripts
        - Audio files: Transcribes and diarizes on-the-fly

        Args:
            input_path: Path to input file
            db_path: Path to speaker database (for audio processing)
            device: Device for audio processing ("cuda" or "cpu")

        Returns:
            Plain-text transcript

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If content cannot be extracted
        """
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
                db_path=db_path,
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

## Summary
A concise paragraph (3–5 sentences) summarising the main discussion.

## Key Topics / Keywords
A bullet list of the most important topics, concepts, and keywords extracted from the transcript.

## Speaker Highlights
For each speaker, one or two sentences describing their main contributions or talking points.

## Sentiment
Overall tone of the conversation (e.g., collaborative, tense, informative, casual).

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
        db_path: str = "speakers_db",
        device: str = "cuda",
    ) -> List[Tuple[str, str, str]]:
        """Extract a knowledge graph from a file.

        Convenience method combining transcript loading and graph extraction.

        Args:
            input_path: Path to input (JSON, TXT, or audio file)
            db_path: Path to speaker database (for audio processing)
            device: Device for audio processing ("cuda" or "cpu")

        Returns:
            List of (subject, relation, object) tuples

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If content cannot be extracted or parsed
            RuntimeError: If Copilot call fails
        """
        transcript = self._load_transcript(input_path, db_path, device)
        return self.extract_graph(transcript)

    def analyze_from_file(
        self,
        input_path: str,
        db_path: str = "speakers_db",
        device: str = "cuda",
    ) -> str:
        """Analyze content from a file.

        Convenience method combining transcript loading and analysis.

        Args:
            input_path: Path to input (JSON, TXT, or audio file)
            db_path: Path to speaker database (for audio processing)
            device: Device for audio processing ("cuda" or "cpu")

        Returns:
            Structured analysis text

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If content cannot be extracted or is empty
            RuntimeError: If Copilot call fails
        """
        transcript = self._load_transcript(input_path, db_path, device)
        return self.analyze(transcript)

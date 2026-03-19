"""Tool: extract_graph — knowledge-graph triple extraction, saved to the database."""

from __future__ import annotations

import json
import re
from typing import List, Tuple

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.transcript import fetch_transcript
from pawn_agent.utils.db import save_graph_triples

NAME = "extract_graph"
DESCRIPTION = (
    "Extract a knowledge graph from a session transcript as (subject, relation, object) "
    "triples and persist the result to the database. "
    "Use this to map entities, relationships, and concepts discussed in a conversation."
)

_SYSTEM_PROMPT = (
    "You are a knowledge graph extraction expert. "
    "Analyse only the transcript provided in the user message. "
    "Do not use any knowledge about the user's codebase, tools, or environment."
)


def _parse_triples(raw: str) -> List[Tuple[str, str, str]]:
    """Strip markdown fences and parse a JSON array of triples."""
    cleaned = re.sub(r"^```[^\n]*\n", "", raw.strip())
    cleaned = re.sub(r"\n```$", "", cleaned.strip())
    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of triples")
    triples: List[Tuple[str, str, str]] = []
    for item in data:
        if isinstance(item, list) and len(item) == 3:
            triples.append((str(item[0]), str(item[1]), str(item[2])))
    return triples


def build(cfg: AgentConfig) -> Tool:
    async def extract_graph(session_id: str) -> str:
        """Extract a knowledge graph from a conversation session transcript.

        Sends the transcript to the model and asks it to identify relationships
        between entities (people, organizations, topics, concepts, places, events).
        Returns 10-40 triples depending on content richness and stores them in
        the ``graph_triples`` database table, replacing any previous extraction
        for the same session.

        Args:
            session_id: Unique session identifier stored in the database.
        """
        from pawn_agent.core.llm_sub import run as llm_run

        try:
            transcript = fetch_transcript(cfg, session_id)
            if transcript.startswith("Error") or transcript.startswith("No transcript"):
                return transcript

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
- Return 10\u201340 triples depending on content richness
- Do NOT include any explanation, markdown, or text outside the JSON array

Example output format:
[["Alice", "works at", "Acme Corp"], ["Bob", "disagrees with", "Alice"], ["Acme Corp", "is located in", "Berlin"]]

---
TRANSCRIPT:
{transcript}
---"""

            raw = await llm_run(cfg, prompt, system_prompt=_SYSTEM_PROMPT)

            try:
                triples = _parse_triples(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                return f"Error parsing graph triples: {exc}\nRaw response (first 300 chars): {raw[:300]}"

            count = save_graph_triples(session_id, triples, cfg.model, cfg.db_dsn)

            preview_lines = [f"  {s} --[{r}]--> {o}" for s, r, o in triples[:5]]
            preview = "\n".join(preview_lines)
            suffix = f"\n  … ({count - 5} more)" if count > 5 else ""
            return f"Extracted and saved {count} triples for session {session_id!r}:\n{preview}{suffix}"
        except Exception as exc:
            return f"Error extracting knowledge graph: {exc}"

    return Tool(extract_graph)

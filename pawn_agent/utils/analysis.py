"""Core structured-analysis logic shared by analyze_summary and vectorize tools."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.transcript import fetch_transcript
from pawn_agent.utils.db import save_session_analysis

_SYSTEM_PROMPT = (
    "You are an expert conversation analyst. "
    "Analyse only the transcript provided in the user message. "
    "Do not use any knowledge about the user's codebase, tools, or environment."
)

_PROMPT_TEMPLATE = """\
You are an expert conversation analyst. Below is a speaker-diarized transcript. \
Please provide a structured analysis with the following sections:

## Title
A short, descriptive title for the conversation (5\u201310 words).

## Summary
A concise paragraph (10\u201320 sentences) summarising the main discussion.

## Key Topics / Keywords
A bullet list of the most important topics, concepts, and keywords extracted from the transcript.

## Speaker Highlights
For each speaker, one or two sentences describing their main contributions or talking points.

## Sentiment
Overall tone of the conversation \u2014 write one or two descriptive sentences.

## Sentiment Tags
Up to 3 short, lowercase, single-word sentiment labels (e.g. collaborative, tense, formal).
Output ONLY a comma-separated list on a single line, no bullet points, no extra text.

## Tags
5\u201310 short, lowercase, single-word-or-hyphenated tags covering key topics, entities, and tone.
Output ONLY a comma-separated list on a single line, no bullet points, no extra text.

---
TRANSCRIPT:
{transcript}
---

Respond only with the structured analysis above. Do not repeat the transcript."""


def _split_tags(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    cleaned = raw.strip().strip("`").strip()
    cleaned = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", cleaned)
    cleaned = re.sub(r"\s*\n\s*", ", ", cleaned)
    tags = [
        re.sub(r"[`*_]", "", t).strip().lower()
        for t in cleaned.split(",")
        if t.strip()
    ]
    return tags if tags else None


def parse_sections(analysis_text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "title": None, "summary": None, "key_topics": None,
        "speaker_highlights": None, "sentiment": None,
        "sentiment_tags": None, "tags": None,
    }
    for block in re.split(r"(?m)^## ", analysis_text):
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
            result["sentiment_tags"] = _split_tags(content)
        elif "sentiment" in heading:
            result["sentiment"] = content
        elif heading == "tags":
            result["tags"] = _split_tags(content)
    return result


async def run_analysis(cfg: AgentConfig, session_id: str) -> str:
    """Fetch transcript, run structured analysis, persist to DB, return raw content.

    Raises:
        ValueError: If transcript cannot be loaded.
        RuntimeError: If the LLM call fails.
    """
    from pawn_agent.core.llm_sub import run as llm_run

    transcript = fetch_transcript(cfg, session_id)
    if transcript.startswith("Error") or transcript.startswith("No transcript"):
        raise ValueError(transcript)

    content = await llm_run(
        cfg,
        _PROMPT_TEMPLATE.format(transcript=transcript),
        system_prompt=_SYSTEM_PROMPT,
    )

    sections = parse_sections(content)
    save_session_analysis(
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
        dsn=cfg.db_dsn,
    )
    return content

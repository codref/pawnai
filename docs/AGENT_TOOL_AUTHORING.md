# pawn-agent Tool Authoring Spec

Machine-readable reference for implementing tools in `pawn_agent/tools/`.
Target reader: AI copilot implementing or modifying tools.

---

## Discovery Contract

The registry in `pawn_agent/tools/__init__.py` auto-discovers every non-private
module in the package at session startup. A module is included if:

1. Filename does **not** start with `_`.
2. The module exposes a `build` callable.

Modules are loaded in alphabetical order by filename.

Required module-level exports:

```python
NAME: str         # identifier shown in `pawn-agent tools`
DESCRIPTION: str  # one-line description shown in `pawn-agent tools`

def build(cfg: AgentConfig) -> pydantic_ai.Tool: ...
```

`NAME` and `DESCRIPTION` fall back to module name / empty string if absent, but
they must be present for the tool to be usable by the agent.

---

## Module Skeleton

```python
"""Tool: <name> — <one-line description>."""

from __future__ import annotations

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig

NAME = "<name>"
DESCRIPTION = "<one-line description for CLI and LLM tool list>"


def build(cfg: AgentConfig) -> Tool:
    def <name>(<param>: <type>, ...) -> str:
        """<Full docstring — this is what the LLM reads when choosing and calling the tool.>

        Args:
            <param>: <description>
        """
        ...

    return Tool(<name>)
```

**Critical rules:**
- `build` receives only `cfg: AgentConfig`. There is no `client` parameter.
- The inner function is passed directly to `Tool(fn)`.
- The inner function's **docstring** is the LLM-visible description — make it
  precise and actionable. The module-level `DESCRIPTION` is only for the CLI.
- Return a `str` in all branches, including errors (see Error Handling below).
- Do not raise exceptions from the tool function — catch and return error strings.

---

## Sync vs Async Tools

Use a regular function unless the tool needs to `await` something:

```python
# Sync — database reads, SiYuan API calls
def build(cfg: AgentConfig) -> Tool:
    def my_tool(session_id: str) -> str:
        ...
    return Tool(my_tool)
```

Use `async def` when calling `llm_sub.run` or any other coroutine:

```python
# Async — single-turn LLM sub-calls
def build(cfg: AgentConfig) -> Tool:
    async def my_tool(session_id: str) -> str:
        from pawn_agent.core.llm_sub import run as llm_run
        result = await llm_run(cfg, prompt, system_prompt="...")
        return result
    return Tool(my_tool)
```

---

## AgentConfig — Relevant Properties

`cfg` is an instance of `pawn_agent.utils.config.AgentConfig`.

| Property | Type | Description |
|---|---|---|
| `cfg.db_dsn` | `str` | PostgreSQL DSN for SQLAlchemy |
| `cfg.siyuan_url` | `str` | SiYuan HTTP API base URL |
| `cfg.siyuan_token` | `str` | SiYuan API token |
| `cfg.siyuan_notebook` | `str` | SiYuan notebook ID |
| `cfg.siyuan_path_template` | `str` | Path template for session documents |
| `cfg.embed_model` | `str` | Sentence-transformers model name |
| `cfg.embed_dim` | `int` | Embedding dimension (e.g. 1024) |
| `cfg.embed_device` | `str` | `"cpu"` or `"cuda"` |
| `cfg.pydantic_model` | `str` | PydanticAI model string (e.g. `"openai:gpt-4o"`) |
| `cfg.pydantic_api_key` | `Optional[str]` | API key for pydantic model |
| `cfg.pydantic_base_url` | `Optional[str]` | Base URL override for pydantic model |
| `cfg.model` | `str` | Copilot sub-agent model name |

---

## Shared Utilities

Import inside the function body for heavy dependencies to avoid slowing startup.

### Database

```python
from pawn_agent.utils.db import make_db_session

db = make_db_session(cfg.db_dsn)
try:
    rows = db.execute(...).fetchall()
finally:
    db.close()
```

ORM models available from `pawn_agent.utils.db` (re-exported from `pawn_core.database`):

| Model | Table | Key columns |
|---|---|---|
| `TranscriptionSegment` | `transcription_segments` | `session_id`, `speaker_label`, `start_time`, `end_time`, `text` |
| `SpeakerName` | `speaker_names` | `session_id`, `speaker_label`, `name` |
| `SessionAnalysis` | `session_analysis` | `session_id`, `title`, `summary`, `key_topics`, `speaker_highlights`, `sentiment`, `sentiment_tags`, `tags`, `model`, `analyzed_at` |
| `RagSource` | `rag_sources` | `id`, `source_type`, `external_id`, `display_name`, `created_at` |
| `TextChunk` | `text_chunks` | `source_id`, `speaker_name`, `start_time`, `end_time`, `text`, `metadata`, `embedding` |
| `GraphTriple` | `graph_triples` | `session_id`, `subject`, `relation`, `object`, `model`, `extracted_at` |

Higher-level DB helpers in `pawn_agent.utils.db`:

```python
from pawn_agent.utils.db import get_session_analysis, save_session_analysis, save_graph_triples
```

### Transcript

```python
from pawn_agent.utils.transcript import fetch_transcript

text = fetch_transcript(cfg, session_id)
# Returns formatted speaker-turn text, or an error string starting with "Error" / "No transcript"
```

### SiYuan

```python
from pawn_agent.utils.siyuan import siyuan_post, do_save_to_siyuan

# Raw API call
result = siyuan_post(cfg.siyuan_url, cfg.siyuan_token, "/api/...", payload_dict)

# Save content as a child document under the session page
doc_url = do_save_to_siyuan(cfg, session_id, title, markdown_content, path=None, tags=None)
```

`do_save_to_siyuan` creates a document at
`conversations/{date}/{session_id}/{title}` unless `path` overrides it.
Returns the SiYuan URL string.

### LLM Sub-call

For single-turn completions using the configured PydanticAI model:

```python
from pawn_agent.core.llm_sub import run as llm_run

raw: str = await llm_run(cfg, user_prompt, system_prompt="Optional system prompt")
```

Uses `cfg.pydantic_model` / `cfg.pydantic_api_key` / `cfg.pydantic_base_url`.

### Vectorization

```python
from pawn_agent.utils.vectorize import vectorize_session, vectorize_siyuan_page

n_chunks, source_id = vectorize_session(
    session_id=session_id,
    db_dsn=cfg.db_dsn,
    embed_model=cfg.embed_model,
    embed_device=cfg.embed_device,
    embed_dim=cfg.embed_dim,
)

n_chunks = vectorize_siyuan_page(
    page_id=page_id,          # resolved SiYuan block ID
    siyuan_url=cfg.siyuan_url,
    siyuan_token=cfg.siyuan_token,
    db_dsn=cfg.db_dsn,
    embed_model=cfg.embed_model,
    embed_device=cfg.embed_device,
    embed_dim=cfg.embed_dim,
)
```

---

## Error Handling

Never raise from a tool. Catch all exceptions and return a descriptive string:

```python
try:
    result = do_something()
    return result
except Exception as exc:
    return f"Error doing something: {exc}"
```

Callers (including the LLM) distinguish errors by the string prefix, so start
error returns with `"Error: ..."` or `"Error <verb>ing ...: ..."`.

---

## Heavy Import Deferral

Import heavy dependencies (sentence-transformers, SQLAlchemy engines, etc.)
inside the inner function body or inside `build`, not at module top-level.
The exception is a model loaded **once per session** — load it inside `build`
so it is shared across all invocations:

```python
def build(cfg: AgentConfig) -> Tool:
    # Loaded once at session startup; shared across all calls
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(cfg.embed_model, device=cfg.embed_device,
                                  truncate_dim=cfg.embed_dim or None)

    def my_tool(query: str) -> str:
        vec = _model.encode(query).tolist()
        ...

    return Tool(my_tool)
```

---

## Tool Docstring Guidelines

The inner function docstring is sent verbatim to the LLM. Write it to answer:

1. **When to call this tool** — disambiguate from similar tools.
2. **What each parameter means** — be exact about expected format.
3. **What the return value contains** — so the LLM knows how to use it.
4. **What NOT to do** — e.g. "Never call X before this" or "Use Y instead if…".

Use Google-style `Args:` blocks. Keep it under ~200 words.

---

## Complete Examples

### Sync tool — DB read

```python
"""Tool: list_sessions — list all sessions with transcription data."""

from __future__ import annotations

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig

NAME = "list_sessions"
DESCRIPTION = "List all sessions that have transcription segments in the database."


def build(cfg: AgentConfig) -> Tool:
    def list_sessions(limit: int = 20) -> str:
        """List sessions that have stored transcription data.

        Returns a newline-separated list of session IDs ordered by most
        recently created. Use this to discover available session IDs before
        calling query_conversation or analyze_summary.

        Args:
            limit: Maximum number of sessions to return (1–100).
        """
        from sqlalchemy import select, func
        from pawn_agent.utils.db import make_db_session, TranscriptionSegment

        limit = max(1, min(100, limit))
        db = make_db_session(cfg.db_dsn)
        try:
            rows = db.execute(
                select(TranscriptionSegment.session_id)
                .group_by(TranscriptionSegment.session_id)
                .order_by(func.max(TranscriptionSegment.start_time).desc())
                .limit(limit)
            ).fetchall()
        finally:
            db.close()

        if not rows:
            return "No sessions found."
        return "\n".join(r.session_id for r in rows)

    return Tool(list_sessions)
```

### Async tool — LLM sub-call

```python
"""Tool: custom_analysis — run a bespoke free-form analysis on a session."""

from __future__ import annotations

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig

NAME = "custom_analysis"
DESCRIPTION = "Run a free-form LLM analysis on a session transcript using a custom prompt."

_SYSTEM = "You are an expert analyst. Analyse only the transcript given. Return plain Markdown."


def build(cfg: AgentConfig) -> Tool:
    async def custom_analysis(session_id: str, instruction: str) -> str:
        """Run a bespoke analysis on a session transcript.

        Fetches the full transcript and sends it to the LLM with your custom
        instruction. Returns the model's Markdown response.

        Use this for one-off analyses that do not fit the standard
        analyze_summary format (Title / Summary / Topics / Sentiment).
        For the standard analysis, use analyze_summary instead.

        Args:
            session_id: Unique session identifier stored in the database.
            instruction: Free-form analysis instruction (e.g. "List all
                action items with owner and deadline").
        """
        from pawn_agent.core.llm_sub import run as llm_run
        from pawn_agent.utils.transcript import fetch_transcript

        transcript = fetch_transcript(cfg, session_id)
        if transcript.startswith("Error") or transcript.startswith("No transcript"):
            return transcript

        prompt = f"{instruction}\n\n---\nTRANSCRIPT:\n{transcript}\n---"
        try:
            return await llm_run(cfg, prompt, system_prompt=_SYSTEM)
        except Exception as exc:
            return f"Error running analysis: {exc}"

    return Tool(custom_analysis)
```

---

## Existing Tools — Quick Reference

| Tool module | NAME | Sync/Async | Primary dependency |
|---|---|---|---|
| `analyze_summary.py` | `analyze_summary` | async | `utils/analysis.py`, `utils/siyuan.py` |
| `extract_graph.py` | `extract_graph` | async | `llm_sub`, `utils/transcript.py`, `utils/db.py` |
| `fetch_siyuan_page.py` | `fetch_siyuan_page` | sync | `utils/siyuan.py` |
| `get_analysis.py` | `get_analysis` | sync | `utils/db.py` |
| `query_conversation.py` | `query_conversation` | sync | `utils/transcript.py` |
| `rag_stats.py` | `rag_stats` | sync | `utils/db.py` (SQLAlchemy direct) |
| `save_to_siyuan.py` | `save_to_siyuan` | sync | `utils/siyuan.py` |
| `search_knowledge.py` | `search_knowledge` | sync | `sentence_transformers`, `utils/db.py` |
| `vectorize.py` | `vectorize` | async | `utils/vectorize.py`, `utils/siyuan.py` |

---

## Checklist for a New Tool

- [ ] File placed in `pawn_agent/tools/`, filename does not start with `_`
- [ ] `NAME`, `DESCRIPTION` defined at module level
- [ ] `build(cfg: AgentConfig) -> Tool` defined and returns `Tool(fn)`
- [ ] Inner function has a precise docstring (when to call, args, return, caveats)
- [ ] All exceptions caught; error branches return `"Error ...: ..."` strings
- [ ] Heavy imports deferred inside `build` or the inner function
- [ ] No new utils or abstractions unless used by more than one tool

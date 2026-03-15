# pawn-agent Tools

Tools are auto-discovered at runtime from the `pawn_agent/tools/` package.
Any non-private Python module placed there is picked up automatically — no
registration or import changes required.

## Listing available tools

```
pawn-agent tools
```

## Writing a new tool

Create a file in `pawn_agent/tools/`. The module must export three things:

| Export | Type | Purpose |
|---|---|---|
| `NAME` | `str` | Tool name shown to the LLM and in `pawn-agent tools` |
| `DESCRIPTION` | `str` | One-line description shown in `pawn-agent tools` |
| `build(cfg, client)` | callable | Factory that returns a Copilot SDK `Tool` |

### Minimal example

```python
# pawn_agent/tools/my_tool.py

from copilot import CopilotClient, Tool, define_tool
from pydantic import BaseModel, Field

from pawn_agent.utils.config import AgentConfig

NAME = "my_tool"
DESCRIPTION = "One-line description shown in pawn-agent tools."


class MyToolParams(BaseModel):
    session_id: str = Field(description="Unique session identifier.")


def build(cfg: AgentConfig, client: CopilotClient) -> Tool:
    @define_tool(description="Full description seen by the LLM when selecting tools.")
    def my_tool(params: MyToolParams) -> str:
        # cfg and client are available via closure
        return "result"

    return my_tool  # type: ignore[return-value]
```

### Async tools

Tools that call the LLM (e.g. via `client.create_session`) must be `async`:

```python
def build(cfg: AgentConfig, client: CopilotClient) -> Tool:
    @define_tool(description="...")
    async def my_tool(params: MyToolParams) -> str:
        session = await client.create_session({"model": cfg.model, ...})
        try:
            response = await session.send_and_wait(MessageOptions(prompt="..."), timeout=120)
            return response.data.content or ""
        finally:
            await session.disconnect()

    return my_tool  # type: ignore[return-value]
```

## Shared helpers

Shared code lives in `pawn_agent/utils/` and is imported directly by tools:

| Module | Contents |
|---|---|
| `utils/db.py` | ORM models (`TranscriptionSegment`, `SpeakerName`, `SessionAnalysis`) and `make_db_session()` |
| `utils/transcript.py` | `fetch_transcript(cfg, session_id)`, `parse_sections()`, `ANALYSIS_PROMPT` |
| `utils/siyuan.py` | `do_save_to_siyuan(cfg, ...)`, `siyuan_post()`, `resolve_path()`, `build_siyuan_markdown()` |

## Discovery rules

- Module filename must **not** start with `_`
- Module must expose a `build` callable
- `NAME` and `DESCRIPTION` are optional but strongly recommended (`pawn-agent tools` falls back to the module name and an empty string)
- Modules are loaded in alphabetical order

## Available tools

| Tool | Description |
|---|---|
| `query_conversation` | Fetch and return the full transcript for a session from the database. |
| `analyze_custom` | Perform a free-form analysis; set `save` to the user's phrasing to also persist to SiYuan. |
| `save_to_siyuan` | Save already-generated Markdown content to SiYuan Notes as a new document. |

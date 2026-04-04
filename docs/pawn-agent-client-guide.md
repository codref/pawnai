# PawnAgent API Client Guide

This document describes how to build a client that talks to **pawn-agent** through the **litellm proxy**.
The proxy exposes a standard OpenAI-compatible API, so any OpenAI SDK works out of the box.

---

## Architecture overview

```
Your client
    │
    │  POST /v1/chat/completions  (OpenAI format)
    ▼
litellm proxy  :4000
    │
    │  POST /chat  (pawn-agent native format)
    ▼
pawn-agent server  :8000
    │
    ├── PostgreSQL (session history, RAG, analysis)
    └── SiYuan notes
```

The litellm proxy translates the standard OpenAI request envelope into pawn-agent's `/chat` endpoint.
Your client never calls pawn-agent directly; it only talks to the proxy.

---

## litellm proxy

### Base URL

```
http://localhost:4000
```

### Authentication

Set the `LITELLM_MASTER_KEY` environment variable on the proxy to enable auth.
When set, every request must include:

```
Authorization: Bearer <LITELLM_MASTER_KEY>
```

When `LITELLM_MASTER_KEY` is not set, the proxy is open (dev mode).

### Model name

The model name encodes both the target and the optional model override:

| Model string | Meaning |
|---|---|
| `pawn-agent` | Use the model configured in `pawnai.yaml` |
| `pawn_agent/default` | Same as above (internal form) |
| `pawn_agent/openai:gpt-4o` | Override model to `openai:gpt-4o` for this request |
| `pawn_agent/anthropic:claude-3-5-sonnet-latest` | Override model to Anthropic |
| `pawn_agent/ollama:qwen3:14b` | Override to a local Ollama model |

Use the `pawn-agent` alias (defined in `litellm_config.yaml`) from your client.
If you need a one-off model override, use the full `pawn_agent/<provider>:<model>` form.

---

## Request format

Standard OpenAI `/v1/chat/completions` request.

```json
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer <key>

{
  "model": "pawn-agent",
  "messages": [
    { "role": "user", "content": "Summarise session abc123 and save to SiYuan" }
  ],
  "user": "my-session-id"
}
```

### Key fields

| Field | Required | Notes |
|---|---|---|
| `model` | yes | `pawn-agent` or `pawn_agent/<override>` |
| `messages` | yes | Only the **last** `user` message is forwarded as the prompt. Include full history only for context; the agent maintains its own history per session. |
| `user` | recommended | Maps to `session_id` in pawn-agent. Omit to get an auto-generated deterministic session (MD5 of first message). Use a stable UUID per conversation to get persistent memory. |

### What the proxy ignores

The proxy sets `drop_params: true`, so unsupported OpenAI parameters (`temperature`, `top_p`, `max_tokens`, `stream`, etc.) are silently dropped.
Streaming is **not supported** — requests always return a single completion.

---

## Response format

Standard OpenAI response object.

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1711900000,
  "model": "pawn-agent",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here is the summary of session abc123 ..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 120,
    "total_tokens": 130
  }
}
```

Extract the reply from `choices[0].message.content`.
Token counts are **estimates** (rough word-based approximation, not tokenizer-accurate).

---

## Session management

Session history is stored in PostgreSQL (`agent_session_turns` table).
The agent accumulates full conversation history per `session_id` across requests.

### Start a new conversation

Pick a stable UUID and pass it as `user` on every request in that conversation:

```python
import uuid
SESSION_ID = str(uuid.uuid4())   # generate once, reuse across turns
```

### Reset / clear a session

Send the sentinel prompt `/reset` through the proxy with the same `user` (session_id):

```python
client.chat.completions.create(
    model="pawn-agent",
    messages=[{"role": "user", "content": "/reset"}],
    user=SESSION_ID,
)
# → "Session reset. (3 turns deleted)"
```

The proxy intercepts `/reset`, calls `DELETE /sessions/{session_id}` on pawn-agent,
and returns the confirmation message. The `/chat` endpoint is never called.

---

## What the agent can do

The agent auto-discovers tools from `pawn_agent/tools/`. Currently available:

| Tool | What it does |
|---|---|
| `query_conversation` | Fetch the full transcript for a diarization session |
| `search_knowledge` | Semantic search over transcripts and SiYuan pages (RAG) |
| `extract_graph` | Extract knowledge-graph triples (subject → relation → object) |
| `fetch_siyuan_page` | Read a SiYuan note by path |
| `get_analysis` | Return stored analysis (title, summary, topics, sentiment, tags) |
| `analyze_summary` | Run a fresh standard analysis on a session |
| `save_to_siyuan` | Write Markdown content to a SiYuan note |
| `rag_stats` | Show RAG index summary |
| `vectorize` | Embed a session or SiYuan page into the RAG index |

Prompt the agent in natural language. It selects and chains tools automatically.

Example prompts:
- `"Summarise session abc123"` → uses `analyze_summary`
- `"What did Alice say about the budget in session abc123?"` → uses `query_conversation` or `search_knowledge`
- `"Save a note titled Meeting Notes with the summary of session abc123 to SiYuan"` → chains `analyze_summary` → `save_to_siyuan`
- `"Index session abc123 into the knowledge base"` → uses `vectorize`

---

## Python client example

```python
from openai import OpenAI
import uuid

client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-litellm-master-key",   # empty string if no auth
)

SESSION_ID = str(uuid.uuid4())

def chat(prompt: str) -> str:
    response = client.chat.completions.create(
        model="pawn-agent",
        messages=[{"role": "user", "content": prompt}],
        user=SESSION_ID,
    )
    return response.choices[0].message.content

# One-shot
print(chat("Summarise session abc123"))

# Multi-turn: each call shares SESSION_ID, the agent remembers context
print(chat("Now save that summary to SiYuan under 'Meetings/2026-03'"))
```

---

## curl examples

```bash
# Health check (pawn-agent directly, not through proxy)
curl http://localhost:8000/health

# One-shot prompt via litellm proxy
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "pawn-agent",
    "messages": [{"role": "user", "content": "List available tools"}],
    "user": "test-session-1"
  }'

# Same request with a model override
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "pawn_agent/openai:gpt-4o",
    "messages": [{"role": "user", "content": "Summarise session abc123"}],
    "user": "test-session-2"
  }'

# Delete a session (pawn-agent directly)
curl -X DELETE http://localhost:8000/sessions/test-session-1 \
  -H "Authorization: Bearer $PAWN_AGENT_TOKEN"
```

---

## Running the stack

```bash
# Start postgres + siyuan + litellm proxy
docker compose -f docker/docker-compose.yml up -d

# Start pawn-agent HTTP server (in a separate terminal, with venv active)
source .venv/bin/activate
pawn-agent serve --host 0.0.0.0 --port 8000
```

Environment variables for docker-compose:

| Variable | Default | Purpose |
|---|---|---|
| `LITELLM_MASTER_KEY` | _(none, open)_ | Auth key for the proxy |
| `PAWN_AGENT_URL` | `http://localhost:8000` | Where the proxy finds pawn-agent |

---

## Limitations and gotchas

- **No streaming**: the response is always a single blocking completion. Do not set `stream: true`.
- **Single message forwarded**: only the last `user` message in `messages` becomes the prompt. History management is server-side by `session_id`, not client-side via the messages array.
- **Token counts are approximate**: `usage` is estimated, not from a real tokenizer.
- **Model idle timeout**: after 10 minutes of inactivity the agent cache is cleared (configurable via `api.model_idle_timeout_minutes`). Sessions persist in the DB; only the in-memory agent object is recycled.
- **Session reset via `/reset` sentinel**: send `{"role": "user", "content": "/reset"}` as the last message to clear the session through the proxy. No need to call pawn-agent directly.
- **pawn-agent auth is separate from litellm auth**: the proxy forwards a Bearer token to pawn-agent from its own config (`api_key` in `litellm_config.yaml`). Your client only needs the litellm master key.

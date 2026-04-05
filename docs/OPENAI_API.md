# pawn-agent OpenAI-compatible API

The pawn-agent HTTP server (`pawn-agent serve`) exposes an OpenAI-compatible
REST API on port 8000 (configurable).  Any OpenAI client library works without
modification — just point `base_url` at the server.

---

## Base URL and authentication

```
http://localhost:8000
```

All endpoints except `GET /health` require a Bearer token:

```
Authorization: Bearer <api.token from pawnai.yaml>
```

If `api.token` is not set the server starts in **open mode** (no auth required).
Useful for local development; not recommended in production.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completions (stateful or stateless) |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text (Parakeet / Whisper) |
| `POST` | `/v1/audio/speech` | Text-to-speech (Kokoro) |
| `POST` | `/knowledge` | Index content into the RAG store |
| `DELETE` | `/sessions/{session_id}` | Clear a session's conversation history |
| `GET` | `/health` | Liveness probe (no auth) |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/openapi.json` | OpenAPI spec |

---

## Chat completions

### Request

```
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer <token>
```

```json
{
  "model": "pawn-agent",
  "messages": [
    { "role": "user", "content": "Summarise session abc123" }
  ],
  "user": "my-session-id",
  "stream": false
}
```

#### Model field — session mode and model override

| `model` value | Session mode | LLM backend |
|---|---|---|
| `pawn-agent` | stateful (DB history) | configured default |
| `pawn-agent/openai:gpt-4o` | stateful | gpt-4o |
| `pawn-agent/anthropic:claude-3-5-sonnet-latest` | stateful | Anthropic |
| `pawn-agent/stateless` | stateless (request-only history) | configured default |
| `pawn-agent/stateless/openai:gpt-4o` | stateless | gpt-4o |

**Stateful** mode loads and saves conversation history in PostgreSQL keyed by
`user` (session_id).  **Stateless** mode uses the `messages` array from the
request directly — useful with clients that manage history themselves (Open
WebUI, Continue, etc.).

#### Session ID

Set `user` to a stable UUID to persist conversation across requests.  Omit it
to get an auto-generated ID derived from the first message (deterministic but
not guessable).

#### Reset a session

Send `/reset` as the last user message to clear history:

```json
{ "model": "pawn-agent", "messages": [{"role": "user", "content": "/reset"}], "user": "my-session-id" }
```

Or call `DELETE /sessions/{session_id}` directly.

#### Streaming

Set `"stream": true` to receive Server-Sent Events (SSE).  The full response is
generated first; tokens are then streamed word-by-word.

### Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1711900000,
  "model": "pawn-agent",
  "choices": [{
    "index": 0,
    "message": { "role": "assistant", "content": "Session abc123 ..." },
    "finish_reason": "stop"
  }],
  "usage": { "prompt_tokens": 10, "completion_tokens": 120, "total_tokens": 130 }
}
```

> Token counts are word-based estimates, not tokenizer-accurate.

### Python SDK example

```python
from openai import OpenAI
import uuid

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-token")
SESSION = str(uuid.uuid4())

def chat(prompt: str) -> str:
    r = client.chat.completions.create(
        model="pawn-agent",
        messages=[{"role": "user", "content": prompt}],
        user=SESSION,
    )
    return r.choices[0].message.content

print(chat("Summarise session abc123"))
print(chat("Now save that summary to SiYuan under 'Meetings/2026'"))
```

### curl example

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PAWN_AGENT_TOKEN" \
  -d '{
    "model": "pawn-agent",
    "messages": [{"role": "user", "content": "List available tools"}],
    "user": "test-session"
  }'
```

---

## Audio transcription

```
POST /v1/audio/transcriptions
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

Accepted formats: WAV, FLAC, OGG Opus, MP3, M4A, WebM (anything ffmpeg can
decode).  Non-WAV/FLAC inputs are converted to 16 kHz mono WAV automatically.

| Form field | Required | Values |
|---|---|---|
| `file` | yes | audio file |
| `model` | no | any string (accepted, ignored — always uses Parakeet/Whisper) |
| `response_format` | no | `json` (default), `verbose_json`, `text` |

**`json`** — `{"text": "transcribed text"}`
**`verbose_json`** — `{"text": "...", "words": [{"word": "...", "start": 0.5, "end": 0.8}]}`
**`text`** — plain string, `Content-Type: text/plain`

### Python SDK example

```python
with open("recording.wav", "rb") as f:
    t = client.audio.transcriptions.create(
        model="whisper-1",   # ignored by pawn-agent
        file=f,
        response_format="verbose_json",
    )
print(t.text)
print(t.words)   # word-level timestamps
```

### curl example

```bash
# JSON response
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $PAWN_AGENT_TOKEN" \
  -F "file=@recording.wav" \
  -F "model=whisper-1" \
  -F "response_format=json"

# Plain text
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $PAWN_AGENT_TOKEN" \
  -F "file=@recording.ogg" \
  -F "response_format=text"
```

---

## Audio speech (TTS)

```
POST /v1/audio/speech
Authorization: Bearer <token>
Content-Type: application/json
```

Powered by [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M).  See
[TTS.md](TTS.md) for the full language and voice reference.

### Request body

```json
{
  "model": "tts-1",
  "input": "Hello, world.",
  "voice": "alloy",
  "language": "en",
  "response_format": "wav",
  "speed": 1.0
}
```

| Field | Required | Default | Notes |
|---|---|---|---|
| `model` | yes | — | Accepted, ignored |
| `input` | yes | — | Text to synthesise |
| `voice` | no | config `tts_voice` | OpenAI alias or native Kokoro ID (e.g. `af_heart`) |
| `language` | no | config `tts_language` | BCP-47 code: `en`, `it`, `fr`, `es`, `pt`, `hi`, `ja`, `zh` |
| `response_format` | no | `wav` | `wav`, `mp3`, `opus`, `aac`, `flac`, `pcm` |
| `speed` | no | `1.0` | Range 0.25 – 4.0 |

### Python SDK example

```python
# Save to file
response = client.audio.speech.create(
    model="tts-1",
    input="Hello from pawn-agent.",
    voice="alloy",
    extra_body={"language": "en"},  # pawn-agent extension
)
response.stream_to_file("output.wav")

# Italian
response = client.audio.speech.create(
    model="tts-1",
    input="Ciao, come stai?",
    voice="alloy",
    extra_body={"language": "it"},
)
response.stream_to_file("output_it.wav")
```

### curl examples

```bash
# English WAV
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PAWN_AGENT_TOKEN" \
  -d '{"model":"tts-1","input":"Hello world","voice":"alloy","response_format":"wav"}' \
  -o out.wav -w "\nHTTP %{http_code}\n"

# Italian MP3
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PAWN_AGENT_TOKEN" \
  -d '{"model":"tts-1","input":"Ciao mondo","language":"it","response_format":"mp3"}' \
  -o out.mp3 -w "\nHTTP %{http_code}\n"

# Speed up (1.5×)
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PAWN_AGENT_TOKEN" \
  -d '{"model":"tts-1","input":"Fast speech test","speed":1.5,"response_format":"wav"}' \
  -o fast.wav -w "\nHTTP %{http_code}\n"
```

---

## Knowledge ingestion (RAG)

```
POST /knowledge
Content-Type: application/json
Authorization: Bearer <token>
```

Indexes content into the pgvector RAG store for use by the agent's
`search_knowledge` tool.

```json
{ "text": "Inline plain text to index..." }
```
```json
{ "session_id": "abc123" }
```
```json
{ "siyuan_path": "/Meetings/2026/april" }
```

Exactly one of `text`, `session_id`, or `siyuan_path` must be set.

Response: `{"chunks": 12, "message": "Indexed 12 chunks from inline text."}`

---

## Session management

### Clear a session

```
DELETE /sessions/{session_id}
Authorization: Bearer <token>
```

Permanently removes all stored turns for the session from PostgreSQL.

```bash
curl -s -X DELETE http://localhost:8000/sessions/my-session-id \
  -H "Authorization: Bearer $PAWN_AGENT_TOKEN"
```

---

## Health check

```
GET /health
```

No auth required.  Returns `{"status": "ok"}`.

---

## Using with any OpenAI client

Point the client's `base_url` at pawn-agent directly — no proxy needed:

```python
# Python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-token")
```

```typescript
// TypeScript / Node
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:8000/v1", apiKey: "your-token" });
```

```bash
# Environment variable for tools that read OPENAI_BASE_URL
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=your-token
```

---

## Going through the litellm proxy

See [pawn-agent-client-guide.md](pawn-agent-client-guide.md) for the full
litellm proxy setup.  The proxy adds model aliasing and a unified auth layer
but the API shape is identical — just change the `base_url` to
`http://localhost:4000`.

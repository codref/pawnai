# Testing pawn-agent via litellm proxy

## Prerequisites

1. Install the litellm extras:
   ```bash
   pip install -e ".[litellm]"
   ```

2. Start pawn-agent API server:
   ```bash
   pawn-agent serve
   # listening on http://localhost:8000
   ```

3. Start litellm proxy:
   ```bash
   litellm --config litellm_config.yaml
   # listening on http://localhost:4000
   ```

   Or via Docker:
   ```bash
   make build
   docker run --rm -p 4000:4000 pawn-litellm-proxy
   ```

---

## Basic chat

Send a single prompt. The `user` field is used as `session_id` — omit it to get an auto-derived session.

> **Note:** the litellm proxy always requires a `LITELLM_MASTER_KEY` Bearer token when one is set at startup.

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "pawn-agent",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "user": "my-session"
  }' | jq .
```

Expected response shape:
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "pawn_agent/default",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 43,
    "total_tokens": 55
  }
}
```

---

## Multi-turn conversation

Reuse the same `user` value across calls — pawn-agent resumes the session from PostgreSQL history automatically.

```bash
# Turn 1
curl -s http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{"model":"pawn-agent","messages":[{"role":"user","content":"My name is Alice."}],"user":"alice-001"}' \
  | jq .choices[0].message.content

# Turn 2 — agent remembers the session
curl -s http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{"model":"pawn-agent","messages":[{"role":"user","content":"What is my name?"}],"user":"alice-001"}' \
  | jq .choices[0].message.content
```

---

## Model override

Override the backend model for a single request by using a named alias defined in `litellm_config.yaml`.

```bash
# Add to litellm_config.yaml:
#   - model_name: pawn-agent-gpt4o
#     litellm_params:
#       model: pawn_agent/openai:gpt-4o
#       api_base: http://localhost:8000

curl -s http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "pawn-agent-gpt4o",
    "messages": [{"role": "user", "content": "Summarise session abc123"}],
    "user": "my-session"
  }' | jq .
```

---

## With proxy master key

If `LITELLM_MASTER_KEY` is set when starting the proxy, pass it as a Bearer token:

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-my-proxy-key" \
  -d '{"model":"pawn-agent","messages":[{"role":"user","content":"Hello"}],"user":"my-session"}' \
  | jq .
```

---

## With pawn-agent Bearer token

If `api.token` is set in `pawnai.yaml`, update `api_key` in `litellm_config.yaml`:

```yaml
litellm_params:
  model: pawn_agent/default
  api_base: http://localhost:8000
  api_key: "your-pawn-agent-token"
```

---

## Health checks

```bash
# litellm proxy health (no auth required)
curl -s http://localhost:4000/health | jq .

# pawn-agent server health (no auth required)
curl -s http://localhost:8000/health | jq .
```

---

## List available models

```bash
curl -s http://localhost:4000/v1/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" | jq .
```

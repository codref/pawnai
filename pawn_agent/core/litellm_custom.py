"""litellm CustomLLM adapter for pawn_agent /chat API.

Translates OpenAI-format ``/v1/chat/completions`` requests into pawn_agent
``POST /chat`` calls and wraps the reply in a ``ModelResponse``.

Registration in ``litellm_config.yaml``::

    litellm_settings:
      custom_provider_map:
        - provider: "pawn_agent"
          custom_handler: pawn_agent.core.litellm_custom.pawn_agent_llm

Model override
--------------
The litellm model string encodes an optional pawn_agent model override as the
suffix after the first ``/``::

    pawn_agent/default                              # use server-configured model
    pawn_agent/openai:gpt-4o                        # override to openai:gpt-4o
    pawn_agent/anthropic:claude-3-5-sonnet-latest   # override to that model

The suffix is forwarded as ``model`` in the ``/chat`` request body; the server
passes it to ``_get_or_create_agent(cfg, model_override=...)``.

Session mapping
---------------
The OpenAI ``user`` field is used as ``session_id``.  If absent, a stable UUID
is derived from the MD5 of the first user message so the same conversation
always maps to the same pawn_agent session.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Optional

import httpx
from litellm import CustomLLM
from litellm.types.utils import Choices, Message, ModelResponse, Usage


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _last_user_message(messages: list[dict]) -> str:
    """Return the content of the last user-role message."""
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return ""


def _session_id(messages: list[dict], user: Optional[str]) -> str:
    """Resolve a pawn_agent session_id from the OpenAI request.

    Priority:
    1. ``user`` field from the OpenAI request.
    2. Deterministic UUID derived from the MD5 of the first user message.
    3. Random UUID as last resort.
    """
    if user:
        return user
    first = next((m["content"] for m in messages if m.get("role") == "user"), "")
    if first:
        return str(uuid.UUID(hashlib.md5(first.encode()).hexdigest()))
    return str(uuid.uuid4())


def _pawn_model(litellm_model: str) -> Optional[str]:
    """Extract a pawn_agent model override from the litellm model string.

    Returns ``None`` when the suffix is absent or equals ``"default"``,
    meaning the server should use its configured default.
    """
    suffix = litellm_model.split("/", 1)[-1] if "/" in litellm_model else ""
    return suffix if suffix and suffix != "default" else None


def _count_tokens(model: str, messages: list[dict], reply: str) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens), falling back to 0 on error."""
    try:
        from litellm import token_counter  # noqa: PLC0415

        prompt_tokens = token_counter(model=model, messages=messages)
        completion_tokens = token_counter(model=model, text=reply)
    except Exception:
        prompt_tokens = 0
        completion_tokens = 0
    return prompt_tokens, completion_tokens


def _build_response(reply: str, model: str, messages: list[dict]) -> ModelResponse:
    prompt_tokens, completion_tokens = _count_tokens(model, messages, reply)
    return ModelResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content=reply),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# CustomLLM
# ──────────────────────────────────────────────────────────────────────────────


class PawnAgentLLM(CustomLLM):
    """Proxies OpenAI-format completion requests to the pawn_agent /chat endpoint."""

    def _headers(self, api_key: Optional[str]) -> dict:
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _body(self, model: str, messages: list[dict], user: Optional[str]) -> dict:
        body: dict = {
            "session_id": _session_id(messages, user),
            "prompt": _last_user_message(messages),
        }
        pawn_model = _pawn_model(model)
        if pawn_model:
            body["model"] = pawn_model
        return body

    def completion(
        self,
        model: str,
        messages: list[dict],
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        base = (api_base or "http://localhost:8000").rstrip("/")
        resp = httpx.post(
            f"{base}/chat",
            json=self._body(model, messages, kwargs.get("user")),
            headers=self._headers(api_key),
            timeout=180,
        )
        resp.raise_for_status()
        return _build_response(resp.json()["reply"], model, messages)

    async def acompletion(
        self,
        model: str,
        messages: list[dict],
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        base = (api_base or "http://localhost:8000").rstrip("/")
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{base}/chat",
                json=self._body(model, messages, kwargs.get("user")),
                headers=self._headers(api_key),
            )
        resp.raise_for_status()
        return _build_response(resp.json()["reply"], model, messages)


# Module-level singleton referenced by litellm_config.yaml
pawn_agent_llm = PawnAgentLLM()

"""LM backends for pawn-agent.

Provides two backend options:

- **CopilotLM**: wraps the GitHub Copilot SDK (``github-copilot-sdk``) as a
  DSPy-compatible language model.  Useful when working inside an environment
  that already authenticates with GitHub Copilot.

- **build_lm**: factory that returns the right DSPy LM based on ``AgentConfig``.
  Use ``backend="openai"`` to point at any OpenAI-compatible endpoint
  (Ollama, Azure OpenAI, LM Studio, etc.) via DSPy's built-in litellm path.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import dspy

from pawn_agent.utils.config import AgentConfig


# ──────────────────────────────────────────────────────────────────────────────
# Copilot backend
# ──────────────────────────────────────────────────────────────────────────────


class CopilotLM(dspy.LM):
    """DSPy LM adapter backed by the GitHub Copilot SDK.

    The Copilot SDK is async; this class bridges it into DSPy's synchronous
    call interface using ``asyncio.run()``.

    Args:
        model: Copilot model identifier (e.g. ``"gpt-4o"``).
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        # Initialise dspy.LM with a sentinel model string.
        # We override __call__ so litellm is never actually invoked.
        super().__init__(model=f"copilot/{model}")
        self._copilot_model = model

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Call the Copilot API and return DSPy-compatible completions."""
        # Flatten messages (system / user / assistant turns) to a single
        # prompt string.  The Copilot SDK's MessageOptions only accepts a
        # plain ``prompt`` field, so we inline the role context explicitly.
        if messages:
            parts: list[str] = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                if not content:
                    continue
                if role == "system":
                    # system instructions come first, no prefix
                    parts.append(content)
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
                else:
                    parts.append(content)
            text = "\n\n".join(parts)
        else:
            text = prompt or ""

        response_text = asyncio.run(self._async_call(text))

        # DSPy expects a list of choices in openai-response format.
        choices = [{"message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}]
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        response = {"choices": choices, "usage": usage, "model": self._copilot_model}

        # Track in history (DSPy inspects this for logging / caching).
        self.history.append({"prompt": text, "response": response, **kwargs})

        return [response_text]

    async def _async_call(self, prompt: str) -> str:
        from copilot import CopilotClient, PermissionHandler, MessageOptions  # noqa: PLC0415

        client = CopilotClient()
        await client.start()
        session = await client.create_session({
            "model": self._copilot_model,
            "on_permission_request": PermissionHandler.approve_all,
        })
        response = await session.send_and_wait(MessageOptions(prompt=prompt))
        await session.destroy()
        await client.stop()
        return response.data.content


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────


def build_lm(cfg: AgentConfig) -> dspy.LM:
    """Return the appropriate DSPy LM for the given config.

    - ``cfg.backend == "copilot"`` → :class:`CopilotLM`
    - anything else → ``dspy.LM`` via litellm (OpenAI-compatible endpoint)

    Args:
        cfg: Populated :class:`~pawn_agent.utils.config.AgentConfig`.

    Returns:
        A DSPy LM instance ready to pass to ``dspy.configure(lm=...)``.
    """
    if cfg.backend == "copilot":
        return CopilotLM(model=cfg.model)

    # OpenAI-compatible path (Ollama, Azure, LM Studio, …)
    kwargs: dict[str, Any] = {}
    if cfg.openai_base_url:
        kwargs["api_base"] = cfg.openai_base_url
    if cfg.openai_api_key:
        kwargs["api_key"] = cfg.openai_api_key

    return dspy.LM(f"openai/{cfg.model}", **kwargs)

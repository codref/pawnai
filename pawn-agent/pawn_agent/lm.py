"""GitHub Copilot SDK → DSPy LM adapter.

Wraps the ``github-copilot-sdk`` ``CopilotClient`` as a DSPy-compatible
language model so that all DSPy modules (Predict, ChainOfThought, etc.)
transparently call Copilot as their LLM backend.

The Copilot SDK is async-native; DSPy may call the LM in either a sync or
async context.  This adapter handles both:
- If no running event loop exists (typical CLI / thread context): uses
  ``asyncio.run()``.
- If an event loop is already running (e.g. inside the queue listener):
  uses ``asyncio.get_event_loop().run_until_complete()`` via a thread
  executor to avoid nested-loop errors.

Usage::

    import dspy
    from pawn_agent.lm import CopilotLM

    lm = CopilotLM(model="gpt-4o", temperature=0.0)
    dspy.configure(lm=lm)

    # All dspy.Predict / dspy.ChainOfThought calls now use Copilot
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class CopilotLM:
    """DSPy-compatible LM backed by the GitHub Copilot SDK.

    Args:
        model:       Copilot model identifier (e.g. ``"gpt-4o"``).
        temperature: Sampling temperature (passed via ``SessionConfig`` when
                     the SDK supports it; otherwise best-effort).
        max_tokens:  Maximum tokens to generate (informational; forwarded
                     where the SDK accepts it).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # DSPy checks these attributes on LM objects
        self.history: List[Dict[str, Any]] = []

    # ── async core ────────────────────────────────────────────────────────────

    async def _acall(self, prompt: str) -> str:
        """Send *prompt* to the Copilot SDK and return the response text."""
        from copilot import CopilotClient, SessionConfig, MessageOptions  # lazy import

        client = CopilotClient()
        try:
            await client.start()
            session = await client.create_session(SessionConfig(model=self.model))
            response = await session.send_and_wait(MessageOptions(prompt=prompt))
            await session.destroy()

            if response and response.data.content:
                return response.data.content

            raise RuntimeError("Copilot SDK returned an empty response")
        finally:
            await client.stop()

    # ── sync bridge ───────────────────────────────────────────────────────────

    def _sync_call(self, prompt: str) -> str:
        """Run ``_acall`` from a synchronous context.

        Handles both cases: no event loop running (uses ``asyncio.run``), or
        a loop already running (offloads to a dedicated thread to avoid
        nested-loop errors with nested asyncio calls).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — safe to use asyncio.run()
            return asyncio.run(self._acall(prompt))

        # A loop is already running (e.g. inside the queue listener handler).
        # Run the coroutine in a separate thread with its own event loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self._acall(prompt))
            return future.result()

    # ── DSPy LM protocol ─────────────────────────────────────────────────────

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        n: int = 1,
        **kwargs: Any,
    ) -> List[str]:
        """DSPy calls this to get completions.

        DSPy may pass either a ``prompt`` string or a ``messages`` list
        (OpenAI chat format).  We normalise both to a single prompt string.

        Args:
            prompt:   Plain-text prompt (used by older DSPy signatures).
            messages: Chat messages list (used by newer DSPy signatures).
            n:        Number of completions (Copilot SDK only supports 1).
            **kwargs: Ignored (temperature / max_tokens are set at init time).

        Returns:
            List of completion strings (always length 1 for Copilot).
        """
        if messages:
            # Flatten chat messages into a single prompt string
            parts: List[str] = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"[System]\n{content}")
                elif role == "assistant":
                    parts.append(f"[Assistant]\n{content}")
                else:
                    parts.append(f"[User]\n{content}")
            effective_prompt = "\n\n".join(parts)
        elif prompt:
            effective_prompt = prompt
        else:
            raise ValueError("CopilotLM.__call__: either 'prompt' or 'messages' must be provided")

        logger.debug("CopilotLM calling model=%r prompt_len=%d", self.model, len(effective_prompt))

        response_text = self._sync_call(effective_prompt)

        # Record in history so DSPy's inspect_history() works
        self.history.append({
            "prompt": effective_prompt,
            "response": response_text,
            "model": self.model,
        })

        return [response_text]

    # ── DSPy compatibility helpers ────────────────────────────────────────────

    def inspect_history(self, n: int = 1) -> str:
        """Return a human-readable string of the last *n* LM interactions."""
        entries = self.history[-n:]
        lines: List[str] = []
        for i, entry in enumerate(entries, 1):
            lines.append(f"--- Call {i} (model={entry['model']}) ---")
            lines.append(f"PROMPT:\n{entry['prompt'][:500]}{'...' if len(entry['prompt']) > 500 else ''}")
            lines.append(f"RESPONSE:\n{entry['response'][:500]}{'...' if len(entry['response']) > 500 else ''}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CopilotLM(model={self.model!r}, temperature={self.temperature})"

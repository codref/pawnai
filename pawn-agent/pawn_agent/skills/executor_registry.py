"""ExecutorRegistry — maps tool ``function`` keys to Python callables.

Each registered callable must have the signature::

    async def execute(params: dict, cfg: AgentConfig) -> dict

Registration happens explicitly at boot time in
:mod:`pawn_agent.executors.registration`, keeping the registry itself
free of import-time side effects.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict

logger = logging.getLogger(__name__)

# Type alias for tool executor callables
ExecutorFn = Callable[[Dict[str, Any], Any], Awaitable[Dict[str, Any]]]


class ExecutorRegistry:
    """Maps tool ``function`` string keys to async Python callables.

    Example::

        registry = ExecutorRegistry()
        registry.register("s3.download", s3_executor.download)
        registry.register("transcription.run", transcription_executor.run)

        result = await registry.call("s3.download", {"audio_path": "s3://..."}, cfg)
    """

    def __init__(self) -> None:
        self._executors: Dict[str, ExecutorFn] = {}

    def register(self, key: str, fn: ExecutorFn) -> None:
        """Register *fn* under *key*.

        Args:
            key: The ``function`` value from the tool YAML (e.g. ``"s3.download"``).
            fn:  An async callable ``(params, cfg) -> dict``.
        """
        if key in self._executors:
            logger.warning("Overwriting existing executor for key %r", key)
        self._executors[key] = fn
        logger.debug("Registered executor %r → %s.%s", key, fn.__module__, fn.__qualname__)

    def get(self, key: str) -> ExecutorFn:
        """Return the callable registered under *key*.

        Raises:
            KeyError: If no executor is registered for *key*.
        """
        try:
            return self._executors[key]
        except KeyError:
            available = ", ".join(sorted(self._executors))
            raise KeyError(
                f"No executor registered for tool function {key!r}. "
                f"Available: {available or '(none)'}"
            ) from None

    async def call(self, key: str, params: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
        """Look up and invoke the executor for *key*.

        Args:
            key:    Tool function key.
            params: Resolved parameters dict (no templates at this point).
            cfg:    :class:`~pawn_agent.config.AgentConfig` instance.

        Returns:
            The dict returned by the executor.
        """
        fn = self.get(key)
        logger.debug("Calling executor %r with params keys=%s", key, list(params))
        return await fn(params, cfg)

    def registered_keys(self) -> list[str]:
        """Return a sorted list of all registered executor keys."""
        return sorted(self._executors)

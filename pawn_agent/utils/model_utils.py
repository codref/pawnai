"""Model override helpers shared between the CLI and the queue listener.

Selections resolve through :mod:`pawn_agent.utils.model_catalog`. A catalog
id (``provider@model``) switches provider, credentials, and profile. A
legacy string such as ``openai:gpt-4o`` only replaces the LiteLLM model.
"""

from __future__ import annotations

# Kept so older imports do not break. Catalog ids are not these prefixes.
_PYDANTIC_PREFIXES = (
    "openai:",
    "anthropic:",
    "google-gla:",
    "google-vertex:",
    "groq:",
    "mistral:",
    "bedrock:",
    "cohere:",
)


def _apply_model_override(cfg, model: str) -> None:
    """Apply a CLI ``--model`` value or a queue/schedule model string."""
    from pawn_agent.utils.model_catalog import apply_model_selection  # noqa: PLC0415

    apply_model_selection(cfg, model)

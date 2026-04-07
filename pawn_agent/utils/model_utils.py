"""Model override helpers shared between the CLI and the queue listener."""

from __future__ import annotations

# PydanticAI provider prefixes — anything else is treated as a bare model name.
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
    """Set cfg.pydantic_model from a CLI --model value.

    If the value already carries a PydanticAI provider prefix (e.g. 'openai:gpt-4o')
    it is used as-is.  Otherwise the current provider prefix is preserved and only
    the model name is replaced — so '--model qwen3.5:9b' keeps 'openai:' when the
    config points at an Ollama/OpenAI-compatible endpoint.
    """
    if any(model.startswith(p) for p in _PYDANTIC_PREFIXES):
        cfg.pydantic_model = model
    else:
        # Bare model name (e.g. "qwen3.5:9b") — keep the configured provider prefix
        current_prefix = cfg.pydantic_model.split(":")[0] + ":"
        cfg.pydantic_model = f"{current_prefix}{model}"

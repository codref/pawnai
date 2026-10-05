"""Catalog of selectable chat models.

A catalog id is ``provider@model`` (for example ``ollama@gemma4:4b``).
Each id binds an OpenAI-compatible provider (base URL and API key) to a
sallm compiled profile. The LiteLLM wire id is always ``openai/{model}``.

``pydantic_model`` / ``pydantic_api_key`` / ``pydantic_base_url`` are gone.
Callers read :class:`ModelSelection` from the config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, cast

from pawn_agent.profiles import load_compiled_profile, resolve_profile_path

logger = logging.getLogger(__name__)

# Legacy CLI / queue strings that are not catalog ids. Colon prefix only.
_LEGACY_PREFIXES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google-gla": "gemini",
    "google": "gemini",
    "groq": "groq",
    "mistral": "mistral",
}

_LEGACY_PROVIDER_ORDER = ("openai", "anthropic", "google", "groq", "mistral")

# state_dir path -> catalog id chosen for autonomous work in this process.
_background_memory: dict[str, str] = {}


@dataclass(frozen=True)
class ModelSelection:
    """One resolved chat model: wire id, credentials, and profile path."""

    catalog_id: str
    litellm_model: str
    api_base: Optional[str]
    api_key: Optional[str]
    user_agent: Optional[str]
    profile: Optional[str]
    provider: str
    # OpenRouter only. ``none`` | ``low`` | ``medium`` | ``high``.
    reasoning: Optional[str] = None
    # OpenRouter only. ``balanced`` | ``nitro`` | ``floor`` | ``exacto``.
    route: Optional[str] = None


@dataclass(frozen=True)
class CatalogEntry:
    """One row in the configured model list."""

    id: str
    provider: str
    model: str
    profile: Optional[str]
    api_base: Optional[str]
    api_key: Optional[str]
    user_agent: Optional[str]
    reasoning: Optional[str] = None
    route: Optional[str] = None
    vision: bool = False

    def selection(self) -> ModelSelection:
        return ModelSelection(
            catalog_id=self.id,
            litellm_model=openai_litellm_id(self.model),
            api_base=self.api_base,
            api_key=self.api_key,
            user_agent=self.user_agent,
            profile=self.profile,
            provider=self.provider,
            reasoning=self.reasoning,
            route=self.route,
        )


_API_ROUTE_SUFFIXES = (
    "/chat/completions",
    "/completions",
    "/responses",
    "/messages",
)


def normalize_openai_base_url(raw: Optional[str]) -> Optional[str]:
    """Return the OpenAI-compatible root LiteLLM should call.

    The OpenCode docs list full routes such as
    ``https://opencode.ai/zen/v1/chat/completions``. LiteLLM appends
    ``/chat/completions`` itself, so a pasted route becomes
    ``.../chat/completions/chat/completions`` and the site returns HTML 404.
    Keep the ``/v1`` root. A ``/models/<id>`` suffix is stripped the same way.
    """
    text = (raw or "").strip().rstrip("/")
    if not text:
        return None
    lower = text.lower()
    for suffix in _API_ROUTE_SUFFIXES:
        if lower.endswith(suffix):
            text = text[: -len(suffix)].rstrip("/")
            lower = text.lower()
            break
    marker = "/models/"
    index = lower.rfind(marker)
    if index != -1:
        text = text[:index].rstrip("/")
    if text and not text.lower().endswith("/v1"):
        logger.warning(
            "provider base_url %s does not end with /v1; " "chat calls go to %s/chat/completions",
            raw,
            text,
        )
    return text or None


def completion_headers(selection: ModelSelection, session_id: Optional[str]) -> dict[str, str]:
    """Headers for one chat call.

    ``user_agent`` is sent as ``User-Agent`` when the provider sets it.
    OpenCode Zen and Go refuse to route a call that omits
    ``x-opencode-session``, so that header carries the conversation id
    whenever the base URL is an ``opencode.ai`` host.
    """
    headers: dict[str, str] = {}
    agent = (selection.user_agent or "").strip()
    if agent:
        headers["User-Agent"] = agent
    host = ""
    base = selection.api_base or ""
    if "://" in base:
        host = base.split("://", 1)[1].split("/", 1)[0].split("@")[-1].split(":")[0]
    host = host.lower()
    session = " ".join((session_id or "").split())
    if session and (host == "opencode.ai" or host.endswith(".opencode.ai")):
        headers["x-opencode-session"] = session
    return headers


_ROUTE_SUFFIXES = {"nitro", "floor", "exacto"}
_ROUTES = {"balanced", *_ROUTE_SUFFIXES}
_REASONING = {"none", "low", "medium", "high"}


def _hostname(api_base: Optional[str]) -> str:
    base = api_base or ""
    if "://" not in base:
        return ""
    return base.split("://", 1)[1].split("/", 1)[0].split("@")[-1].split(":")[0].lower()


def _is_openrouter(api_base: Optional[str]) -> bool:
    host = _hostname(api_base)
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def routed_litellm_model(selection: ModelSelection) -> str:
    """LiteLLM id, with an OpenRouter routing suffix when one is selected."""
    model = selection.litellm_model
    route = (selection.route or "").strip().lower()
    if route in _ROUTE_SUFFIXES and _is_openrouter(selection.api_base):
        return f"{model}:{route}"
    return model


def think_override(selection: ModelSelection) -> Optional[str | bool]:
    """Profile ``think`` override for an OpenRouter reasoning effort."""
    if not _is_openrouter(selection.api_base):
        return None
    effort = (selection.reasoning or "").strip().lower()
    if effort == "none":
        return False
    if effort in {"low", "medium", "high"}:
        return effort
    return None


def apply_turn_options(
    cfg: Any,
    *,
    reasoning: Optional[str] = None,
    route: Optional[str] = None,
) -> ModelSelection:
    """Override reasoning effort and OpenRouter route on the active selection."""
    selection = active_selection(cfg)
    updates: dict[str, str] = {}
    if reasoning and reasoning.strip():
        effort = reasoning.strip().lower()
        if effort not in _REASONING:
            known = ", ".join(sorted(_REASONING))
            raise ValueError(f"Unknown reasoning {reasoning!r}. Choose one of: {known}")
        updates["reasoning"] = effort
    if route and route.strip():
        chosen = route.strip().lower()
        if chosen not in _ROUTES:
            known = ", ".join(("balanced", "nitro", "floor", "exacto"))
            raise ValueError(f"Unknown route {route!r}. Choose one of: {known}")
        updates["route"] = chosen
    if updates:
        selection = replace(selection, **updates)
        cfg._selection_override = selection
    return selection


def openai_litellm_id(model_name: str) -> str:
    """LiteLLM id for an OpenAI-compatible chat model."""
    name = (model_name or "").strip()
    if name.startswith("openai/"):
        return name
    prefix = "openai:"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    return f"openai/{name}"


def profiles_directory(cfg: Any) -> Optional[Path]:
    """Absolute ``agent.profiles_dir``, or None when unset."""
    raw = (getattr(getattr(cfg, "agent", None), "profiles_dir", None) or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def catalog_entries(cfg: Any) -> list[CatalogEntry]:
    """Return the model catalog, building and caching it on *cfg*."""
    cached = getattr(cfg, "_catalog_cache", None)
    if cached is not None:
        return cast(list[CatalogEntry], cached)
    built = _build_catalog(cfg)
    try:
        cfg._catalog_cache = built
    except Exception:
        logger.debug("model catalog cache skipped", exc_info=True)
    return built


def default_selection(cfg: Any) -> ModelSelection:
    """Yaml default, ignoring a per-process model override."""
    entries = catalog_entries(cfg)
    want = (getattr(cfg.agent, "default", None) or "").strip()
    if want:
        match = next((entry for entry in entries if entry.id == want), None)
        if match is None:
            known = ", ".join(entry.id for entry in entries)
            raise ValueError(f"agent.default {want!r} is not in the model catalog ({known})")
        return match.selection()
    return entries[0].selection()


def is_catalog_id(cfg: Any, raw: str) -> bool:
    if not hasattr(cfg, "agent"):
        return False
    text = (raw or "").strip()
    return any(entry.id == text for entry in catalog_entries(cfg))


def catalog_model_or_none(cfg: Any, raw: Optional[str]) -> Optional[str]:
    """Return *raw* when it is a catalog id, else None (keep the default)."""
    text = (raw or "").strip()
    if text and is_catalog_id(cfg, text):
        return text
    return None


def public_catalog(cfg: Any) -> dict[str, Any]:
    """Model list safe to send to the plugin (no credentials)."""
    entries = catalog_entries(cfg)
    default_id = default_selection(cfg).catalog_id
    background = get_background_model(cfg) or default_id
    return {
        "default": default_id,
        "background": background,
        "models": [_public_model(entry) for entry in entries],
    }


def _public_model(entry: CatalogEntry) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": entry.id,
        "provider": entry.provider,
        "model": entry.model,
        "vision": bool(entry.vision),
    }
    if _is_openrouter(entry.api_base):
        item["reasoning"] = ["none", "low", "medium", "high"]
        item["reasoning_default"] = entry.reasoning or "low"
        item["routes"] = ["balanced", "nitro", "floor", "exacto"]
        item["route_default"] = entry.route or "balanced"
    return item


def model_is_vision(cfg: Any, catalog_id: Optional[str]) -> bool:
    """True when *catalog_id* is a configured model flagged ``vision: true``."""
    text = (catalog_id or "").strip()
    if not text:
        return False
    return any(entry.id == text and entry.vision for entry in catalog_entries(cfg))


def apply_model_selection(cfg: Any, raw: str) -> ModelSelection:
    """Point *cfg* at a catalog id, or a legacy model string.

    A catalog id switches model, base URL, API key, and profile.
    Anything else keeps the current profile and credentials and only
    replaces the LiteLLM model string (old ``--model openai:gpt-4o``).
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("model selection is empty")
    match = next((entry for entry in catalog_entries(cfg) if entry.id == text), None)
    if match is not None:
        selection = match.selection()
    else:
        base = default_selection(cfg)
        selection = replace(
            base,
            catalog_id=text,
            litellm_model=_legacy_litellm(text, base.litellm_model),
        )
    cfg._selection_override = selection
    return selection


def active_selection(cfg: Any) -> ModelSelection:
    """Override when set, otherwise the yaml default."""
    override = getattr(cfg, "_selection_override", None)
    if isinstance(override, ModelSelection):
        return override
    return default_selection(cfg)


def background_model_path(cfg: Any) -> Path:
    from pawn_agent.core.sallm_factory import resolve_state_dir  # noqa: PLC0415

    return resolve_state_dir(cfg) / "background_model"


def get_background_model(cfg: Any) -> Optional[str]:
    """Catalog id used when a turn does not name a model.

    Loaded from ``{state_dir}/background_model`` once per process, then kept
    in memory. Missing or unknown files fall back to the configured default
    (including a process ``--model`` override).
    """
    if not hasattr(cfg, "agent"):
        return None
    path = background_model_path(cfg)
    key = str(path)
    cached = _background_memory.get(key)
    if cached:
        return cached
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text and is_catalog_id(cfg, text):
            _background_memory[key] = text
            return text
        logger.warning("Ignoring unknown background model %r in %s", text, path)
    current = active_selection(cfg).catalog_id
    _background_memory[key] = current
    return current


def set_background_model(cfg: Any, catalog_id: str) -> str:
    """Persist the autonomous-work model. Raises ValueError when unknown."""
    text = (catalog_id or "").strip()
    if not is_catalog_id(cfg, text):
        known = ", ".join(entry.id for entry in catalog_entries(cfg)) or "(none)"
        raise ValueError(f"Unknown model {text!r}. Choose one of: {known}")
    path = background_model_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    _background_memory[str(path)] = text
    return text


def reset_background_model(cfg: Any) -> str:
    """Drop the runtime override and return the yaml default id."""
    path = background_model_path(cfg)
    path.unlink(missing_ok=True)
    default_id = default_selection(cfg).catalog_id
    _background_memory[str(path)] = default_id
    return default_id


def try_model_command(cfg: Any, text: str) -> Optional[str]:
    """Handle ``/model``, ``/model <id>``, and ``/model reset``.

    Returns None when *text* is not a model command.
    """
    raw = (text or "").strip()
    head, _, tail = raw.partition(" ")
    if head.lower() != "/model":
        return None
    arg = tail.strip()
    if not arg:
        return format_model_status(cfg)
    if arg.lower() == "reset":
        chosen = reset_background_model(cfg)
        return f"Background model reset to {chosen}."
    try:
        chosen = set_background_model(cfg, arg)
    except ValueError as exc:
        return str(exc)
    return f"Background model set to {chosen}."


def format_model_status(cfg: Any) -> str:
    entries = catalog_entries(cfg)
    default_id = default_selection(cfg).catalog_id
    background = get_background_model(cfg) or default_id
    lines = [
        f"Background model: {background}",
        f"Configured default: {default_id}",
        "",
        "Models:",
    ]
    if entries:
        lines.extend(f"- {entry.id}" + (" (vision)" if entry.vision else "") for entry in entries)
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append(
        "/model <id> sets the background model. /model reset restores the configured default."
    )
    return "\n".join(lines)


def _build_catalog(cfg: Any) -> list[CatalogEntry]:
    agent = cfg.agent
    providers = getattr(agent, "providers", None) or {}
    if providers:
        return _catalog_from_providers(cfg, providers)
    legacy = _legacy_entry(cfg)
    if legacy is not None:
        return [legacy]
    profile = _resolve_optional(cfg, getattr(agent.sallm, "profile", None) or "large.yaml")
    return [
        CatalogEntry(
            id="openai@gpt-4o",
            provider="openai",
            model="gpt-4o",
            profile=profile,
            api_base=None,
            api_key=None,
            user_agent=None,
        )
    ]


def _catalog_from_providers(cfg: Any, providers: dict[str, Any]) -> list[CatalogEntry]:
    entries: list[CatalogEntry] = []
    seen: dict[str, str] = {}
    for provider_name, provider in providers.items():
        name = str(provider_name).strip()
        if not name or "@" in name:
            raise ValueError(f"provider name must not contain '@': {provider_name!r}")
        models = getattr(provider, "models", None) or []
        if not models:
            raise ValueError(f"provider {name!r} has no model profiles")
        for item in models:
            profile_raw = (getattr(item, "profile", None) or "").strip()
            if not profile_raw:
                raise ValueError(f"provider {name!r} has a model entry without a profile")
            resolved = _resolve_required(cfg, profile_raw)
            explicit = (getattr(item, "model", None) or "").strip()
            model_name = explicit or _target_model(resolved)
            if not model_name:
                raise ValueError(
                    f"profile {profile_raw!r} has no target_model; set model: on the entry"
                )
            catalog_id = f"{name}@{model_name}"
            if catalog_id in seen:
                raise ValueError(
                    f"duplicate model id {catalog_id!r} ({seen[catalog_id]} and {profile_raw})"
                )
            seen[catalog_id] = profile_raw
            entries.append(
                CatalogEntry(
                    id=catalog_id,
                    provider=name,
                    model=model_name,
                    profile=str(resolved),
                    api_base=normalize_openai_base_url(getattr(provider, "base_url", None)),
                    api_key=getattr(provider, "api_key", None),
                    user_agent=(getattr(provider, "user_agent", None) or "").strip() or None,
                    reasoning=_choice(
                        getattr(provider, "reasoning", None),
                        _REASONING,
                        "reasoning",
                    ),
                    route=_choice(getattr(provider, "route", None), _ROUTES, "route"),
                    vision=bool(getattr(item, "vision", False)),
                )
            )
    if not entries:
        raise ValueError("agent.providers is set but contains no models")
    return entries


def _legacy_entry(cfg: Any) -> Optional[CatalogEntry]:
    agent = cfg.agent
    for name in _LEGACY_PROVIDER_ORDER:
        block = getattr(agent, name, None)
        if block is None:
            continue
        model_name = (getattr(block, "model", None) or "").strip() or "gpt-4o"
        profile = _resolve_optional(cfg, getattr(agent.sallm, "profile", None))
        return CatalogEntry(
            id=f"{name}@{model_name}",
            provider=name,
            model=model_name,
            profile=profile,
            api_base=normalize_openai_base_url(getattr(block, "base_url", None)),
            api_key=getattr(block, "api_key", None),
            user_agent=None,
        )
    return None


def _resolve_optional(cfg: Any, raw: Optional[str]) -> Optional[str]:
    text = (raw or "").strip()
    if not text:
        return None
    path = resolve_profile_path(text, profiles_dir=profiles_directory(cfg))
    return str(path) if path is not None else None


def _resolve_required(cfg: Any, raw: str) -> Path:
    try:
        path = resolve_profile_path(raw, profiles_dir=profiles_directory(cfg))
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    if path is None:
        raise ValueError(f"sallm profile not found: {raw!r}")
    return path


def _target_model(path: Path) -> str:
    compiled = load_compiled_profile(path)
    return (compiled.target_model or "").strip()


def _choice(raw: Optional[str], allowed: set[str], label: str) -> Optional[str]:
    text = (raw or "").strip().lower()
    if not text:
        return None
    if text not in allowed:
        known = ", ".join(sorted(allowed))
        raise ValueError(f"Unknown {label} {raw!r}. Choose one of: {known}")
    return text


def _legacy_litellm(raw: str, current: str) -> str:
    """Map an old ``--model`` string onto a LiteLLM id, keeping *current*'s prefix."""
    text = raw.strip()
    if "/" in text and ":" not in text.split("/", 1)[0]:
        return text
    if ":" in text:
        prefix, rest = text.split(":", 1)
        if prefix in _LEGACY_PREFIXES:
            return f"{_LEGACY_PREFIXES[prefix]}/{rest}"
    if "/" in current:
        head = current.split("/", 1)[0]
        return f"{head}/{text}"
    return openai_litellm_id(text)

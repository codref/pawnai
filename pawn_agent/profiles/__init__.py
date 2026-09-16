"""Load sallm CompiledProfile artifacts (YAML or JSON)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import yaml
from sallm import CompiledProfile

logger = logging.getLogger(__name__)

# Packaged profiles live next to this module.
_PROFILES_DIR = Path(__file__).resolve().parent


def bundled_profiles_dir() -> Path:
    """Directory of profiles shipped with ``pawn_agent``."""
    return _PROFILES_DIR


def resolve_profile_path(raw: Optional[str]) -> Optional[Path]:
    """Resolve a profile path from config.

    Search order for relative paths: cwd, then ``pawn_agent/profiles/``.
    Empty / None disables the compiled profile overlay.
    """
    text = (raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        if not path.is_file():
            raise FileNotFoundError(f"sallm profile not found: {path}")
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    bundled = (_PROFILES_DIR / path.name).resolve()
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(
        f"sallm profile not found: {text!r} "
        f"(tried {cwd_candidate} and {bundled})"
    )


def load_compiled_profile(path: Path | str) -> CompiledProfile:
    """Load a sallm CompiledProfile from YAML or JSON."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text)
    else:
        # Prefer YAML; fall back to JSON for extensionless paths.
        try:
            data = yaml.safe_load(text) or {}
        except Exception:
            data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"sallm profile must be a mapping: {p}")
    return CompiledProfile(
        target_model=data.get("target_model") or "",
        instructions=dict(data.get("instructions") or {}),
        demonstrations=dict(data.get("demonstrations") or {}),
        budgets=dict(data.get("budgets") or {}),
        metadata=dict(data.get("metadata") or {}),
    )


def load_profile_from_config(raw: Optional[str]) -> Optional[CompiledProfile]:
    """Resolve + load ``agent.sallm.profile``, or None when unset."""
    path = resolve_profile_path(raw)
    if path is None:
        return None
    logger.debug("Loading sallm compiled profile from %s", path)
    return load_compiled_profile(path)

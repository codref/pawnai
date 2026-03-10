"""Template expression resolver for skill YAML step parameters.

Supported expression syntax (double-brace, dot-path)::

    {{input.audio_path}}         # value from the caller-supplied input dict
    {{steps.1.local_path}}       # output field from step 1
    {{steps.2.session_id}}       # output field from step 2
    {{context.session_id}}       # value from the message context dict

Expressions can appear:
- As the sole value of a param (the resolved value replaces the string).
- Embedded inside a larger string: ``"prefix-{{input.name}}-suffix"`` — the
  expression is replaced in-place (always as a string in this case).

Any template that cannot be resolved (missing key) raises
:class:`TemplateResolutionError` to surface data-flow mistakes early.
"""

from __future__ import annotations

import re
from typing import Any, Dict

_EXPR_RE = re.compile(r"\{\{([^}]+)\}\}")


class TemplateResolutionError(Exception):
    """Raised when a ``{{...}}`` expression cannot be resolved."""


def _resolve_path(path: str, ctx: Dict[str, Any]) -> Any:
    """Walk a dot-path into *ctx* and return the value.

    Args:
        path:  Dot-separated key path, e.g. ``"steps.1.local_path"``.
        ctx:   The resolution context dict.

    Returns:
        The value found at the path.

    Raises:
        TemplateResolutionError: If any segment of the path is missing.
    """
    parts = path.strip().split(".")
    current: Any = ctx

    for i, part in enumerate(parts):
        # Support integer keys for the steps sub-dict (step numbers are ints)
        int_key: Any = None
        try:
            int_key = int(part)
        except ValueError:
            pass

        if isinstance(current, dict):
            if part in current:
                current = current[part]
            elif int_key is not None and int_key in current:
                current = current[int_key]
            else:
                resolved_so_far = ".".join(parts[:i])
                available = list(current.keys())
                raise TemplateResolutionError(
                    f"Cannot resolve '{{{{ {path} }}}}': "
                    f"key {part!r} not found in '{resolved_so_far or 'root'}'. "
                    f"Available keys: {available}"
                )
        else:
            raise TemplateResolutionError(
                f"Cannot resolve '{{{{ {path} }}}}': "
                f"expected a dict at '{'.'.join(parts[:i])}', got {type(current).__name__}"
            )

    return current


def resolve_value(value: Any, ctx: Dict[str, Any]) -> Any:
    """Resolve template expressions in *value* using *ctx*.

    - Non-string values are returned unchanged.
    - A string that is *entirely* a single expression (e.g. ``"{{input.x}}"``
      has its resolved value returned as-is (preserving the original type).
    - A string with *embedded* expressions has each expression replaced with
      its string representation.

    Args:
        value: The raw parameter value from the YAML.
        ctx:   Resolution context with ``input``, ``steps``, and ``context`` keys.

    Returns:
        The resolved value.
    """
    if not isinstance(value, str):
        return value

    matches = _EXPR_RE.findall(value)
    if not matches:
        return value

    # Pure expression: return the resolved value preserving its type
    stripped = value.strip()
    sole_match = _EXPR_RE.fullmatch(stripped)
    if sole_match:
        return _resolve_path(sole_match.group(1), ctx)

    # Embedded expression(s): replace each with str() of the resolved value
    def _replacer(m: re.Match) -> str:
        resolved = _resolve_path(m.group(1), ctx)
        return str(resolved)

    return _EXPR_RE.sub(_replacer, value)


def resolve_params(params: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve all template expressions in a flat params dict.

    Args:
        params: Raw params dict from a skill tool-step YAML.
        ctx:    Resolution context.

    Returns:
        New dict with all values resolved.
    """
    return {key: resolve_value(val, ctx) for key, val in params.items()}

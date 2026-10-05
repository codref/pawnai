"""Paths an agent turn wrote, recovered from CliTool observations.

Vault writes happen in a tool subprocess, so the parent only sees stdout.
Success lines from ``note_write``, ``note_append``, ``task_update``, and
``session_analyze --save`` are stable; this module parses those.
"""

from __future__ import annotations

import re
from typing import Any

_MUTATING_TOOLS = frozenset({"note_write", "note_append", "task_update", "session_analyze"})

# ``wrote|appended|updated {key} (etag=…)`` and ``Saved analysis to vault: {key}``.
_PATH_RE = re.compile(
    r"^(?:wrote|appended|updated) (.+?) \(etag=.*$|^Saved analysis to vault: (.+)$",
    re.MULTILINE,
)


def vault_paths_from_steps(steps: Any) -> list[str]:
    """Return vault keys successfully written during *steps* (ask() result).

    Observations that start with ``Error`` are skipped. Order is first-seen.
    """
    paths: list[str] = []
    seen: set[str] = set()
    for step in steps or []:
        if not isinstance(step, dict) or step.get("kind") != "action":
            continue
        for tc in step.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("action") or "")
            if name not in _MUTATING_TOOLS:
                continue
            obs = str(tc.get("observation") or "")
            if obs.lstrip().startswith("Error"):
                continue
            for match in _PATH_RE.finditer(obs):
                key = (match.group(1) or match.group(2) or "").strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                paths.append(key)
    return paths

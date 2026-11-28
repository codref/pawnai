"""Observation types and how they map onto coworker Items."""

from __future__ import annotations

OBSERVATION_TYPES: tuple[str, ...] = (
    "decision",
    "commitment",
    "question",
    "block",
    "contradiction",
    "prediction",
    "concern",
    "fact",
)

# Types that become Inbox items, and the item ``kind`` they get. The others
# (prediction, concern, fact) are stored as observations only.
ITEM_KINDS: dict[str, str] = {
    "decision": "decision",
    "commitment": "commitment",
    "question": "open_question",
    "block": "block",
    "contradiction": "contradiction",
}

# Field that holds the item's owner, per type.
OWNER_FIELD: dict[str, str] = {
    "decision": "owner",
    "commitment": "debtor",
    "block": "owner",
    "question": "asked_by",
}


def enabled_types(configured: list[str] | None) -> list[str]:
    """Configured types in canonical order; unknown names are ignored."""
    wanted = {str(t).strip().lower() for t in (configured or OBSERVATION_TYPES)}
    return [t for t in OBSERVATION_TYPES if t in wanted]

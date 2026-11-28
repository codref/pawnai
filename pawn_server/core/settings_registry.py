"""Allowlist of ``pawnai.yaml`` fields the portal settings panel may read/write.

Only paths listed here are exposed by ``/v1/portal/settings``; everything else
(credentials, providers, buckets, …) stays yaml/env-only. ``apply`` says when a
change takes effect:

- ``live``: read on every call, so patching the running config is enough.
- ``new_chats``: baked into an agent session when it is built; open chats keep
  the old value until they are reset or the server restarts.
- ``restart``: read once when a worker starts (crons, enable flags, loops).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

GROUPS: list[tuple[str, str]] = [
    ("models", "Models"),
    ("agent", "Agent"),
    ("coworker", "Coworker"),
    ("emergence", "Emergence"),
    ("capture", "Capture & research"),
    ("vault", "Vault"),
    ("scheduler", "Scheduler"),
    ("matrix", "Matrix bot"),
    ("portal", "Portal"),
]

AUTONOMY_MODES = ["off", "suggest_only", "approve_writes", "limited_act"]

# A virtual field: the background model lives in ``{state_dir}/background_model``
# (shared with the ``/model`` chat command), not in the override file.
BACKGROUND_MODEL = "agent.default"


@dataclass(frozen=True)
class SettingField:
    path: str
    label: str
    group: str
    type: str  # bool | int | float | str | text | enum | list | cron | tz | model
    help: str = ""
    apply: str = "live"  # live | new_chats | restart
    secret: bool = False
    choices: Optional[list[str]] = None
    nullable: bool = False
    tz_path: Optional[str] = None  # cron fields: which field holds the timezone
    min: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


FIELDS: list[SettingField] = [
    # ── models ──
    SettingField(
        BACKGROUND_MODEL,
        "Background model",
        "models",
        "model",
        "Used by autonomous work (scheduler, coworker, Matrix, vault watcher) and when a "
        "turn names no model. Same setting as the /model chat command.",
    ),
    SettingField(
        "capture.model",
        "Research capture model",
        "models",
        "model",
        "Model for capture_enrich jobs. Empty uses the background model.",
        nullable=True,
    ),
    # ── agent ──
    SettingField("agent.name", "Agent name", "agent", "str", "How Pawn calls itself.", apply="new_chats"),
    SettingField(
        "agent.anima",
        "Anima file",
        "agent",
        "str",
        "Path to the persona Markdown file. Empty for none.",
        apply="new_chats",
        nullable=True,
    ),
    SettingField(
        "agent.strip_thinking",
        "Strip thinking",
        "agent",
        "bool",
        "Remove <think> blocks from answers.",
        apply="new_chats",
    ),
    SettingField(
        "agent.sallm.max_steps",
        "Max tool steps",
        "agent",
        "int",
        "ReAct steps per turn before the agent must answer.",
        apply="new_chats",
        min=1,
    ),
    # ── coworker ──
    SettingField(
        "coworker.enabled",
        "Coworker enabled",
        "coworker",
        "bool",
        "Extract items after sessions and run the morning and weekly crons.",
        apply="restart",
    ),
    SettingField(
        "coworker.timezone",
        "Timezone",
        "coworker",
        "tz",
        "Day boundaries and cron times. The cron worker picks it up on restart.",
        apply="restart",
    ),
    SettingField(
        "coworker.briefing_cron",
        "Morning briefing",
        "coworker",
        "cron",
        "Cron for the Today.md briefing.",
        apply="restart",
        tz_path="coworker.timezone",
    ),
    SettingField(
        "coworker.weekly_cron",
        "Weekly review",
        "coworker",
        "cron",
        "Cron for the weekly review note.",
        apply="restart",
        tz_path="coworker.timezone",
    ),
    SettingField(
        "coworker.me",
        "Me",
        "coworker",
        "list",
        "Names and speaker labels that mean you (owner matching).",
    ),
    SettingField(
        "coworker.commitment_days",
        "Commitment age (days)",
        "coworker",
        "int",
        "An open commitment older than this is flagged.",
        min=1,
    ),
    SettingField(
        "coworker.stale_days",
        "Stale after (days)",
        "coworker",
        "int",
        "Threads with no movement for this long are stale.",
        min=1,
    ),
    SettingField(
        "coworker.autonomy.mode",
        "Autonomy",
        "coworker",
        "enum",
        "How far the coworker may act on its own.",
        choices=AUTONOMY_MODES,
    ),
    SettingField(
        "coworker.people.refresh_after_session",
        "Refresh People after sessions",
        "coworker",
        "bool",
        "Append Appearances and propose Facts on People notes after diarization.",
    ),
    SettingField("coworker.notify.ntfy_url", "ntfy URL", "coworker", "str", "ntfy server for push alerts."),
    SettingField("coworker.notify.topic", "ntfy topic", "coworker", "str"),
    SettingField(
        "coworker.notify.token",
        "ntfy token",
        "coworker",
        "str",
        "Write-only. Leave empty to keep the current token.",
        secret=True,
    ),
    # ── emergence ──
    SettingField(
        "emergence.enabled",
        "Emergence enabled",
        "emergence",
        "bool",
        "Run the observation layer during coworker extraction and the night shift. "
        "Needs the coworker loop for the night cron.",
        apply="restart",
    ),
    SettingField(
        "emergence.fast_model",
        "Per-window model",
        "emergence",
        "model",
        "Model for triage and typed extraction over each window. Empty uses the "
        "background model.",
        nullable=True,
    ),
    SettingField(
        "emergence.types",
        "Observation types",
        "emergence",
        "list",
        "Types to extract: decision, commitment, question, block, contradiction, "
        "prediction, concern, fact.",
    ),
    SettingField(
        "emergence.always_types",
        "Always-extract types",
        "emergence",
        "list",
        "Extracted from every window whatever triage says.",
    ),
    SettingField(
        "emergence.think_types",
        "Thinking types",
        "emergence",
        "list",
        "Types whose extraction lets the model reason first (2-5k tokens a call).",
    ),
    SettingField(
        "emergence.triage_threshold",
        "Triage threshold",
        "emergence",
        "float",
        "A type is extracted from a window when triage scores it at least this (0-1).",
        min=0,
    ),
    SettingField(
        "emergence.quote_min_score",
        "Quote match minimum",
        "emergence",
        "float",
        "Observations whose quote matches the window less than this are rejected (0-1).",
        min=0,
    ),
    SettingField(
        "emergence.window_chars",
        "Window size (chars)",
        "emergence",
        "int",
        "Characters per extraction window (~4 per token).",
        min=1000,
    ),
    SettingField(
        "emergence.window_overlap",
        "Window overlap (chars)",
        "emergence",
        "int",
        "Characters shared between consecutive windows.",
        min=0,
    ),
    SettingField(
        "emergence.few_shot",
        "Few-shot from labels",
        "emergence",
        "bool",
        "Add retrieved examples from your verdicts to each typed prompt.",
    ),
    SettingField(
        "emergence.ranker",
        "Relevance ranker",
        "emergence",
        "bool",
        "Score observations with the ranker trained on your labels.",
    ),
    SettingField(
        "emergence.ranker_min_score",
        "Ranker filing threshold",
        "emergence",
        "float",
        "Items scoring below this are kept as observations but not filed. 0 scores only. "
        "Applies once the ranker reaches its minimum AUC.",
        min=0,
    ),
    SettingField(
        "emergence.directed",
        "Directed pass",
        "emergence",
        "bool",
        "Ask each window about nearby open subjects (threads, questions, boards).",
    ),
    SettingField(
        "emergence.night_cron",
        "Night shift",
        "emergence",
        "cron",
        "Cron for the night jobs, run inside the coworker loop.",
        apply="restart",
        tz_path="coworker.timezone",
    ),
    SettingField(
        "emergence.night_jobs",
        "Night jobs",
        "emergence",
        "list",
        "Jobs the night shift runs, in priority order.",
    ),
    SettingField(
        "emergence.night_max_minutes",
        "Night budget (minutes)",
        "emergence",
        "int",
        "Wall-clock budget for one night shift.",
        min=1,
    ),
    SettingField(
        "emergence.night_max_calls",
        "Night budget (model calls)",
        "emergence",
        "int",
        "Model calls one night shift may make.",
        min=1,
    ),
    SettingField(
        "emergence.brief_max_per_day",
        "Provocations per morning",
        "emergence",
        "int",
        "Provocations at the head of the morning brief.",
        min=0,
    ),
    SettingField(
        "emergence.dream_max_hypotheses",
        "Hunches per night",
        "emergence",
        "int",
        "Dreaming hypotheses kept per night.",
        min=0,
    ),
    SettingField(
        "emergence.people_tone",
        "People tone trend",
        "emergence",
        "bool",
        "Track a tone trend on People notes (opt-in).",
    ),
    # ── capture ──
    SettingField(
        "capture.auto_enrich",
        "Auto-enrich research captures",
        "capture",
        "bool",
        "Queue a capture_enrich job for each research snippet.",
    ),
    SettingField(
        "capture.auto_file",
        "Auto-file enriched captures",
        "capture",
        "bool",
        "File atoms into their proposed collection without asking.",
    ),
    SettingField(
        "capture.instructions",
        "Research instructions",
        "capture",
        "text",
        "Extra guidance for the research_capture skill.",
    ),
    # ── vault ──
    SettingField(
        "vault.auto_push_transcript",
        "Auto-push transcripts",
        "vault",
        "bool",
        "Write the session note to the vault after each diarized chunk.",
    ),
    SettingField(
        "vault.screenshot_vision",
        "Describe screenshots",
        "vault",
        "bool",
        "Caption screenshot changes with the background vision model.",
    ),
    SettingField(
        "vault.agent_root",
        "Agent root",
        "vault",
        "str",
        "Vault folder Pawn writes under. Changing it moves nothing.",
        apply="restart",
    ),
    # ── scheduler ──
    SettingField(
        "agent_scheduler.enabled",
        "Scheduler enabled",
        "scheduler",
        "bool",
        "Run durable schedules inside pawn-server serve.",
        apply="restart",
    ),
    SettingField(
        "agent_scheduler.default_timezone",
        "Default timezone",
        "scheduler",
        "tz",
        "Timezone for schedules that do not name one.",
    ),
    # ── matrix ──
    SettingField(
        "matrix_bot.enabled",
        "Matrix bot enabled",
        "matrix",
        "bool",
        "Start the Matrix chat bot with pawn-server serve.",
        apply="restart",
    ),
    SettingField(
        "matrix_bot.command_prefix",
        "Room command prefix",
        "matrix",
        "str",
        "Messages in group rooms must start with this.",
        apply="restart",
    ),
    SettingField(
        "matrix_bot.notify_room_id",
        "Notify room",
        "matrix",
        "str",
        "Room id for outbound ready-for-review alerts.",
        apply="restart",
        nullable=True,
    ),
    # ── portal ──
    SettingField(
        "portal.auto_name",
        "Auto-name chats",
        "portal",
        "bool",
        "Let the background model title and tag chats until you edit them.",
    ),
    SettingField(
        "portal.auto_name_every",
        "Re-name every N messages",
        "portal",
        "int",
        "0 names a chat after its first answer only.",
        min=0,
    ),
    SettingField(
        "portal.index_images",
        "Caption images for search",
        "portal",
        "bool",
        "Index vault images by a caption from the vision background model.",
    ),
    SettingField(
        "portal.index_interval_seconds",
        "Index reconcile interval (s)",
        "portal",
        "int",
        "Seconds between vault index passes. 0 disables the loop.",
        apply="restart",
        min=0,
    ),
]

BY_PATH: dict[str, SettingField] = {f.path: f for f in FIELDS}


def get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def nested(path: str, value: Any) -> dict[str, Any]:
    """``"a.b.c", v`` → ``{"a": {"b": {"c": v}}}``."""
    out: Any = value
    for part in reversed(path.split(".")):
        out = {part: out}
    return out


def dict_get(data: dict[str, Any], path: str) -> tuple[bool, Any]:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def dict_pop(data: dict[str, Any], path: str) -> bool:
    """Remove *path* and prune parents left empty. True when it existed."""
    parts = path.split(".")
    chain: list[dict[str, Any]] = [data]
    cur: Any = data
    for part in parts[:-1]:
        if not isinstance(cur, dict) or not isinstance(cur.get(part), dict):
            return False
        cur = cur[part]
        chain.append(cur)
    if parts[-1] not in cur:
        return False
    del cur[parts[-1]]
    for depth in range(len(parts) - 1, 0, -1):
        if not chain[depth]:
            del chain[depth - 1][parts[depth - 1]]
    return True


def env_name(path: str) -> str:
    return "PAWN_" + path.upper().replace(".", "__")

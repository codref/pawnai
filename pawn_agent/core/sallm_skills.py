"""Pawn skill modes for the embedded sallm agent.

Skills are *modes* (prompt + visible tool subset), not capabilities.
The controller picks a skill each turn; ReAct then only sees that skill's tools.

Owns: Skill / SkillRegistry definitions for pawn domain work.
Does not own: CliTool registration (see sallm_tools.py) or Agent construction.
"""

from __future__ import annotations

from sallm import Skill, SkillRegistry

# tools=None → every CliTool is visible (same as sallm's default CONVERSE).
# An empty tools=() hid sessions_list when the controller kept "converse",
# so prompts like "list all sessions" failed with "I don't have a tool…".
CONVERSE = Skill(
    name="converse",
    description=(
        "Default chat and general requests. Also use when the user asks to "
        "list, browse, or discover stored diarization sessions."
    ),
    prompt=(
        "Active skill: converse.\n"
        "Stay helpful and concise. Do not invent diarization session ids.\n"
        "When the user asks to list / show / find stored sessions, call "
        "sessions_list via a ```run block (optionally --limit N).\n"
        "For transcripts or analysis use session_transcript / session_analyze.\n"
        "To delete a diarization session, ALWAYS ask the user to confirm the "
        "exact session name in chat first, then call:\n"
        "```run\n"
        "session_delete --session-id <id> --confirm <id>\n"
        "```\n"
        "Never call session_delete without that matching confirmation.\n"
        "Do NOT pass --save / do NOT call siyuan_save unless the user "
        "explicitly asks to save or push to SiYuan Notes.\n"
        "When they do ask, run exactly:\n"
        "```run\n"
        "siyuan_save --session-id <id> --from-analysis\n"
        "```\n"
        "Never paste long Markdown into --content; never print tool argv as prose."
    ),
    tools=None,
)

SESSIONS = Skill(
    name="sessions",
    description=(
        "User wants to list, discover, inspect, quote, summarize, or delete "
        "diarization conversation sessions / transcripts. Prefer this "
        "(push/replace) whenever the request mentions sessions, transcripts, "
        "or session analysis."
    ),
    prompt=(
        "Active skill: sessions.\n"
        "Use sessions_list / session_transcript / session_analyze / "
        "session_delete via ```run blocks.\n"
        "Never invent session ids — list first when the id is unclear.\n"
        "session_analyze persists to the DB only; do NOT pass --save unless "
        "the user explicitly asks for SiYuan.\n"
        "Before session_delete, ALWAYS ask the user to confirm the exact "
        "session name in chat. Only then call:\n"
        "```run\n"
        "session_delete --session-id <id> --confirm <id>\n"
        "```\n"
        "(--confirm must exactly equal --session-id). Never delete without "
        "that confirmation.\n"
        "When the conversation key itself is a diarization session name "
        "(typical for queue/scheduler runs), prefer that id for transcript tools.\n"
        "Keep answers short; prefer summaries over dumping full transcripts."
    ),
    tools=(
        "sessions_list",
        "session_transcript",
        "session_analyze",
        "session_delete",
    ),
)

NOTES = Skill(
    name="notes",
    description=(
        "User wants a session summary/analysis saved into SiYuan Notes "
        "(store / save / push to SiYuan)."
    ),
    prompt=(
        "Active skill: notes.\n"
        "Tools MUST be called inside ```run fences — never print bare argv.\n"
        "After session_analyze (or when an analysis already exists), save with:\n"
        "```run\n"
        "siyuan_save --session-id <id> --from-analysis\n"
        "```\n"
        "Do NOT paste the summary into --content (it breaks). "
        "Or use session_analyze --session-id <id> --save in one step."
    ),
    tools=("siyuan_save", "session_analyze"),
)

SCHEDULING = Skill(
    name="scheduling",
    description=(
        "User wants to create, update, pause, resume, or cancel durable "
        "scheduled agent work (proposals only)."
    ),
    prompt=(
        "Active skill: scheduling.\n"
        "Use schedule_propose via ```run. This only creates a proposal — "
        "it never applies changes. The application must approve proposals.\n"
        "For create/update pass --schedule as a JSON object string with "
        "schedule_kind once|interval|cron and a session_id."
    ),
    tools=("schedule_propose",),
)

OPS = Skill(
    name="ops",
    description=(
        "User wants a progress update or notification pushed to an external "
        "queue target (e.g. Matrix)."
    ),
    prompt=(
        "Active skill: ops.\n"
        "Use queue_push via ```run only when the user explicitly asks to "
        "notify / publish / enqueue.\n"
        "Pass --payload as a JSON object string without a 'command' key."
    ),
    tools=("queue_push",),
)


def build_pawn_skills() -> SkillRegistry:
    """Return the pawn skill registry (converse is registered explicitly)."""
    return SkillRegistry([CONVERSE, SESSIONS, NOTES, SCHEDULING, OPS])

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
        "To rename a speaker on a session (SPEAKER_XX or a wrong display name):\n"
        "```run\n"
        "session_relabel --session-id <id> --from SPEAKER_00 --to Davide\n"
        "```\n"
        "Resolve the session id first (sessions_list / ask). This updates "
        "transcript labels and propagates the name across embeddings. "
        "Existing SiYuan diary pages refresh automatically; if the user "
        "also wants SiYuan updated and no diary page exists yet, add "
        "--push-siyuan.\n"
        "To delete a diarization session, ALWAYS ask the user to confirm the "
        "exact session name in chat first, then call:\n"
        "```run\n"
        "session_delete --session-id <id> --confirm <id>\n"
        "```\n"
        "Never call session_delete without that matching confirmation.\n"
        "Do NOT pass --save / do NOT call siyuan_save unless the user "
        "explicitly asks to save or push to SiYuan Notes.\n"
        "If they ask for a (deep) analysis AND SiYuan, resolve a real "
        "diarization session id first (sessions_list / ask the user), then "
        "ONE call only:\n"
        "```run\n"
        "session_analyze --session-id <id> --save\n"
        "```\n"
        "Do not also call siyuan_save; do not analyze twice; do not write a "
        "second free-form note for the same request.\n"
        "If analysis already exists and they only ask to save it:\n"
        "```run\n"
        "siyuan_save --session-id <id> --from-analysis\n"
        "```\n"
        "Free-form / chat-thread Markdown only when it is NOT a session "
        "analysis and the user asked to save that substance:\n"
        "```run\n"
        'siyuan_save --session-id <id> --title "…" --content-file @note\n'
        "```\n"
        "```file note\n"
        "…full markdown…\n"
        "```\n"
        "Never paste Markdown into --content; never invent session ids; "
        "never print tool argv as prose.\n"
        "NEVER save tool failures, --help output, fallback recipes, or "
        "agent self-troubleshooting into SiYuan. If session_analyze cannot "
        "run (no diarization id), ask which session to use — do not invent a "
        "meta-note about the tooling problem."
    ),
    tools=None,
)

SESSIONS = Skill(
    name="sessions",
    description=(
        "User wants to list, discover, inspect, quote, summarize, relabel "
        "speakers on, or delete diarization conversation sessions / "
        "transcripts. Prefer this (push/replace) whenever the request "
        "mentions sessions, transcripts, speaker rename, or session analysis."
    ),
    prompt=(
        "Active skill: sessions.\n"
        "Use sessions_list / session_transcript / session_analyze / "
        "session_relabel / session_delete via ```run blocks.\n"
        "Never invent session ids — list first when the id is unclear.\n"
        "session_analyze persists to the DB only; do NOT pass --save unless "
        "the user explicitly asks for SiYuan. When they want analysis AND "
        "SiYuan, still prefer a single session_analyze --save (notes skill).\n"
        "To rename / correct a speaker (e.g. SPEAKER_00 → Davide):\n"
        "```run\n"
        "session_relabel --session-id <id> --from SPEAKER_00 --to Davide\n"
        "```\n"
        "--from may be a raw SPEAKER_XX label or the wrong display name shown "
        "in the transcript. This updates segments and propagates the name to "
        "embeddings via speaker_names. Existing SiYuan diary Speakers+"
        "Transcript pages refresh automatically (Annotations kept). If the "
        "user also asks to update SiYuan and no diary page exists yet:\n"
        "```run\n"
        "session_relabel --session-id <id> --from SPEAKER_00 --to Davide "
        "--push-siyuan\n"
        "```\n"
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
        "session_relabel",
        "session_delete",
    ),
)

NOTES = Skill(
    name="notes",
    description=(
        "User wants a session summary/analysis or other Markdown saved into "
        "SiYuan Notes (store / save / push to SiYuan). Prefer this when the "
        "request combines analysis with saving to SiYuan."
    ),
    prompt=(
        "Active skill: notes.\n"
        "Tools MUST be called inside ```run fences — never print bare argv.\n"
        "session_analyze / --from-analysis need a real diarization session id "
        "(not the Matrix/API chat key). When the id is unclear, call "
        "sessions_list first; if still ambiguous, ask the user which session.\n"
        "ONE SiYuan write per user request. Never call session_analyze twice. "
        "Never call both session_analyze --save and siyuan_save in the same turn.\n"
        "Pick exactly one recipe:\n"
        "A) User wants a (deep) session analysis AND SiYuan — single call only:\n"
        "```run\n"
        "session_analyze --session-id <id> --save\n"
        "```\n"
        'Do NOT also write a free-form note. "Deep" still means this one call.\n'
        "B) Analysis already in DB; user only asks to save it:\n"
        "```run\n"
        "siyuan_save --session-id <id> --from-analysis\n"
        "```\n"
        "C) Free-form note of user-facing chat substance (NOT session analysis, "
        "never --content). Only when the user asked to save that content and "
        "recipe A/B do not apply:\n"
        "```run\n"
        'siyuan_save --session-id <id> --title "…" --content-file @note\n'
        "```\n"
        "```file note\n"
        "…full markdown about the discussion topics…\n"
        "```\n"
        "Hard rules for SiYuan content:\n"
        "- Save the substance of the conversation / transcript analysis.\n"
        "- NEVER save tool probing, unknown-tool errors, missing --session-id, "
        "--help output, fallback patterns, iterative verification, or other "
        "agent self-troubleshooting.\n"
        "- If recipe A cannot run because no diarization session is identified, "
        "STOP and ask which session — do not invent a meta-analysis of the "
        "tooling problem as a free-form note.\n"
        '- Repeated "analyze and save" means re-do recipe A on the same '
        "session (or ask), not a new note about process maturity.\n"
        "After the tool succeeds, answer briefly; do not re-analyze or re-save."
    ),
    tools=("sessions_list", "siyuan_save", "session_analyze"),
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

SIYUAN_TASKS = Skill(
    name="siyuan_tasks",
    description=(
        "SiYuan @pawn task: read linked notes, optionally deep-analyze "
        "diarization sessions, append a reviewable draft under a parent "
        "block, and set custom-agent status to review."
    ),
    prompt=(
        "Active skill: siyuan_tasks.\n"
        "You are fulfilling a SiYuan @pawn instruction delivered by the "
        "watcher. Tools MUST be called inside ```run fences.\n"
        "1) siyuan_read --block-id <id> [--include-children] [--resolve-refs] "
        "to load context; follow explicit ((block refs)) only.\n"
        "2) If the note names a diarization session, resolve via "
        "sessions_list then session_transcript / session_analyze — never "
        "invent session ids.\n"
        "3) Append the draft under parent_block_id:\n"
        "```run\n"
        "siyuan_append --parent-id <parent> --as-result --request-id <uuid> "
        "--content-file @note\n"
        "```\n"
        "```file note\n"
        "…structured highlights: tasks, decisions, risks…\n"
        "```\n"
        "4) Mark the trigger block review:\n"
        "```run\n"
        "siyuan_set_status --block-id <trigger> --status review "
        "--request-id <uuid> --output-id <new_block_id>\n"
        "```\n"
        "Hard rules: append-only; never delete/overwrite human blocks; never "
        "dump tool errors into SiYuan; include the Approve for Pawn memory "
        "checklist (via --as-result)."
    ),
    tools=(
        "siyuan_read",
        "siyuan_append",
        "siyuan_set_status",
        "sessions_list",
        "session_transcript",
        "session_analyze",
    ),
)


def build_pawn_skills() -> SkillRegistry:
    """Return the pawn skill registry (converse is registered explicitly)."""
    return SkillRegistry(
        [CONVERSE, SESSIONS, NOTES, SCHEDULING, OPS, SIYUAN_TASKS]
    )

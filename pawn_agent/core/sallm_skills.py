"""Pawn skill modes for the embedded sallm agent.

Skills are *modes* (prompt + visible tool subset), not capabilities.
The controller picks a skill each turn; ReAct then only sees that skill's tools.

Owns: Skill / SkillRegistry definitions for pawn domain work.
Does not own: CliTool registration (see sallm_tools.py) or Agent construction.
"""

from __future__ import annotations

from sallm import Skill, SkillRegistry

# tools=None → every CliTool is visible (same as sallm's default CONVERSE).
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
        "transcript labels and propagates the name across embeddings.\n"
        "Existing vault transcript notes refresh automatically; if the user "
        "also wants a vault note and none exists yet, add --push-vault.\n"
        "To delete a diarization session, ALWAYS ask the user to confirm the "
        "exact session name in chat first, then call:\n"
        "```run\n"
        "session_delete --session-id <id> --confirm <id>\n"
        "```\n"
        "Never call session_delete without that matching confirmation.\n"
        "Do NOT pass --save / do NOT call note_write unless the user "
        "explicitly asks to save or push to vault notes.\n"
        "If they ask for a (deep) analysis AND vault save, resolve a real "
        "diarization session id first, then ONE call only:\n"
        "```run\n"
        "session_analyze --session-id <id> --save\n"
        "```\n"
        "Never paste Markdown into tool args; never invent session ids; "
        "never print tool argv as prose.\n"
        "If session_analyze cannot run (no diarization id), ask which session "
        "to use — do not invent meta-notes about tooling problems."
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
        "Do NOT pass --save unless the user explicitly asks for vault notes. "
        "When they want analysis AND vault, prefer a single "
        "session_analyze --save (notes skill).\n"
        "To rename / correct a speaker (e.g. SPEAKER_00 → Davide):\n"
        "```run\n"
        "session_relabel --session-id <id> --from SPEAKER_00 --to Davide\n"
        "```\n"
        "--from may be a raw SPEAKER_XX label or the wrong display name shown "
        "in the transcript. This updates segments and propagates the name to "
        "embeddings via speaker_names. Existing vault Speakers+Transcript "
        "notes refresh automatically. If the user also asks to create a vault "
        "note and none exists yet:\n"
        "```run\n"
        "session_relabel --session-id <id> --from X --to Y --push-vault\n"
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
        "User wants to save, write, or update vault Markdown notes "
        "(including session analysis --save). Prefer this when the request "
        "combines analysis with saving to the vault."
    ),
    prompt=(
        "Active skill: notes.\n"
        "Tools MUST be called inside ```run fences — never print bare argv.\n"
        "session_analyze needs a real diarization session id "
        "(not the Matrix/API chat key). When the id is unclear, call "
        "sessions_list first; if still ambiguous, ask the user which session.\n"
        "ONE vault write per user request. Never call session_analyze twice. "
        "Never call both session_analyze --save and note_write for the same "
        "analysis in the same turn.\n"
        "A) User wants a (deep) session analysis AND vault — single call only:\n"
        "```run\n"
        "session_analyze --session-id <id> --save\n"
        "```\n"
        "B) Free-form note under Pawn/:\n"
        "```run\n"
        'note_write --path "Pawn/Notes/Title.md" --content-file @note\n'
        "```\n"
        "plus a ```file note block (sallm writes a temp file). Do not paste "
        "long bodies into --content.\n"
        "Hard rules for vault content:\n"
        "- Write freely under Pawn/; outside that root only when the note "
        "has pawn: editable.\n"
        "- Never touch .obsidian/.\n"
        "- Never dump tool errors / --help / fallback journaling into notes.\n"
        "If no diarization session is identified for analysis, STOP and ask.\n"
        "After the tool succeeds, answer briefly; do not re-analyze."
    ),
    tools=(
        "sessions_list",
        "session_analyze",
        "note_read",
        "note_search",
        "note_write",
        "note_append",
    ),
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

VAULT_TASKS = Skill(
    name="vault_tasks",
    description=(
        "Vault task: read linked notes, optionally deep-analyze diarization "
        "sessions named in the instruction, and write the result via "
        "task_update. Prefer for queue/watcher vault_run turns."
    ),
    prompt=(
        "Active skill: vault_tasks.\n"
        "You are fulfilling a vault task instruction delivered by the "
        "watcher or HTTP fast path.\n"
        "1) note_read --path <key> [--follow-links 1] for the linked note "
        "and any [[wiki links]] in the instruction/context.\n"
        "2) For diarization sessions named in the note, use sessions_list / "
        "session_transcript / session_analyze — never invent session ids.\n"
        "3) When done, update the task:\n"
        "```run\n"
        "task_update --task-id <uuid> --status review --result-file @note\n"
        "```\n"
        "plus a ```file note block with the Result markdown.\n"
        "Do not delete or overwrite human notes. Do not dump tool errors "
        "into the vault. Prefer writing under Pawn/; outside that root only "
        "when pawn: editable is set."
    ),
    tools=(
        "note_read",
        "note_search",
        "note_write",
        "note_append",
        "task_update",
        "sessions_list",
        "session_transcript",
        "session_analyze",
    ),
)


def build_pawn_skills() -> SkillRegistry:
    """Return the pawn skill registry (converse is registered explicitly)."""
    return SkillRegistry([CONVERSE, SESSIONS, NOTES, SCHEDULING, OPS, VAULT_TASKS])

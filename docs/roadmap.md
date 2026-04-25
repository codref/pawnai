# Pawn-Agent Roadmap

This document captures the near-term direction for evolving `pawn-agent` from a
user-driven transcript query assistant into a proactive personal knowledge
agent. The guiding idea is simple: Pawn should keep the current diarize, query,
and SiYuan workflow, while gaining enough autonomy to notice useful patterns,
suggest actions, and eventually run bounded background analysis without waiting
for a direct chat prompt.

## Current Baseline

Pawn already has several foundations needed for autonomy:

- `pawn-diarize` can create diarized, transcribed sessions.
- `pawn-agent` can query conversations, analyze sessions, search indexed
  knowledge, remember explicit facts, and save generated content to SiYuan.
- `pawn-server` exposes the agent through an OpenAI-compatible API and a
  queue listener.
- The queue listener can run prompts for a specific diarization session and
  persist execution history in `agent_runs`.
- LangGraph session state is persisted in Postgres, so conversations can
  survive server restarts.

The missing layer is not model intelligence. The missing layer is initiative:
event sources, notification policy, approval gates, and a bounded autonomous
loop.

## Phase 1: Matrix Push Notifications

The first feature should be Matrix push notifications. Pawn is already usable
through chat, so Matrix is the natural place for the agent to surface useful
insights without requiring the user to poll.

Initial notification use cases:

- A diarization session completed.
- A standard analysis was generated.
- A session contains likely decisions, tasks, risks, follow-ups, or useful
  SiYuan notes.
- A queued agent job completed or failed.
- The agent needs approval before taking a write action.

Suggested notification shape:

```text
New session analyzed: project-sync-2026-04-25

Detected:
- 3 possible follow-up tasks
- 2 decisions
- 1 SiYuan-worthy summary

Actions:
[Show tasks] [Save summary] [Ignore]
```

Implementation direction:

- Add a Matrix notification client, configured from `pawnai.yaml`.
- Add a notification abstraction so Matrix is one transport, not hard-coded
  into analysis logic.
- Emit notifications from queue jobs and future autonomous jobs.
- Record notification attempts and outcomes for debugging.
- Start with text-only messages, then add action handling through Matrix
  replies or commands.

## Phase 2: Autonomous Background Analysis Loop

The next step is a bounded autonomous loop for background analysis. This should
not be an unrestricted agent that continuously acts. It should be an event-driven
worker that evaluates specific events, decides whether work is worthwhile, and
submits explicit jobs.

Core events:

- `session.completed`
- `session.analysis.completed`
- `siyuan.page.updated`
- `matrix.message.received`
- `schedule.daily_review`
- `schedule.weekly_review`

Initial autonomous jobs:

- Detect tasks, commitments, and follow-ups from a session.
- Detect decisions and open questions.
- Detect entities: people, projects, tools, places, recurring themes.
- Suggest links to existing SiYuan pages.
- Suggest whether a session should be indexed for semantic search.
- Produce a daily or weekly briefing.

The loop should always produce auditable records:

- input event
- prompt used
- retrieved context
- proposed action
- policy decision
- final action or notification
- errors and retries

## Phase 3: Agent Self-Queue Interaction

Pawn-agent should be able to interact with its own queue. This gives the agent a
safe way to break larger analysis into bounded jobs instead of trying to do
everything inside one LLM turn.

The important primitive is:

```json
{
  "command": "run",
  "prompt": "Search recent sessions for unresolved decisions about Project X.",
  "session_id": "project-sync-2026-04-25",
  "model": "optional:model-override"
}
```

Future queue commands could include:

- `notify`: send a Matrix notification.
- `analyze_session`: run a specific analysis template.
- `search`: run cross-session or SiYuan knowledge search.
- `daily_review`: summarize recent activity.
- `index`: vectorize a session or SiYuan page.
- `propose_action`: create a pending action requiring approval.

Recommended design constraints:

- The agent can enqueue bounded jobs, not arbitrary shell commands.
- Every self-enqueued job must include a parent `agent_run_id` or causal event
  ID.
- Add loop protection: max depth, max jobs per event, max jobs per day.
- Add duplicate suppression so the agent does not repeatedly ask itself the
  same search.
- Require approval for external writes until trust is earned.

This turns the queue into the agent's task substrate. The autonomous loop can
simply push prompts to the queue, while the existing queue listener remains the
execution engine.

## Autonomy Policy

Autonomy should be configurable. A useful first config shape:

```yaml
agent_autonomy:
  enabled: true
  mode: suggest_only          # off | suggest_only | approve_writes | limited_act
  min_insight_score: 0.72
  max_notifications_per_day: 5
  max_self_enqueued_jobs_per_event: 3
  require_approval_for_siyuan_writes: true
  require_approval_for_matrix_replies: false
```

Suggested policy levels:

- `off`: no autonomous work.
- `suggest_only`: analyze and notify, but do not write to SiYuan or enqueue
  follow-up jobs without explicit approval.
- `approve_writes`: run analysis and queue follow-up searches, but require
  approval for persistent writes.
- `limited_act`: allow low-risk actions automatically, with rate limits and
  audit logs.

## State Of The Art: Life Coach Agent

A "life coach agent" is a higher-level version of this roadmap: an agent that
observes daily interactions, builds a private model of the user's goals and
patterns, and gives timely leverage rather than generic advice.

Current state of the art is strongest in these areas:

- **Tool use**: models can call structured tools, query databases, save notes,
  schedule tasks, and use APIs.
- **Retrieval**: agents can search personal knowledge stores and ground advice
  in actual history.
- **Long-context reasoning**: large models can process longer transcripts and
  synthesize themes across conversations.
- **Workflow orchestration**: LangGraph-style systems can persist state, pause
  for approval, resume later, and keep execution auditable.
- **Multimodal input**: audio, text, screenshots, and documents can all become
  part of the personal context stream.

The weak points are still important:

- **Reliability**: agents can over-interpret conversations or infer intent too
  strongly.
- **Memory quality**: long-term memory needs curation, expiration, correction,
  and source links.
- **Privacy**: a life coach agent handles extremely sensitive data and must be
  local-first or tightly controlled.
- **Consent**: proactive analysis can feel intrusive unless the user controls
  what is observed and when the agent speaks.
- **Evaluation**: "good coaching" is hard to test. The system needs feedback
  loops, not only better prompts.

For Pawn, the realistic path is not to build a vague motivational chatbot. The
interesting target is a grounded life-operations agent:

- It knows what happened because it has transcripts and notes.
- It knows what matters because the user explicitly teaches it preferences and
  goals.
- It speaks up only when there is a concrete useful intervention.
- It links every insight back to source material.
- It distinguishes facts, interpretations, and suggestions.
- It asks before writing, escalating, or changing durable memory.

Examples of "superpowers" such an agent could provide:

- Detect commitments the user forgot to capture.
- Notice repeated unresolved topics across weeks.
- Identify when a conversation created a decision but no owner.
- Suggest reconnecting with someone after a meaningful exchange.
- Build a map of projects, people, and open loops from conversations.
- Turn scattered voice notes into structured SiYuan pages.
- Warn when current plans conflict with previous stated priorities.
- Produce a weekly personal operating review grounded in real interactions.

The most valuable version of this agent is not always-on advice. It is quiet,
contextual augmentation: a second memory, a pattern detector, and a patient
operator that helps the user convert lived conversations into durable knowledge
and better next actions.

## Suggested First Milestone

Build the smallest end-to-end proactive loop:

1. A session finishes diarization.
2. A queue job runs background analysis for high-value insights.
3. The result is scored.
4. If the score passes the threshold, Pawn sends a Matrix notification.
5. The user can reply with a command such as `save`, `tasks`, `ignore`, or
   `more`.
6. The selected action is handled through the existing queue listener and
   recorded in `agent_runs`.

This milestone establishes the core product behavior: Pawn notices something,
speaks up in Matrix, waits for lightweight direction, and uses its own queue to
continue the work.

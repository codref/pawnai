# Speaker people vault + autonomous analysis

## Summary

Give Pawn a durable, Obsidian-native model of people that improves after every
diarization session and every agent conversation — without stuffing facts into
per-chat sallm memory.

Identity (voice + stable id) stays in the Speakers gallery. Bios, links, tags,
and conversation history live as vault notes under `People/`, written with the
same curated / section-managed discipline as transcript notes and coworker
items. After diarization (and optionally after chat), Pawn queues a bounded
speaker-analysis pass that proposes or applies updates under autonomy policy.

## Goals

- One person note per gallery speaker, navigable with Obsidian tags + wikilinks.
- Transcripts and analyses backlink into person notes (and vice versa).
- Agent sessions (CLI, Matrix, Obsidian, queue) all read/write the same store
  via tools — not conversation-scoped `Agent.remember`.
- Autonomy: after a meeting, Pawn can refresh people knowledge without a human
  prompt, within coworker autonomy caps.
- Never auto-enroll voiceprints; never silently invent people.

## Non-goals (v1)

- Replacing the Speakers gallery or changing identify thresholds.
- Auto-enrolling embeddings from unmatched `SPEAKER_XX` clusters.
- Dumping full bios into every system prompt.
- Cross-conversation sallm memory merge.
- Full CRM (email, phone sync, contact import).

---

## Architecture

```
diarize / chat turn
        │
        ▼
 chain_agent / session_completed  ──► coworker items (existing)
        │
        └─► enqueue speakers_refresh (new command)
                    │
                    ▼
           extract people facts from transcript
                    │
                    ├─ resolve → Speakers gallery (id / create proposal)
                    ├─ ensure People/{Name}.md
                    ├─ append Appearances + Facts (or propose)
                    └─ wikilink from transcript Speakers table
```

Two stores, one person:

| Concern | Store | Why |
|---------|--------|-----|
| Voice identity, aliases for matching | Postgres `speakers` / `speaker_enrollments` | Needed by reidentify; small, curated |
| Bio, relationship, history, tags | Vault `People/{Name}.md` | Obsidian graph, Sync Engine, user-editable |
| Short card for tools | `speakers.notes` (optional mirror of Summary) | Fast `speakers_show` without vault read |

---

## Vault note contract

### Path

- Default folder: `People/` at vault root (sibling of `Ideas/`, `Goals.md`).
- Config: `coworker.people_dir` (or top-level `people.dir`) — default `People`.
- Filename: gallery `display_name` sanitized like idea titles
  (`People/Davide.md`). Rename of gallery display name renames/moves the note
  (or leaves a stub wikilink — decide in implementation; prefer move +
  Obsidian-compatible redirect line).

Why root `People/` not `Pawn/People/`: first-class in the user’s graph; same
pattern as Ideas. Writes go through a dedicated people writer (`skip_guards`),
not free `note_write`.

### Frontmatter

```yaml
---
pawn: person
speaker_id: davide
aliases: [Dave]
tags: [person, coworker]
status: active          # active | archived
updated: 2026-10-06
---
```

- `pawn: person` marks the note type (scanner/tools). It is **not** a free
  `pawn: editable` blank check — only the people writer may mutate managed
  sections.
- `speaker_id` links to gallery row. Missing id = vault-only stub until linked.
- `tags` always include `person`; agent/coworker may add topical tags
  (`#hiring`, `#family`) under autonomy rules.
- Optional: `me: true` when the person is in `coworker.me`.

### Body sections (managed vs preserved)

Mirror transcript note discipline:

| Section | Owner | Behaviour |
|---------|--------|-----------|
| `# {Name}` title | system | Synced to display_name |
| `## Summary` | system (agent may rewrite under autonomy) | Short card, ≤ ~8 lines |
| `## Facts` | system append | Dated bullets with source wikilink |
| `## Appearances` | system rewrite/append | List of `[[Pawn/Transcripts/…]]` (+ date, role) |
| `## Notes` | **user preserved** | Freeform; never overwritten by Pawn |
| Optional `## Open loops` | system append | Soft pointers into items/threads |

Example:

```markdown
---
pawn: person
speaker_id: davide
aliases: [Dave]
tags: [person, cofounder]
updated: 2026-10-06
---

# Davide

## Summary

Cofounder. Usually owns infra decisions. Prefers async updates.

## Facts

- 2026-10-06: Agreed S3 stays through Friday. ([[Pawn/Transcripts/2026-10-06 monday-standup.md]])
- 2026-09-20: Introduced as voice gallery speaker `davide`.

## Appearances

- [[Pawn/Transcripts/2026-10-06 monday-standup.md]] — 12m talk · 2026-10-06
- [[Pawn/Analyses/monday-standup.md]]

## Notes

(user annotations stay here)
```

### Backlinks from transcripts

Extend vault transcript push so the Speakers table (or a line under it) uses
wikilinks when a gallery mapping exists:

`| [[People/Davide]] | 12m 04s | 18 |`

Unresolved `SPEAKER_XX` stay plain text. Relabel/reidentify refresh updates
links on next push (existing auto-refresh path).

Analyses saved under `Pawn/Analyses/` should mention people with the same
wikilink form when the analyze skill knows the gallery id.

---

## Gallery bridge

### Existing

- `speakers create|show|list|enroll|…`
- Agent CliTools: `speakers_list`, `speaker_enroll`, `session_reidentify`,
  `session_relabel`

### Add

| Tool / CLI | Purpose |
|------------|---------|
| `speakers_show` | Expose notes, aliases, linked vault path |
| `speakers_update` | Set aliases / short `notes` (card) after confirm or autonomy allow |
| `people_ensure` | Create stub `People/{Name}.md` + optional gallery row |
| `people_show` | Read person note (or Summary+Facts slice) |
| `people_append` | Append Facts / Appearances; never touch `## Notes` |
| `people_link_session` | Add Appearance + optional Fact from a session id |

Skills (`converse`, `sessions`, `notes`, `coworker`): when a named person is
discussed, prefer `people_show` / `speakers_show` over inventing biography.

`speakers.notes` = optional one-paragraph mirror of Summary for offline/tool
speed; vault remains source of truth for history.

---

## Autonomous loop

### Trigger

After successful `transcribe-diarize` when `chain_agent` runs:

1. Keep existing `session_completed` → coworker extract/score (unchanged).
2. **Also** enqueue `command: speakers_refresh` with
   `{session_id, depth, event_id}` (same lineage / self-job caps as research).

Optional later: after interactive chat turns that mention people heavily —
out of scope for v1 unless cheap heuristics appear.

Config sketch:

```yaml
coworker:
  enabled: true
  people_dir: People
  people:
    refresh_after_session: true   # enqueue speakers_refresh
    create_stubs: true            # new gallery names → stub note
    auto_tags: true               # topical tags under limited_act
  autonomy:
    mode: limited_act             # or suggest_only
    auto_actions:
      - research
      - people_refresh            # new
```

### `speakers_refresh` handler

Deterministic-first, LLM second (same spirit as coworker extract):

1. Load session segments + `session_speaker_map` / display names.
2. For each resolved gallery speaker (or confidently named label):
   - `people_ensure` stub if missing.
   - Append Appearance wikilink to transcript (and analysis if present).
3. Run a small structured extract (reuse `llm_sub` / coworker extract style):
   - New durable facts about each person (role, preference, commitment *about*
     them, relationship) — not meeting action items (those stay Items).
   - Proposed aliases.
   - Proposed tags.
4. Autonomy `decide(cfg, "people_refresh", …)`:
   - `deny` → no-op (log).
   - `needs_approval` → write proposal note under
     `Pawn/Reviews/people-{session}.md` or an item `kind: people_update`,
     notify like research.
   - `allow` → `people_append` Facts + update Summary if short delta;
     `speakers_update` aliases when high confidence.
5. Never create gallery voice enrollments here.
6. Never create a new gallery person without either:
   - `create_stubs` + a human-assigned display name already on the session, or
   - approval path for “new person?” proposals.

### Interaction with coworker items

- People refresh does **not** replace item extract.
- Facts that are commitments stay Items; person note gets a Fact + optional
  “see item” link only when useful.
- `coworker.me` person notes can be richer (self model) but still curated.

### Caps

Reuse `reject_reason` / `max_self_jobs_*` / `max_depth`. One
`speakers_refresh` per session event by default. Fingerprint
`people_refresh:{session_id}` to avoid duplicates on re-queue.

---

## Write / authority model

Aligned with schedule proposals and enroll:

| Action | Default authority |
|--------|-------------------|
| Append Appearance after diarize | Automatic when refresh enabled (low risk) |
| Append Fact with source link | `limited_act` + `people_refresh` in `auto_actions`, else propose |
| Rewrite Summary | Same as Facts; keep short |
| Create gallery person | Propose unless name already curated |
| Voice enroll | Always human confirm (existing) |
| Edit `## Notes` | Never (user only) |
| Free `note_write` on `People/` | Denied (no `pawn: editable`; dedicated tools only) |

Autonomy helper today denies `writes_outside_pawn`. People writes must use a
**dedicated code path** (like `capture_idea` / Goals slash) that sets
`writes_outside_pawn=False` for the policy check by treating
`people_refresh` as an allowlisted action_kind, not generic note_write.

---

## Agent UX

### Skills

- On “who is X?” → `people_show` then gallery show; do not hallucinate.
- On “remember that X …” → `people_append` (or propose).
- On session review → offer enroll only after quality confirm; separately
  refresh people notes if not already queued.

### Slash / Matrix (optional v1.1)

- `/people` list stubs.
- `/people Davide` show summary.
- Triage words for people proposals: `approve` / `reject` (reuse item actions).

### Plugin

- No required UI for v1; Obsidian graph + tags are enough.
- Later: Inbox cards for `kind: people_update` proposals.

---

## Implementation phases

### Phase 0 — Contract + docs

- Lock note path, frontmatter, section names.
- Document in `docs/SPEAKERS.md` (or new `docs/PEOPLE.md`) and link from
  `docs/COWORKER.md`.
- Config keys + example yaml.

### Phase 1 — Read/write plumbing

- People note helpers: ensure, parse sections, append Facts/Appearances,
  preserve Notes (unit tests like `test_vault_transcript.py`).
- CliTools: `people_show`, `people_ensure`, `people_append`, `speakers_show`,
  `speakers_update`.
- Skill prompt updates.
- Transcript Speakers table wikilinks when mapped.

### Phase 2 — Autonomous refresh

- Queue command `speakers_refresh`.
- Wire from diarize `chain_agent` alongside `session_completed` (or as a step
  inside process_session — prefer separate command for isolation/retry).
- Extract prompt + structured JSON schema for facts/aliases/tags.
- Autonomy + proposal notes + fingerprinting.
- Tests: refresh dry-run, section preserve, autonomy deny/allow.

### Phase 3 — Conversation linking polish

- Analysis save inserts person wikilinks.
- Relabel/reidentify triggers Appearance rename / transcript link refresh
  (mostly free via existing push).
- Optional: weekly coworker pass that consolidates Facts into Summary.

### Phase 4 — Chat-triggered refresh (optional)

- After Matrix/Obsidian turns that introduce durable person facts, enqueue a
  lightweight `people_append` proposal. Only if Phase 2 is stable.

---

## Testing plan

- Unit: section merge preserves `## Notes`; duplicate Appearance no-ops;
  wikilink formatting; filename slug.
- Unit: autonomy allow/deny/propose for `people_refresh`.
- Integration: fake session → refresh → People note + Speakers table links.
- Regression: Ideas/`Goals.md` write guards unchanged; enroll still confirm-only.
- Manual: Obsidian graph shows People ↔ Transcripts; tags filter `person`.

---

## Open decisions

1. **Folder**: confirm `People/` at vault root (recommended) vs `Pawn/People/`.
2. **New unnamed speakers**: stub as `People/SPEAKER_00-session.md` (noisy) vs
   only refresh resolved gallery names (recommended for v1).
3. **Proposal surface**: `Pawn/Reviews/people-*.md` vs coworker item
   `kind: people_update` (prefer item for Matrix triage reuse).
4. **Summary rewrites**: append-only Facts forever vs periodic Summary
   compaction job.
5. **Rename policy**: move file on gallery rename vs stable filename from
   `speaker_id` (`People/davide.md` with title Davide) — **prefer id-based
   filename** for stable wikilinks; display name only in H1.

Recommendation on (5): use `People/{speaker_id}.md` as the stable key, H1 =
display_name, and optional `aliases` for search. Obsidian display can use the
H1; wikilinks stay stable across renames.

---

## Success criteria

- After three meetings with the same enrolled speaker, their People note has
  Appearances + sourced Facts without manual note_write.
- Asking “who is Davide?” in a fresh Matrix room yields vault+gallery context
  via tools, not empty memory.
- User `## Notes` never clobbered.
- Voice enroll still requires explicit confirmation.
- Autonomy off / suggest_only never silently rewrites bios.

# People notes

Vault bios for Speakers gallery people. Voice identity stays in Postgres;
durable knowledge lives under `People/{speaker_id}.md` so every agent
conversation (CLI, Matrix, Obsidian, queue) can read the same record.

See also: [SPEAKERS.md](SPEAKERS.md), [plans/speaker-people-vault.md](plans/speaker-people-vault.md).

## Why two stores

| Store | Holds |
|-------|--------|
| Speakers gallery | Stable id, display name, aliases, voice enrollments, short notes card |
| `People/{id}.md` | Summary, Facts, Appearances, tags, user Notes |

Chat memory (sallm) is **per conversation key** and is not used as the people CRM.

## Note contract

Path: `{coworker.people_dir}/{speaker_id}.md` (default `People/davide.md`).
Filename is the gallery id so Obsidian wikilinks survive renames. H1 is the
display name.

```markdown
---
pawn: person
speaker_id: davide
aliases: [Dave]
tags: [person, cofounder]
status: active
updated: 2026-10-06
---

# Davide

## Summary

…

## Facts

- 2026-10-06: … ([[Pawn/Transcripts/…]])

## Appearances

- [[Pawn/Transcripts/…]] — 12m talk · 2026-10-06

## Notes

(user-owned — Pawn never overwrites this section)
```

`pawn: person` is **not** a free `pawn: editable` pass. Only the people writer
(and CliTools `people_*`) may mutate managed sections.

## Autonomy

After diarization, when `coworker.enabled` and
`coworker.people.refresh_after_session` (default true):

1. `session_completed` runs the coworker item loop.
2. `speakers_refresh` ensures stubs, appends Appearances, extracts durable facts.
3. Autonomy `people_refresh`:
   - `limited_act` + `people_refresh` in `auto_actions` → write Facts
   - otherwise → coworker item `kind: people_update` (approve / reject)

Voice enroll is never automatic.

```yaml
coworker:
  enabled: true
  people_dir: People
  people:
    refresh_after_session: true
    refresh_after_chat: false   # propose only when true
    create_stubs: true
    auto_tags: true
  autonomy:
    mode: limited_act
    auto_actions: [research, people_refresh]
```

## Agent tools

| Tool | Purpose |
|------|---------|
| `speakers_list` / `speakers_show` / `speakers_update` | Gallery card |
| `people_show` / `people_ensure` / `people_append` | Vault bio |
| `speaker_enroll` / `session_reidentify` | Voice (confirm first) |

### CLI (preferred for a one-off)

```bash
# People bios only — does NOT create Pawn/Items
pawn-server coworker people-refresh --session <diarization-session-id>

# Coworker meeting items only — does NOT refresh People
pawn-server coworker process --session <diarization-session-id>
```

Queue command (agent topic, not Matrix): `speakers_refresh` with
`{session_id, force?}`.

## Obsidian tips

- Tag filter: `tag:#person`
- Graph: Speakers table on transcripts uses `[[People/davide|Davide]]`
- Keep private thoughts under `## Notes`

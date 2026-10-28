# Day planner

The Day tab is a meeting-first calendar for one local day: Outlook (or manual)
events, diarization sessions, open items tied to those sessions, and LLM prep.

There is **no Microsoft Graph sync**. Work calendar intake is a one-gesture
clip from Outlook Web via the browser extension.

## Event notes

Vault notes under `{agent_root}/Events/` (default `Pawn/Events/`):

```yaml
---
pawn: event
id: evt_…
title: Product weekly
start: 2026-10-28T10:00   # local wall time (coworker.timezone)
end: 2026-10-28T10:30
source: outlook           # outlook | manual | capture
source_url: https://outlook.office.com/…
attendees: [Ada, Bob]
organizer: Ada
location: Conf Room B
meeting_url: https://teams.microsoft.com/…
topics: []
session_id: null          # linked diarization session
fingerprint: …            # upsert key (URL or title+start)
prep_job_id: null
prep_at: null
---
# Product weekly

## Prep
_(written by meeting_prep)_
```

Capture / Outlook clip also enriches frontmatter from invite text when the
browser tab title is a stub (`Calendar | Microsoft Teams`, …): subject,
`(Organizer|Required|Optional)` people, times, location, and Teams/Zoom links.
When the clip includes an image (whole-view / region) and fields are still
thin, the server runs one vision call on the background model (must be
flagged `vision: true` in `agent.providers`) to fill title / attendees /
times / location.

Config:

```yaml
planner:
  events_dir: "{agent_root}/Events"
```

Upsert is idempotent on `fingerprint` (Outlook URL preferred).

## API

| Endpoint | Role |
|----------|------|
| `GET /v1/planner/day?date=YYYY-MM-DD` | Merged timeline for one day |
| `GET /v1/planner/events?from=&to=` | Event range |
| `POST /v1/planner/events` | Create / upsert |
| `POST /v1/planner/events/{id}/link` | Set or clear `session_id` |
| `DELETE /v1/planner/events/{id}` | Delete the event note |
| `POST /v1/planner/events/{id}/prep` | Enqueue `meeting_prep` job (202) |

Capture target `kind: event` on `POST /v1/captures` also upserts an event
(title/start/end + snippet body).

## Day tab (Obsidian)

Pawn pane → **Day**: prev / today / next, date picker, **Add event**, timeline
blocks (event and/or session), nested open items, link suggestions, **Prep**,
**Unlink**, **Delete** (confirm). **Sessions** stays for train / extract /
reprocess.

Offline: vault scan of `Pawn/Events/` + `Pawn/Transcripts/` by date. Prep and
link need the server.

## Chat / slash

`/event <title> [when]` writes a manual event (no model turn), e.g.
`/event Coffee with Ada 2026-10-29T09:00`.

## Outlook (browser extension)

1. Open the meeting in Outlook Web.
2. Side panel → **Add to planner** (or target **Planner event…** and Save).
3. **Add to planner** scrapes title / times / people / meeting link into the
   Capture target and stages a lean tray note (join URL / meeting id /
   passcode). It does **not** Save — review the tray, then click Save.
   Chrome labels (`Tracking`, `Chat`, …) are filtered. The server also
   re-parses clip text / screenshots so a later whole-view can upgrade a stub.
4. Later clips (selection / whole-view screenshot) on the same **Planner event**
   target append into that note under `Events/assets/` (snippet markers), not a
   new stub.

Hosts: `outlook.office.com`, `outlook.office365.com`, `outlook.live.com`,
`outlook.cloud.microsoft`.

## Session linking

Same-day title token overlap suggests a link; you confirm. Heuristics are only
for this join UX — not for prep content.

## Meeting prep (LLM-first)

**Prep** enqueues job kind `meeting_prep` (background model). Skill
`meeting_prep` uses `sessions_list`, `session_transcript`, `people_show`,
`note_search`, `knowledge_search`, `note_read`, and writes only via
`event_prep_update` into `## Prep` (wikilinks, sparse). No heuristic “related
notes” list is shown as prep.

## Related

- [BROWSER_CAPTURE.md](BROWSER_CAPTURE.md) — capture targets including `event`
- [COWORKER.md](COWORKER.md) — items / Goals / Today.md (attention inbox, not agenda)
- [OBSIDIAN_AGENT.md](OBSIDIAN_AGENT.md) — plugin pane

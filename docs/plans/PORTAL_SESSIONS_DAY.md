# Plan: Sessions, Day and Speakers in the Pawn Portal

Status: proposal, 2026-11-27. Scope: bring the Obsidian plugin's **Sessions** and **Day** views into `portal/`, plus a new **Speakers** gallery. This is not a straight port. Each view is reworked to remove the plugin's modal and vault-hopping steps.

Decisions already made:

- Sessions get a browse tab plus one tab per session (`session:{id}`). Train, Extract and Reprocess move into the session tab. There is no Train modal any more.
- Audio playback is in scope. It is proxied through pawn-server with HTTP Range support.
- The Day view gets drag to move and resize, a week view, a "Plan my day" side chat and an inline event editor.
- The Speakers gallery view is in scope.

---

## 1. What the plugin does today (parity checklist)

### Sessions (`obsidian-plugin/pawn/src/sessions/*`)

| Feature | Plugin | Portal plan |
|---|---|---|
| List, grouped by day | Cards, 80 max | Dense rows, sticky day headers, infinite scroll |
| Scope All / Needs speakers | Chips | Segmented control with count |
| Speaker facet filter | Chips (top 12) | Chips plus a "more" menu, with the selected speaker pinned first |
| Search | Runs on Enter only | Debounced live search (title, id, speakers, tags, topics) |
| Badges (needs training / trained / analyzed) | Yes | Yes, plus item count and linked event |
| Summary, tags, topics, sentiment | Yes | Yes (row shows one line, session tab shows all) |
| Open transcript | Opens the vault note | Native transcript in the session tab; "Open note" stays as a secondary action |
| Chat about this session | `note:{path}` conversation | Portal side chat with `origin.kind = "session"` |
| Show items | Jumps to the Inbox filter | Items section inside the session tab, with triage verbs |
| Extract items/people | Modal | Small dialog in the session tab |
| Reprocess (rediarize/retranscribe) | Modal | Small dialog; the session shows "processing" until the note refreshes |
| Train speakers | Modal: map, create, enroll, reidentify, relabel, ignore, mark trained | Speakers rail in the session tab with audio samples |
| Offline vault scan | Yes | **Dropped**: the portal is served by pawn-server, so it is never offline from it |

### Day (`obsidian-plugin/pawn/src/day/*`)

| Feature | Plugin | Portal plan |
|---|---|---|
| Prev / Today / Next, date picker | Yes | Yes, plus keyboard (←/→, `t`) |
| 30-minute grid, 72 px slots, overlap columns | Yes | 15-minute snap, compact hour rows, the same overlap algorithm (ported and unit-tested) |
| Click a slot to add an activity (modal) | Yes | Drag or click to create with an inline quick-create popover |
| Sessions as 30-minute chips | Yes | True duration from `duration_seconds`, with a distinct style |
| All-day/untimed and Other TODOs strips above the grid | Yes | Moved to a right-hand **agenda rail**, so the grid keeps its height |
| Focus star + prep band | Frontmatter edit in Obsidian | `PATCH` on the server |
| Prep (LLM job) | Jumps to the Jobs tab | Runs in place: a spinner on the card, and the prep renders in the editor drawer |
| Link suggestion / Unlink | Yes | Yes, in the agenda rail and the editor drawer |
| Delete | Confirm modal | Confirm dialog |
| Now line, auto-scroll | Yes | Yes |
| Week view, drag move/resize, editor, Plan my day | — | **New** |

---

## 2. UX principles for the three views

1. **Do the job where you are.** No modals for frequent work (training, linking, focus, prep). Dialogs only for rare, option-heavy actions (Reprocess, Extract, Merge speakers, Delete).
2. **Everything is a target.** Session rows, transcript turns, events and speakers get `v-target`, so Ask about this, Annotate, Open to the side, Open in a float and Copy link work as they do everywhere else.
3. **Open, don't navigate.** Single-click opens a preview tab and double-click pins it, the same as chats and files. Sessions, the Day view and speakers can sit side by side in splits, for example the Day view on the left and a session on the right.
4. **Live, not refresh buttons.** Vault events (`files.startEvents`) and job status refresh the views. Refresh buttons stay only as a fallback.
5. **Keyboard first.** `j`/`k` move through lists, `Enter` opens, and each view has a few single-key verbs, listed in a `?` cheat sheet per panel.

---

## 3. Phase 0: shared foundations (do first, it touches shared files)

**Tab kinds** (`portal/src/stores/tabs.ts`):
- Add `"sessions"` (tool, `TOOL_REF`), `"session"` (ref = session id), `"day"` (tool) and `"speakers"` (tool). The Speakers detail lives inside its tool tab (master-detail), so there is no `speaker:` kind.
- Update `KINDS`, saved-workspace hydration (unknown kinds are already dropped, so new ones need adding), `TabBar.vue` icons/titles, `Workspace.vue` panel switch (async components) and `GroupView`/`HomePanel` if they enumerate kinds.

**Router** (`portal/src/router.ts`):
- Routes `/sessions`, `/session/:id`, `/day?date=&view=` and `/speakers?id=`.
- Helpers `openSessions`, `openSession(id, opts)`, `openDay({date, view, eventId})`, `openSpeakers(id?)`, extend `openTarget` and `linkFor`. The Day tab keeps the date and view in its URL query, so links like "Day view for 2026-11-27" work.

**Header** (`App.vue`): primary icons **Day** (`calendar_today`), **Sessions** (`graphic_eq`, badge = `needs_speakers` count) and **Inbox**. **Speakers** and **Settings** go into an overflow menu. The header has no room for five more round buttons. Add global shortcuts: `g d`, `g s`, `g i`, `g p` (people/speakers).

**Targets and side chats** (`lib/targets.ts`, `lib/actions.ts`, `pawn_server/core/portal_api.py::_origin_context`):
- `TargetKind` gains `"event"`, `"speaker"` and `"day"` (`"session"` exists already). Annotations: `AnnotationKind` gains `"event"` (`"session"` exists already).
- `_origin_context` handles the new kinds:
  - `session`: inline the transcript note when it is mapped. Otherwise add a short header (title, date, speakers, session id) and the hint "use `session_transcript --session-id X`". If the origin has `t` (seconds), name the timestamp.
  - `event`: inline the event note (includes `## Prep`).
  - `day`: a compact text render of `build_day(date)` plus "Use the day_plan skill". Skills are chosen by the agent, so a plain instruction in the context is enough.
  - `speaker`: the People note if it exists, plus the gallery summary.
- The session header lists existing side chats for the same session (chats whose `origin.ref` = session id), so "Chat about this" reopens the last one instead of always creating a new one.

**Live refresh bus** (`stores/files.ts` already consumes vault events): add `onVaultChange(prefix, fn)` so the sessions store listens to `Pawn/Transcripts/` and the day store to `Pawn/Events/` and `Pawn/Daily/`. Debounce 500 ms.

**Job status in the portal**: the portal has no job event consumer yet. Add `stores/jobs.ts` over `GET /v1/jobs/events` (SSE) with `watchJob(id) → Promise<Job>`. Prep and capture-enrich spinners use it.

**API layer**: `api/types.ts`, `api/client.ts` and `api/mock.ts` for every endpoint below. `VITE_PAWN_MOCK=1` must stay fully usable, including a generated silent WAV for the audio player.

**Shared components** (extract, don't duplicate):
- `ItemCard.vue`, from `InboxPanel.vue`. It is reused by the session Items section and the Day agenda.
- `SpeakerChip.vue`: colored initials avatar. Anonymous labels show as `?1`, `?2`, and the color is stable per name/label.
- `TimeGrid` primitives in `lib/timegrid.ts`: overlap layout (port `layoutOverlaps` from `DayView.ts`), minute↔px mapping and snapping. These are pure functions with vitest tests.

---

## 4. Phase 1: Sessions backend

All new code goes in `pawn_server/core/sessions_api.py`, with thin routes in `api_server.py`. Existing endpoints are reused unchanged.

### 4.1 List improvements (`GET /v1/sessions`)
- Add `offset` (infinite scroll), `from`/`to` (dates), `sort=recent|oldest|duration` and `has_analysis`.
- Facets also return `tags` (top 20) and `total`.
- Add `items_open` (count of open coworker items) and `event` (`{id, title}` when an event links to the session) to each row.

### 4.2 Session detail (`GET /v1/sessions/{id}`)
Extend the existing payload (the Train modal fields stay):
- `analysis`: summary, topics, tags, sentiment and key points from the latest `SessionAnalysis`, plus `analysis_path` when a `Pawn/Analyses/` note exists.
- `labels[].samples`: the three longest segments per label (`{start, end, text}`), to play during training.
- `labels[].talk_seconds` and `labels[].share`.
- `labels[].suggested_speaker_id` / `suggested_score`, from the reidentify identify hit, when not mapped.
- `event`, `items_open`, `side_chats` (ids + titles).
- `processing`: true while a rediarize/retranscribe job is queued or running for the session (from the diarize queue claim, or a marker written by `reprocess`).

### 4.3 Transcript (`GET /v1/sessions/{id}/transcript`) — new
```json
{
  "session_id": "...",
  "duration": 3412.5,
  "files": [{"index": 0, "audio_file": "s3://…", "offset": 0.0, "duration": 1800.2, "mime": "audio/webm"}],
  "speakers": [{"label": "SPEAKER_00", "name": "Ada", "speaker_id": "ada", "ignored": false}],
  "turns": [{"id": "seg-first-id", "label": "SPEAKER_00", "start": 12.3, "end": 48.0,
             "segments": [{"id": "...", "start": 12.3, "end": 20.1, "text": "...", "words": [...]?}]}],
  "captures": [{"kind": "screenshot|annotation", "t": 301.0, "text": "...", "image_key": "Pawn/…"}]
}
```
- Consecutive segments of the same label merge into turns on the server, so the client stays simple.
- Words are included only with `?words=1`, to keep the payload small.
- Captures come from `session_captures`, so screenshots and annotations show inline at their time.

### 4.4 Audio (`GET /v1/sessions/{id}/audio/{index}`) — new
- Proxies S3 `get_object` with the request's `Range` header. Responds `206` with `Content-Range`, `Accept-Ranges: bytes`, and `Content-Type` from the extension.
- **Auth**: `<audio src>` cannot send a bearer header. `POST /v1/sessions/{id}/audio-url` returns per-file URLs signed with HMAC(api.token, `session_id|index|exp`), valid 6 h. The GET accepts either the bearer token or a valid `sig`/`exp`. No raw S3 presigned URLs: the bucket endpoint (MinIO) is often not reachable from the browser.
- **Offsets (spike first)**: `session_state.processed_files` stores only paths, and `file_offsets` from `pawn_core/transcription.py` is not persisted. Plan:
  1. Migration `0028_session_file_offsets`: add a JSONB `file_offsets` column to `session_state`, written by `combined.py` from now on (it already computes `file_offsets` plus `time_cursor`).
  2. Legacy sessions: probe each file's duration once (ffprobe/soundfile on a ranged download), cache it in the same column, and accumulate offsets in `processed_files` order.
- The player maps global time to `(file index, local time)` and switches `<audio>` sources at file boundaries.
- Codec check: webm/opus, ogg, m4a and wav play natively in Chromium and Firefox. Anything else (for example raw PCM) gets `415` and a note in the UI. No server-side transcoding in v1.

### 4.5 Bulk (`POST /v1/sessions/reidentify`) — new
`{session_ids?: [...], needs_speakers?: true, limit: 50}` → runs `reidentify_session_api` for each session in a background task and returns a job id. It is used for "Reidentify N other sessions" after a new enrollment.

### 4.6 Tests
`tests/test_sessions_api.py` (new or extended): transcript turn merging, sample selection, Range responses (206/416), signature expiry, offset fallback, bulk reidentify, list paging/facets.

---

## 5. Phase 2: Sessions UI

### 5.1 `SessionsPanel.vue` (tool tab `sessions:main`)
```
┌ [search…………………] [All | Needs speakers 4] [This week ▾]  ⟳ ┐
│ speakers: (Ada 12) (Bob 7) (Chiara 5) … more ▾                 │
├ TODAY ─────────────────────────────────────────────────────────┤
│ ● Product weekly       10:02 · 42m   AB ?1   3 items  needs ▲  │
│   One-line summary…                                             │
│ ● Call with Bob        15:40 · 12m   B        trained           │
├ YESTERDAY ─────────────────────────────────────────────────────┤
```
- Rows are compact (two lines). Hover shows **Open**, **Chat** and **Train**. Right-click adds **Extract…**, **Reprocess…**, **Reidentify** and **Open note**, plus the shared target actions.
- Multi-select (checkbox on hover, Shift-click ranges) with bulk **Reidentify** and **Mark trained**.
- Store `stores/sessions.ts`: query state, paging, facets and the `needs_speakers` count. The header badge reads the count from here, refreshed on vault events under the transcripts dir.
- Keys: `j`/`k`, `Enter` (open), `o` (open to the side), `t` (train: opens the session tab with the speakers rail focused), `/` (search).

### 5.2 `SessionPanel.vue` (tab `session:{id}`)
```
┌ Product weekly · Thu 27 Nov 10:02 · 42m · ▲ needs training · 📅 linked event ┐
│ [Chat about this ▾] [Extract…] [Reprocess…] [⋯ Open note, Copy link]          │
├──────────────────────────────────────────────────┬─────────────────────────────┤
│ ▶ ◀10 10▶  1.5×  ━━━━━●━━━━━━━━━  12:03 / 42:10  │ SPEAKERS        2 of 3 named│
│ ▓▓░░▓▓▓░░░▓▓▒▒▓ (who-spoke-when strip, colored)  │ ● Ada   18m 43%  ▶ ✓enrolled│
│ [Transcript] [Summary] [Items 3]   🔍 find        │ ● ?1    15m 36%  ▶          │
│                                                  │   “…sample text…”           │
│ 00:12 Ada                                        │   [Map to… ▾] [Enroll]      │
│   Text of the turn…  (current segment lit)       │   [This session only] [Ign] │
│ 00:48 ?1                                         │ ● Bob   9m  21%  ▶ suggested│
│   …                                              │ ─────────────────────────── │
│ [screenshot 05:01 inline]                        │ [Reidentify] [Mark trained] │
└──────────────────────────────────────────────────┴─────────────────────────────┘
```
**Player** (`components/AudioPlayer.vue`, `lib/audio.ts`):
- Multi-file source switching, speed 1×/1.25×/1.5×/2×, ±10 s skips, and space to play/pause while the panel is focused.
- The who-spoke-when strip doubles as the scrubber, with each turn colored by speaker.
- One player at a time across tabs: starting one pauses the others (small global store).

**Transcript**:
- Turns render with a speaker chip and a clickable timestamp (seek and play).
- The current segment is highlighted with the CSS Custom Highlight API, the same mechanism as annotations, so the DOM is never wrapped. "Follow" auto-scroll turns off as soon as the user scrolls.
- Click a speaker chip to filter to that speaker. `n`/`p` jump to the next/previous turn.
- **Find**: filters and highlights matches inside the transcript.
- The transcript container is `data-pawn-anchor-root` with `v-target {kind: "session", ref: id}`, so selecting text allows Annotate or Ask about this. The origin carries the quote and the `t` of the first selected segment. Annotations re-anchor after relabels because the speaker names sit outside the anchored text nodes (render names in `::before`/`aria-label`, or in a sibling column excluded from the anchor root). After a retranscribe they may orphan, which the anchoring module already handles.

**Summary** section: the analysis fields, and the rendered `Pawn/Analyses/` note via `MarkdownView` when it exists.

**Items** section: `ItemCard` list from `/v1/items?source_ref=…`, with full triage verbs and the same store actions as the Inbox.

**Speakers rail** (the old Train modal):
- Per label: color, talk time and share bar, and a ▶ button that plays the three samples in turn.
- A name picker (gallery autocomplete with "Create 'X'"), with the reidentify suggestion pre-filled and marked "suggested 0.82".
- Primary action **Enroll voiceprint**. The confirmation is inline: "Enroll 15 m of ?1 as Bob?" with ▶. There is no extra modal, because you can now hear it.
- Secondary: **This session only** (relabel), **Ignore**/**Unignore**, and a link to the speaker in the Speakers tab.
- Footer: **Reidentify** and **Mark trained** (or "Trained ✓ · undo").
- After an enroll, a banner offers "Reidentify 7 other sessions with unknown speakers" (bulk endpoint §4.5).
- The rail collapses under the transcript below ~900 px.

**Extract / Reprocess dialogs**: the same fields as the plugin modals (`ExtractModal.ts`, `ReprocessModal.ts`), as compact `q-dialog`s. After Reprocess the header shows "Processing…" (from `processing`). The tab reloads when the transcript note changes (vault event) or the detail stops reporting `processing`.

---

## 6. Phase 3: Planner backend (`pawn_server/core/planner.py`)

1. **`update_event(cfg, event_id, **fields)`** plus `PATCH /v1/planner/events/{id}`. Partial update of `title, start, end, attendees, organizer, location, meeting_url, topics, focus, focus_before, session_id` and optionally `body`. It keeps `## Prep`, snippets and unknown frontmatter, renames the note when the day changes (reuse the upsert rename path) and recomputes the fingerprint only for manual events. It returns the event. Server-side focus replaces the plugin's frontmatter edit.
2. **Range view** `GET /v1/planner/range?from=YYYY-MM-DD&to=YYYY-MM-DD` (max 14 days) returns `{days: [PlannerDay-lite…]}`. A new `build_range` loads event notes **once**. Today `build_day` lists and reads every event note on each call (`_load_events`), so a week view would do that seven times. Also add a small in-process cache of parsed event notes keyed by S3 ETag, invalidated by vault events. That speeds up `build_day` too.
3. **Session spans**: return `start`/`end` per session block. Spike: check whether `created_at` is the recording start or the upload/finish time (chunked recorders), and derive `start = created_at` or `created_at - duration` accordingly. The plugin assumed start with no end.
4. **Daily plan**: `PlannerDay.plan` is the `## Plan` section of `Pawn/Daily/{date}.md` (written by `day_plan_update`), plus its path.
5. **Prep**: unchanged (`POST …/prep` → job). Add `prep_job_status` on the event when `prep_job_id` is still running.
6. Tests in `tests/test_planner.py`: PATCH merge/rename/prep preservation, range building, ETag cache invalidation, session spans, plan extraction.

---

## 7. Phase 4: Day UI

### 7.1 `DayPanel.vue` (tool tab `day:main`)
```
┌ ‹ Today › Thu 27 Nov ▾   [Day|Week]   [✦ Plan my day]  [+ Event]   ⋯ ┐
├────────────────────────────────────────────┬─────────────────────────┤
│ all-day: [Offsite ▸]                        │ AGENDA                  │
│ 08 ─────────────────────────────────────── │ ▸ Plan (Pawn/Daily)     │
│ 09 ┌ Product weekly  09:00–09:45 ★ prep ┐   │   - ship X; prep Y      │
│    │ Ada, Bob · Teams ▸Join             │   │ ▸ Untimed (2)           │
│ 10 └────────────────────────────────────┘   │ ▸ Open TODOs (5)  ✓ ✗   │
│    ┊ 🎙 Session 10:02–10:44 (striped)       │ ▸ Unlinked sessions (1) │
│ 11 ───────────── now ────────────────────── │   "Call w/ Bob" Link? ✓ │
└────────────────────────────────────────────┴─────────────────────────┘
```
- **Grid** (`components/TimeGrid.vue`, shared by Day and Week): about 48 px per hour, with a compact/comfortable density toggle. 15-minute snap. It scrolls to now (today) or the first event, and defaults to the 07:00–20:00 window. It shows the now line, focus bands and overlap columns (`lib/timegrid.ts`).
- **Events**: title, time, attendees and badges (source, prep, shot). A **Join** link appears when `meeting_url` is set. ★ toggles focus. A prep spinner shows while its job runs.
- **Sessions**: striped blocks with true duration. A session linked to an event shows as a 🎙 badge on the event instead of a duplicate block. Clicking it opens the session tab, to the side by default, so the Day view stays visible.
- **Agenda rail** (collapsible, remembered per viewer): the Plan (rendered, opens the Daily note), untimed/all-day items, open TODOs (`ItemCard`, triage inline) and unlinked sessions with suggestions (one-click Link).

### 7.2 Interactions
- **Create**: drag across empty time, or click for 30 minutes, to get a quick-create popover (title, Enter saves). "More…" opens the editor drawer.
- **Move**: drag a card with 15-minute snap. In week view it can also move across days.
- **Resize**: drag the bottom edge.
- `Esc` cancels a drag. Updates are optimistic with `PATCH`; on error they roll back and show a toast. Sessions are not draggable.
- **Click** opens the editor drawer, **double-click** opens the event note, and **right-click** shows event target actions (Ask about this, Annotate, Prep, Focus, Delete, Open note).
- Keys: `←`/`→` move by day/week, `t` today, `d`/`w` switch view, `n` new event, `Del` deletes the selected event (with confirm).

### 7.3 Event editor drawer (replaces the agenda rail while open)
- Fields: title; date, start and end (time inputs); an all-day toggle; attendees (chips); organizer; location; meeting URL with Join.
- Focus ★ with prep minutes.
- **Linked session**: a picker of same-day sessions, suggestions first, plus Open session and Unlink.
- **Prep**: rendered markdown, with **Generate**/**Regenerate** that follows the job live.
- Notes: rendered body plus "Edit note", which opens the FilePanel editor to the side.
- A Source link (Outlook), and **Delete**.
- Saving is debounced per field (autosave) with a "Saved" tick. No Save button.

### 7.4 Week view
Seven columns with the same `TimeGrid`, an all-day row, compact cards (title and time) and session blocks. Click a day header to open that day. Data comes from `/v1/planner/range`.

### 7.5 Plan my day
The **✦ Plan my day** button opens or reuses a side chat with `origin {kind: "day", ref: date}`, split to the right of the Day tab. The chat runs the `day_plan` skill. Events it writes via `event_upsert` appear on the grid within about a second, through the vault-event refresh. The Plan section in the agenda updates the same way. Reopening on the same date focuses the existing chat.

---

## 8. Phase 5: Speakers gallery

### 8.1 Backend (`pawn_server/core/speakers_api.py`, new; logic stays in `pawn_diarize/core/speaker_gallery.py`)

| Endpoint | Purpose |
|---|---|
| `GET /v1/speakers?q=&include_inactive=` | List: name, aliases, active, enrollment count and total seconds, session count, last-seen session/date, People note path |
| `GET /v1/speakers/{id}` | Detail: enrollments (`id, source_session_id, start_time, end_time, duration, quality_score, approved_at`) and recent sessions |
| `PATCH /v1/speakers/{id}` | `display_name`, `aliases`, `notes`. Reports how many sessions reference the old name, and optionally queues a bulk reidentify |
| `DELETE /v1/speakers/{id}` | Deactivate (soft), the same as `pawn-diarize speakers deactivate` |
| `DELETE /v1/speakers/{id}/enrollments/{eid}` | `remove_enrollment` |
| `POST /v1/speakers/{id}/merge` `{into}` | **New** `SpeakerGallery.merge_speakers(src, dst)`: move enrollments, repoint `session_speaker_map`, union aliases (plus the source name), deactivate the source, return the affected sessions; optional reidentify |

Enrollment playback reuses the session audio endpoint, because enrollments carry `source_session_id` and start/end. Tests go in `tests/test_speakers_api.py`, with merge covered at the gallery level.

### 8.2 `SpeakersPanel.vue` (tool tab `speakers:main`, `?id=` selects a speaker)
- **List** (left): search, an Active/All filter, sorting by name, last seen or sessions, and a row with avatar, name, aliases, enrollment count and last seen.
- **Detail** (right):
  - Inline rename and alias chips.
  - People note (`People/{id}.md`): open it, or create it via `people_ensure` behind a small endpoint.
  - Enrollments with ▶ and remove.
  - Sessions list (opens the session tab).
  - **Merge into…** (picker plus a confirm that lists the effects) and **Deactivate**.
- Entry points: the header overflow menu, `g p`, a speaker name in the session speakers rail, and the Sessions facet chip context menu.

---

## 9. Phase 6: integration and polish

- **Inbox**: items whose `source_kind = session` get **Open session**, which opens the session tab's Items section.
- **FilePanel**: a `pawn: transcript` note shows a banner "Open as session", and a `pawn: event` note shows "Open in Day".
- **HomePanel**: a "Today" card with the next three events, sessions needing training and a Plan my day shortcut.
- **Search**: the sidebar search also matches sessions (title and summary), opening the session tab.
- **Docs**: `docs/PORTAL.md` (new Sessions, Day and Speakers sections), `docs/PLANNER.md` (portal Day view; the plugin Day tab becomes legacy), `docs/SPEAKERS.md` (gallery UI, merge), and the `AGENTS.md` portal bullet plus test pointers.
- **Plugin**: no changes (frozen). Once this lands, the Sessions and Day tabs stop being reasons to keep it.

---

## 10. Sequencing and parallelism

```
Phase 0 (foundations) ──┬─► Phase 1 → Phase 2   (Sessions)
                        ├─► Phase 3 → Phase 4   (Day)
                        └─► Phase 5             (Speakers; needs §4.4 audio for ▶ only)
                                   └─► Phase 6 (integration, docs)
```
- Phase 0 is small but touches shared files (`tabs.ts`, `router.ts`, `Workspace.vue`, `App.vue`, `api/*`, `targets.ts`, `portal_api.py`). Land it alone first.
- After Phase 0, the Sessions, Day and Speakers tracks can run in parallel sessions. The overlaps are `api/types.ts`, `api/client.ts` and `api/mock.ts`, so each track should add its own section at the end of those files to keep merges trivial.
- Rough size: P0 S–M, P1 M (audio offsets are the risk), P2 L, P3 M, P4 L (drag interactions), P5 M, P6 S.

## 11. Testing

- **Backend**: the new and extended pytest files above, plus the existing `tests/test_portal_api.py` for `_origin_context` kinds.
- **Frontend (vitest)**: `lib/timegrid.test.ts` (overlap layout, snapping, drag math, cross-day moves), `lib/audio.test.ts` (global↔file time mapping, boundary switching), transcript turn rendering and anchoring after relabel, and sessions store paging and facets.
- **Build**: `make portal` and `make portal-test`.
- **Manual**: in `VITE_PAWN_MOCK=1 make portal-dev`, plus a real server pass covering training with audio, reprocess round-trip, drag/resize persistence, Plan my day live updates and merging speakers.

## 12. Open spikes (resolve at the start of the phase)

1. **Audio file offsets** for chunked sessions (§4.4): persist `file_offsets` from now on, and probe legacy files.
2. **`created_at` semantics** for session start/end on the grid (§6.3).
3. **`processing` signal** for reprocess: reuse the diarize queue claim, or write an explicit marker row.
4. **Audio codecs** in stored recordings: list the extensions present in `transcription_segments.audio_file` before deciding on the 415 fallback.

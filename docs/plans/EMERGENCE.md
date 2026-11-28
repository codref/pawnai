# Plan: Emergence layer

Status: proposal, 2026-11-27. Scope: replace "one prompt → item cards" with a
layer that keeps a living model of the user's work world (threads, questions,
decisions, commitments, people) and surfaces only what changes it. The layer
learns from the user's own triage. It builds on the coworker loop
([COWORKER.md](../COWORKER.md)) and keeps Items as one view over the new data.

Decisions already made:

- All ten capabilities below are in scope. They are built on one shared spine
  (observations → threads/questions), so later ones are cheap once the spine exists.
- Open questions are first-class objects and drive extraction (capability 8).
- Extraction must improve from use: every user action is a label.
- Captures are machine context. They are never shown as cards; they only count
  as evidence.
- Latency does not matter; local models are slow but run 24/7. Prefer many
  narrow passes over one clever pass.

Input volume to design for: about 10 hours of recorded meetings a week (roughly
120k transcript tokens), plus captures, Thunderbird mail captures, phone events
and vault notes.

---

## 1. What is wrong today

`process_source` in `pawn_agent/core/coworker/pipeline.py` calls
`extract_items` (`pawn_agent/core/coworker/extract.py`) once per source, then
`score_items` once over all items.

| Problem | Where | Effect |
|---|---|---|
| Source cut at 24,000 characters | `extract.py`: `source_text[:24000]` | Speech is ~50k chars/hour. The second half of a one-hour meeting is never read, which is where decisions usually land. Fails silently. |
| One prompt, five item kinds | `_PROMPT` | Small models are weak at "find everything important". Decisions and commitments need different fields. |
| No knowledge of the user's world | Extraction sees only the source | Cannot tell new information from repetition, or connect to an open question. Linking happens afterwards (`related_lines`) by embedding distance only. |
| Scoring only against `Goals.md` | `score.py` | Anything outside 3–7 active threads is never noticed, so nothing can *emerge*. |
| Feedback thrown away | `todo/file/delete/ignore` only set status and suppress fingerprints | Same mistakes forever. Prompts are hand-tuned. |
| Item is the final unit | `coworker_items` | Items pile up. Nothing accumulates into a storyline, belief or retrospective. |
| No timestamps on items | Item has `quote` but no position | Hard to jump back to the moment; no recall labels possible. |

## 2. Concepts

The pipeline turns sources into **observations**, routes them onto long-lived
**subjects**, and turns changes in subjects into rare **provocations**.

```
Sources ──▶ Windows ──▶ Observations ──▶ Subjects ─────────────▶ Provocations
(transcript, (3k tok,    (atomic claim,    (Thread, Question,       (1–3 / day,
 mail, capture, overlap)  typed, quoted,    Decision, Commitment,    each one asks
 phone, note)             timestamped)      Prediction, Person,      something)
                                            Hypothesis)
                    ▲                                    │
                    └────────── Labels (user actions) ◀──┘
```

| Concept | What it is | Lifetime |
|---|---|---|
| **Source** | One transcript session, mail capture, web capture, phone event batch or vault note | Immutable once processed (re-processing replaces its observations) |
| **Window** | A slice of a source (~3k tokens, ~15% overlap, cut on speaker turns) | Derived, cached by content hash |
| **Observation** | One atomic claim with type, speaker, quote, time offset and confidence | Append-only |
| **Thread** | An evolving storyline across sources ("Platform migration confidence eroding"). `origin: goal` (from `Goals.md`) or `origin: emergent` (proposed by Pawn) | Weeks to months |
| **Question** | An open question with evidence for/against attached over time | Until answered or dropped |
| **Decision** | A choice made, with rationale, **assumptions**, expected outcome and a review date | Until reviewed |
| **Commitment** | Who owes whom what, by when, in which direction (`mine` / `theirs`) | Until closed or stale |
| **Prediction** | A forecast someone made, with a resolution date | Until resolved |
| **Person state** | Recurring concerns, open asks, tone trend, last 1:1 themes | Rolling; lives in `People/{id}.md` ([PEOPLE.md](../PEOPLE.md)) |
| **Hypothesis** | A machine-proposed connection between distant observations | Until promoted or killed |
| **Provocation** | A surfaced change, phrased as a question to the user | One day |
| **Label** | The user's verdict on any of the above | Forever: training data |

Items stay. An Item becomes a *projection* of an actionable observation
(decision / commitment / block) so the existing Inbox, triage verbs, Matrix
replies and `Pawn/Items/*.md` notes keep working.

## 3. Data model

New tables in PostgreSQL (pgvector already present; `knowledge_chunks` uses
`Vector(1024)` from the sallm embedding model). One migration per phase.
Models live in `pawn_agent/utils/db.py` next to the coworker ones.

```text
em_sources        id, kind (session|mail|capture|phone|note), ref, occurred_at,
                  content_hash, processed_at, pipeline_version, status

em_windows        id, source_id, ordinal, start_sec, end_sec, char_start, char_end,
                  content_hash, triage (json: {decision: .9, commitment: .1, ...}),
                  triage_model, embedding vector(1024)

em_observations   id, source_id, window_id, type, text, speaker_id, speaker_name,
                  quote, quote_score (0..1 fuzzy match), start_sec,
                  fields (json, type-specific), confidence, extractor (prompt id+version),
                  model, embedding vector(1024), status (live|rejected|superseded),
                  created_at

em_subjects       id, kind (thread|question|decision|commitment|prediction|hypothesis),
                  title, slug, origin (goal|emergent|user|extracted), status,
                  state_md (current summary, rewritten), fields (json),
                  momentum, last_evidence_at, review_at, note_key,
                  centroid vector(1024), created_at, updated_at

em_links          observation_id, subject_id, role (evidence_for|evidence_against|
                  mentions|updates|resolves|contradicts), score, linked_by (model|user)

em_subject_events subject_id, at, kind (created|state_rewritten|status_changed|
                  merged|split|reviewed), diff (json), cause_observation_ids
                  -- the time machine: "what did I believe in March"

em_labels         id, target_kind (observation|link|subject|provocation|window),
                  target_id, verdict (accept|edit|reject|missed|useful|wrong|not_now),
                  before (json), after (json), surface (portal|matrix|note|chat),
                  created_at

em_provocations   id, day, subject_id, kind, text, score, features (json),
                  shown_at, verdict
```

Type-specific `fields` on observations:

| Type | Fields |
|---|---|
| `decision` | `choice`, `rationale`, `assumptions[]`, `owner`, `alternatives[]` |
| `commitment` | `debtor`, `creditor`, `what`, `due`, `direction` (mine/theirs/other) |
| `prediction` | `claim`, `by_when`, `who`, `confidence_words` ("pretty sure") |
| `question` | `question`, `asked_by`, `blocking` |
| `concern` | `who`, `about`, `intensity` |
| `fact` | `about`, `value` (status updates, numbers, dates) |
| `contradiction` | `claim_a`, `claim_b`, `observation_ids` |

`speaker_id` is the curated Speakers gallery id when the segment is mapped
([SPEAKERS.md](../SPEAKERS.md)); otherwise only `speaker_name`.

Why separate tables instead of more columns on `coworker_items`: observations
are 10–50× more numerous than items, links are many-to-many, and subject
history must be append-only.

## 4. Per-source pipeline

Runs on `session_completed` (replacing the `extract_items` call in
`process_source`), on capture/mail arrival, and on the vault scanner. Each
pass is a job on a new low-priority local queue so a slow model never blocks
the API. Every pass is idempotent by `(window.content_hash, prompt_version,
model)`, so re-running is cheap and prompt changes re-run only what changed.

### Pass 0: windowing (no model)

- Transcript: build lines `[mm:ss] Speaker: text` from segments, keeping
  `start_time`. Cut at ~3k tokens on a speaker-turn boundary; overlap the last
  ~15%. User `## Annotations` are prepended to *every* window as context, not
  as a window.
- Mail / capture / note: reuse `chunk_markdown` with larger windows.
- Store windows and their embeddings. The existing transcript `index_text`
  call can reuse these embeddings instead of re-chunking.

### Pass 1: triage (fast model, tiny output)

One call per window: "Which of these occur: decision, commitment, prediction,
question, concern, fact, contradiction, none? Reply with a score 0–1 each."
Output is ~60 tokens. Windows that score below a threshold on everything stop
here. Expectation: 40–60% of meeting windows are "none".

Later (§7) a trained classifier on window embeddings replaces or pre-filters
this pass.

### Pass 2: typed extraction (fast model, one prompt per type)

For each (window, type) whose triage score clears the threshold, run that
type's prompt with that type's schema. Each prompt is short and specific, and
includes 3 accepted + 2 rejected past examples retrieved by embedding
similarity (§7, level 2). Prompts are versioned files under
`pawn_agent/core/emergence/prompts/{type}.v{n}.txt` so labels can be tied to
a prompt version.

### Pass 3: directed extraction (capability 8)

One call per window with the **live subject digest**: open questions, active
threads (goal and emergent), decisions under review with their assumptions,
open commitments involving the window's speakers. Cap the digest at ~2k tokens,
choosing the subjects nearest the window embedding.

Prompt: "For each listed subject, does this window contain evidence for,
against, an update, or a resolution? Quote it." This finds things the generic
pass misses because the model is looking for something specific.

### Pass 4: verification (no model)

- Fuzzy-match every `quote` against the window text (stdlib `difflib`
  `SequenceMatcher` over sliding spans; `rapidfuzz` is not a dependency and
  is not needed at this volume). Below 0.8 → `status: rejected`, reason `quote_not_found`.
- Map the matched span back to `start_sec` using the window's line timestamps.
- Dedupe across overlapping windows: same type + embedding cosine distance
  < 0.1 + overlapping time → keep the higher-confidence one.

### Pass 5: routing (embedding + small model on ambiguity)

For each surviving observation:

1. Find the k nearest subjects by `centroid` distance.
2. Clear winner (distance under threshold, margin over the second) → link
   directly. Ambiguous → one small call: "which of these subjects, or none?"
3. Directed-pass results already name their subject; link with the role
   given.
4. No match → observation stays *unrouted*. The nightly clusterer (§5) decides
   whether unrouted observations form a new emergent thread.
5. Decisions, commitments and predictions also create their own subject
   (kind decision/commitment/prediction) and link to their thread.

### Pass 6: projections (no model)

- Actionable observations (decision with no owner, commitment, block,
  contradiction on an active goal thread) → `coworker_items` through the
  current `insert_item` / `render_item_note` path, so the Inbox, interrupts and
  `Pawn/Today.md` behave as today.
- Item rows gain `observation_id` and `start_sec` in `payload`, so triage
  verbs write labels (§7) and the Portal can jump to the moment in the
  session audio.
- Scoring against `Goals.md` (`score_items`) becomes a feature in surfacing
  (§6) instead of the only gate.

## 5. Night shift

Runs as durable schedules ([AGENT_SCHEDULER.md](../AGENT_SCHEDULER.md)) in a
quiet window, by default 01:00–06:00 local, on the slow/large model. Each job
has a token budget per night and resumes where it stopped. Jobs in priority
order:

| Job | Capability | What it does | Output |
|---|---|---|---|
| **Thread synthesis** | spine | For each subject with new evidence since the last rewrite: rewrite `state_md` from the previous state plus new observations; compute momentum (evidence rate this week vs. the trailing 4 weeks); log a `state_rewritten` event with a diff | `Pawn/Threads/{slug}.md` |
| **Emergent clustering** | spine | HDBSCAN (`sklearn.cluster.HDBSCAN`; scikit-learn is installed only transitively today, so declare it, or fall back to a numpy agglomerative pass) over unrouted observations from the last 30 days. A cluster with ≥ 4 observations from ≥ 2 sources and ≥ 2 days → propose an emergent thread with a title and state | Thread with `status: proposed` |
| **Assumption watch** | 1 | For each decision under review, check new observations against each assumption: holds / weakened / broken, with quotes | Provocation when weakened or broken |
| **Commitment aging** | 4 | Close commitments with resolving evidence; flag ones past due or untouched for `commitment_days`; split mine vs. theirs | Morning brief and 1:1 prep |
| **Prediction resolution** | 2 | For predictions past `by_when`, search evidence and propose true / false / unclear | Calibration table (user confirms) |
| **Contradiction scan** | 6 | Within each thread, compare new observations with the current state and with each other; across threads, flag the same entity described differently | Provocation |
| **Person refresh** | 5 | Per person with new observations: recurring concerns, open asks both ways, tone trend, themes of the last three 1:1s | `People/{id}.md` sections, via the `people_refresh` autonomy gate |
| **Priority mirror** | 3 | Weekly: meeting minutes and observation counts per thread vs. `Goals.md` order | Weekly review section |
| **Dreaming** | 7 | Sample pairs of distant observations/subjects (different threads, ≥ 2 weeks apart, moderate embedding distance), ask "is there a non-obvious connection?", then a second strict critic pass. Survivors become hypotheses | `Pawn/Hypotheses/*.md` |
| **Council** | 9 | On demand, or for a decision tagged `council`: 4–5 personas (boss, skeptical senior engineer, finance, affected report, you in six months) plus a pre-mortem | Note linked from the decision |
| **Retrospective** | 10 | Weekly / monthly / quarterly from `em_subject_events`: threads born and closed, beliefs revised, decisions that aged well or badly, calibration, "what you worried about 90 days ago" | `Pawn/Retros/{period}.md` |

### Compute budget (rough)

Assumptions: 120k transcript tokens a week plus ~50k from other sources;
3k-token windows with 15% overlap → ~65 windows a week; fast model ~300 tok/s
prefill, ~20 tok/s output; slow model ~100 / 8.

| Pass | Calls/week | Prefill | Output | Time |
|---|---|---|---|---|
| Triage | 65 | ~230k | ~4k | ~15 min |
| Typed extraction | ~90 | ~350k | ~45k | ~60 min |
| Directed extraction | 65 | ~330k | ~25k | ~40 min |
| Routing (ambiguous only) | ~100 | ~100k | ~5k | ~10 min |
| Night shift (all jobs) | — | ~2–3M | ~150k | ~6–8 h on the slow model |

Total: a few hours a week of fast-model time and roughly one night's worth of
slow-model time, spread over seven nights. There is room for 3–5× more passes;
the first use of spare capacity should be **self-consistency** (run typed
extraction twice at different temperatures and keep agreements) before adding
new features.

## 6. Surfacing

The user sees little. Everything still lands in the vault and is searchable;
only provocations interrupt.

A **provocation** is a change in a subject, phrased as a question:

- "You decided X assuming Y. Two meetings this week say Y no longer holds. Revisit?"
- "Luca has raised on-call load in 3 of the last 4 1:1s. Is that on your radar?"
- "An emergent thread formed: *vendor contract doubts* (6 observations, 3 people). Track it?"
- "You predicted the API freeze by 2026-11-15. It slipped. Your hit rate on dates is 40%."

Scoring is a learned ranker (§7, level 1) over features:

| Feature | Meaning |
|---|---|
| `goal_thread` | Linked to an active `Goals.md` thread |
| `novelty` | Distance from what was already surfaced on this subject |
| `momentum` | Change in evidence rate |
| `tension` | Contradiction, broken assumption, overdue commitment |
| `people_weight` | Involves people the user labels as important (learned) |
| `recurrence` | How many independent sources back it |
| `age` | Time since the subject last surfaced |
| `type` | Provocation kind |

Budget: `emergence.brief.max_per_day` (default 3). The morning briefing
(`{coworker.daily_dir}/{date}.md`, default `Pawn/Daily/`) gains a
**Provocations** section at the top. Each provocation has Useful / Wrong /
Not now, the same verbs in Matrix replies (`useful p3`, `wrong p3`). Interrupts
outside the brief stay governed by `Goals.md` `interrupt:` rules and the
existing policy (`policy.py`).

Meeting prep (`meeting_prep.py`, Day tab) gains a **Brief** block per event:
open commitments both ways with the attendees, their recent concerns, threads
they are on, decisions with assumptions they own.

## 7. Learning loop

### Where labels come from

All labels go to `em_labels`, written by the existing surfaces.

| Surface | Label |
|---|---|
| Item triage `todo` / `file` / `task` | `accept` on the observation |
| Item `delete` / `ignore` | `reject` on the observation |
| Editing an item note's text or context | `edit` (before/after: a correction example) |
| Portal transcript annotation or highlight with "Pawn missed this" | `missed` on the window (recall label; the most valuable kind) |
| Provocation Useful / Wrong / Not now | on the provocation |
| Accept / dismiss an emergent thread or hypothesis | on the subject |
| Unlink / relink an observation in a thread note or the Portal | on the link |
| Confirming a prediction's resolution | ground truth for calibration |

### Levels, cheapest first

1. **Relevance ranker (from ~100 labels).** Logistic regression on
   `[observation embedding, type one-hot, provocation features]`, trained
   nightly in seconds with numpy (no new dependency). It ranks provocations
   and filters which observations become Items. Expect the biggest perceived
   gain from this alone.
2. **Retrieved few-shot (from ~50 labels).** Typed extraction prompts include
   the most similar accepted and rejected examples for that type, and edits as
   "wrong → right" pairs.
3. **Triage classifier (from ~300 windows with outcomes).** Logistic
   regression on window embeddings predicting "this window yields an accepted
   observation of type T". It pre-filters pass 1 and later replaces it.
4. **Nightly prompt search (from ~300 labels).** Keep a frozen eval set
   (20% of labeled windows). The night shift tries prompt variants (reworded
   rules, different example counts, field orders) and promotes a variant only
   if it beats the current one on the eval set. DSPy-style, but a small local
   script so it stays debuggable.
5. **Distillation + LoRA (later, optional).** The slow model re-extracts all
   windows at night; its outputs, filtered by the user's labels, fine-tune the
   fast model monthly (unsloth / PEFT, outside the server process). Only worth
   it when levels 2–4 stop improving.

### Metrics

Shown in the Settings tab and logged as `agent_runs`:

- **Precision**: accepted / (accepted + rejected) per type, per prompt version.
- **Recall proxy**: `missed` labels per meeting hour (should fall).
- **Provocation usefulness**: useful / shown, 7-day rolling.
- **Cost**: model seconds per meeting hour.

A prompt or model change that lowers precision on the eval set is never
promoted automatically.

## 8. Vault layout

All under `Pawn/` (free-write root). Notes are rendered from the DB; manual
edits to managed sections are read back as labels, other sections preserved
(same pattern as transcript `## Annotations`).

```text
Pawn/Threads/{slug}.md        pawn: thread   (exists today as the `file` target; becomes managed)
Pawn/Questions/{slug}.md      pawn: question
Pawn/Decisions/{date}-{slug}.md  pawn: decision
Pawn/Commitments.md           one table: mine / theirs / overdue
Pawn/Predictions.md           open + resolved, with hit rate
Pawn/Hypotheses/{slug}.md     pawn: hypothesis
Pawn/Retros/{period}.md       weekly / monthly / quarterly
```

Thread note shape:

```markdown
---
pawn: thread
id: …
origin: emergent
status: active
momentum: rising
people: [Luca, Mira]
---

# Platform migration confidence eroding

## State
(rewritten nightly; 5–10 lines)

## Open questions
- [[Pawn/Questions/is-q2-realistic]] — 4 for / 2 against

## Decisions
- [[Pawn/Decisions/2026-11-03-freeze-api]] — assumption "team B ready" weakened

## Timeline
- 2026-11-20 · Marco: "Q2 at the earliest" · [[transcript#14:32]]
- …

## Notes
(user-owned, never touched)
```

A user can create a question by writing `Pawn/Questions/*.md` with
`pawn: question` or with `/ask-later <question>` in chat. Pawn creates
`status: proposed` questions; they are shown in the brief at most once.

## 9. Portal

- **Threads** tool tab: momentum-sorted list; thread view with state, timeline
  (jump to `start_sec` in the session audio player), linked questions,
  decisions and people; accept/dismiss for proposed threads.
- **Ledger** tool tab: decisions with assumption health, commitments
  (mine/theirs), predictions with calibration.
- **Session tab**: observations as margin marks on the transcript; highlight
  text → "Pawn missed this" (writes a `missed` label and queues a re-extract
  of that window).
- **Settings**: emergence metrics (§7) and the night-shift budget.
- Later: a **Map** view (threads × time, sized by evidence).
- **Brainmap** (2026-11-28): a corkboard visualization type, first board `items`
  ([PORTAL.md](../PORTAL.md#brainmap)). It is the first Portal surface for
  unlink/relink (§7): tie, cut, ghost-string verdicts, lasso into threads,
  accept/burn proposed subjects. Backend: `pawn_server/core/brainmap.py`
  (graph builder, mutations), `brainmap_api.py` (`/v1/portal/brainmaps/*`),
  migration `0033` (`portal_boards` layout). Labels: `target_kind` `link`
  (key = subject id + normalized observation text, so a cut survives
  re-processing; `route`, the ambiguous pass and the directed pass skip
  rejected pairs) and `subject` (accept/dismiss from any surface). Both carry
  no embedding and are left out of `emergence export`, so they never mix
  with extraction verdicts.

## 10. Config

```yaml
emergence:
  enabled: false
  fast_model: ""          # catalog id; empty → agent.default
  slow_model: ""          # catalog id for the night shift; empty → fast_model
  window_tokens: 3000
  window_overlap: 0.15
  triage_threshold: 0.35
  quote_min_score: 0.8
  night:
    window: "01:00-06:00"
    max_tokens_per_night: 2000000
    jobs: [synthesis, clustering, assumptions, commitments, predictions,
           contradictions, people, dreaming]
  brief:
    max_per_day: 3
  learning:
    ranker: true
    few_shot: true
    prompt_search: false
```

`llm_sub.run` gains an optional `model` argument so passes can pick the fast
or slow catalog entry without mutating `cfg`.

## 11. Phases

Each phase is shippable on its own and has an exit test.

| Phase | Build | Exit test |
|---|---|---|
| **0. Fix truncation** | Windowed extraction inside the current `extract_items` (loop over windows, merge, dedupe); keep one prompt. | A 90-minute meeting yields items with timestamps from its last 30 minutes. |
| **1. Observation spine** | `em_sources`, `em_windows`, `em_observations`; passes 0, 1, 2, 4; items projected from observations; `start_sec` on items. | Item precision on a hand-labeled set of 5 meetings ≥ today's; recall clearly higher. |
| **2. Labels + ranker** | `em_labels` written by every triage surface; numpy ranker; few-shot retrieval. | Labels flow from Portal, Matrix and notes; ranker AUC > 0.7 after 2 weeks of use. |
| **3. Subjects** | `em_subjects`, `em_links`, `em_subject_events`; questions and threads (goal + emergent); passes 3 and 5; thread synthesis and clustering at night; thread/question notes. | At least one emergent thread the user accepts in the first two weeks. |
| **4. Ledger** | Decisions with assumptions, commitments, predictions; assumption watch, commitment aging, prediction resolution; meeting-prep Brief block. | A 1:1 prep shows correct open commitments both ways. |
| **5. Provocations** | `em_provocations`, morning-brief section, verdict verbs. | ≥ 50% useful over 2 weeks. |
| **6. Reflection** | Retrospectives, priority mirror, contradiction scan, person refresh extensions. | First monthly retro the user would keep. |
| **7. Imagination** | Dreaming + hypotheses, council, prompt search, optional LoRA. | Hypotheses accepted at a rate the user finds worth the noise. |

Phase 0 landed on 2026-11-28: `window_source`, `locate_quote` and
`merge_items` in `pawn_agent/core/coworker/extract.py`, `start_sec` /
`window` / `quote_score` in the item `payload`, covered by
`tests/test_coworker_extract_windows.py`. Low `quote_score` is recorded but
does not drop items yet; that is pass 4 in Phase 1. The exit test still needs
one real 90-minute meeting run through `pawn-server coworker process`.

Phase 1 landed on 2026-11-28 behind `emergence.enabled` (default off):
migration `0029`, package `pawn_agent/core/emergence/` (`prompts.py`
versioned per type, `passes.py` triage + typed extraction, `pipeline.py`
`observe_source` / `project_items`, `store.py`, `compare.py`), routed from
`coworker.pipeline._extract`, tests in `tests/test_emergence_observations.py`.
Differences from §4: speaker ids are not resolved yet (`speaker_name` only);
embedding columns exist but stay empty; windows are re-run on every pass
(no content-hash cache yet). The exit test runs through
`pawn-server coworker compare --session ID` on five meetings: tick useful
items in each column and compare precision and "found only here".

Phase 1 tuning from four runs on one 61-minute meeting (2026-11-28): thinking
models spent 2-5k tokens per call, so `think_types` limits reasoning to
decision/commitment/block; triage scored clear commitments at 0.2-0.4, so
`always_types` extracts those three kinds from every window and triage only
gates the rest; one statement filed as several kinds is merged into one Item
(`merge_statements`); prompts are decision v2 and question v3; replies that
are not valid JSON are parsed leniently and retried once.

Phase 2 landed on 2026-11-28 (migration `0031`, `labels.py`, `ranker.py`).
Differences from §7: the Inbox history was not backfilled, because of ~830
extracted items only 5 were ever accepted (the rest were mostly cleared in
bulk), so labels start from compare review notes and single-item triage;
`missed` counts as a positive; bulk deletes are stored but not trained on;
nightly prompt search (level 4) came with Phase 7. There is no Portal "Pawn
missed this" button yet; the API endpoint exists for it.

Phases 3-7 landed on 2026-11-28 (migration `0032`; `subjects.py`, `route.py`,
`threads.py`, `ledger.py`, `provocations.py`, `reflect.py`, `dream.py`,
`search.py`, `night.py`, `vault_notes.py`; user guide `docs/EMERGENCE.md`).
Decisions taken while building:

- One table for all subject kinds (`em_subjects.kind`); ledger entries are
  subjects with `parent_id` pointing at their thread.
- Routing runs right after each source (embedding first, one batched model
  call for ambiguous cases); the directed pass stores what it finds as
  `evidence` observations so the evidence keeps its quote and time.
- Emergent clustering requires ≥ 2 sources but not ≥ 2 days, because
  observation timestamps are processing time, not meeting time.
- Notes are the UI: frontmatter `status` and `council: true` are read back;
  Pawn tracks the status it last wrote so a user edit and a DB change are
  told apart. No Portal tabs yet: the portal is being changed in parallel, so
  the backend is exposed as `/v1/emergence/*` for a later Threads / Ledger tab.
- People: digests go to `Pawn/People/`; gallery bios only get open loops
  under autonomy `people_refresh = allow`; tone trends are not built.
- Provocations are scored by fixed per-kind weights plus features until 20
  useful/wrong verdicts exist, then by logistic regression on them.
- Level 5 is an export (`emergence export`), not in-process fine-tuning.
- `prompt_search` is not in the default night jobs (it is expensive); run it
  by hand or add it to `night_jobs`.

## 12. Risks and open questions

- **Hallucinated observations** poison threads. Mitigation: quote verification
  is mandatory; state rewrites cite observation ids; a subject's state may only
  claim what linked observations support (the critic pass checks this nightly
  on a sample).
- **Thread sprawl.** Emergent clustering can propose too many threads.
  Mitigation: proposed threads count against the provocation budget; unaccepted
  ones expire after 21 days; nightly merge pass for near-duplicate centroids.
- **Speaker attribution errors** corrupt commitments and person state.
  Observations from unmapped speakers get lower confidence and never feed
  person state.
- **Privacy and tone.** Person state about reports is sensitive. It stays in
  the vault under `People/` with no notifications about individuals' mood; tone
  trend is opt-in (`emergence.people.tone: false` by default).
- **Model swaps** break comparability. Labels and observations record prompt
  version and model; metrics are per pair.
- **Goals.md authority.** Emergent threads never edit `Goals.md`; promoting one
  goes through the existing `goal_propose` → `/goal apply` path.
- Open: should captures and mail run the full pipeline or only directed
  extraction (pass 3)? Proposal: directed only, since they serve as evidence.
- Open: one queue for all passes or a separate night queue? Proposal: one
  local priority queue with two lanes (fast / slow).
- Open: keep `coworker_items` as a separate table long-term, or make Items a
  DB view over observations once Phase 3 settles?

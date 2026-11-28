# Using the emergence layer

The emergence layer keeps a model of your work world (threads, open questions, decisions with their assumptions, commitments both ways, predictions, people) and surfaces only what changes it: one to three questions a day in the morning brief. It builds on the coworker loop ([COWORKER.md](COWORKER.md)); Items keep working and become one view over the same data. Design and history: [plans/EMERGENCE.md](plans/EMERGENCE.md).

It is off until `emergence.enabled` is true.

## Turn it on

```bash
alembic upgrade head        # 0029 observations, 0031 labels, 0032 subjects
```

```yaml
coworker:
  enabled: true
  me: [Davide]              # whose commitments are "mine"
emergence:
  enabled: true
  night_cron: "30 1 * * *"  # the night shift, in coworker.timezone
```

`pawn-server serve` then runs everything: extraction after each meeting, the night shift on `night_cron`, provocations in the morning briefing (`coworker.briefing_cron`).

## What happens after a meeting

1. **Observations.** The transcript is read in overlapping windows; each window gets a cheap triage and one prompt per kind (decision, commitment, question, block, contradiction, prediction, concern, fact). Every observation must quote the transcript; quotes that are not found are rejected. See [COWORKER.md](COWORKER.md#observation-layer-emergence).
2. **Ledger.** Decisions, commitments and predictions become ledger subjects (a near-identical open one is updated instead).
3. **Routing.** Each observation is attached to the nearest thread or question by embedding; ambiguous ones are settled by one batched model call; the rest stay unrouted for the night.
4. **Directed pass.** Each window is asked about the open subjects closest to it: is there evidence for, against, an update, a resolution? A question answered or a commitment done here closes it.
5. **Items.** Actionable observations still become Inbox items, so triage works as before.

## The night shift

Runs once a night under a budget (`night_max_calls`, default 400; `night_max_minutes`, default 300). Jobs, in order:

| Job | What it does | Writes |
|---|---|---|
| `sync` | `Goals.md` threads become goal threads; your status edits in notes are applied; new notes under `Pawn/Questions/` become questions; unaccepted proposals expire after `proposed_expiry_days` (21) | — |
| `boards` | Question boards ([PORTAL.md](PORTAL.md#boards)): the nearest recent observations are judged and linked, the board's own evidence is dreamt on, a "While you slept" note goes into the board chat, and a `board_update` provocation is raised when the board moved | board chat |
| `clustering` | Unrouted observations from the last 30 days that cluster (≥ 4 observations from ≥ 2 sources) become a **proposed** emergent thread, with its open question | `Pawn/Threads/…` |
| `synthesis` | Every thread/question with new evidence gets its summary rewritten and its momentum (this week vs. the trailing three) updated; answered questions are marked resolved | thread and question notes |
| `assumptions` | Each open decision's assumptions are judged holds / weakened / broken against new evidence; review dates are flagged | `Pawn/Decisions/…` |
| `commitments` | Overdue commitments are flagged, yours and theirs | `Pawn/Commitments.md` |
| `predictions` | Predictions past their date get a proposed outcome for you to confirm | `Pawn/Predictions.md` |
| `contradictions` | New statements on a thread that contradict each other or its summary | — |
| `people` | Per-person digest: recurring concerns, what they owe, what they are owed | `Pawn/People/…` |
| `dreaming` | Pairs of distant observations (different sources, ≥ 7 days apart) are checked for a non-obvious link, then by a strict critic; survivors become proposed hypotheses | `Pawn/Hypotheses/…` |
| `council` | Decisions you tagged `council: true` get five critiques and a pre-mortem | `Pawn/Decisions/…-council.md` |
| `retros` | The last finished week, month and quarter get a retrospective with the priority mirror and calibration | `Pawn/Retros/…` |
| `prompt_search` | Off by default (add it to `night_jobs`): tries prompt variants against your labels | `Pawn/Reviews/prompt-search-…` |

Run it by hand, all or some jobs:

```bash
pawn-server emergence night
pawn-server emergence night --job sync --job synthesis
```

## The morning brief

The briefing writes `Pawn/Daily/{date}.md` with a **Provocations** section first: at most `brief_max_per_day` (3) questions, each with a short id.

```markdown
## Provocations

- You decided **Freeze the API on Monday** assuming “Team B is ready by Monday”. That assumption looks broken: Mira: team B is not ready. Revisit? `p3a1`
- A storyline is forming: **Vendor contract renewal risk** (6 observations across 3 sources, Luca, Ansgar). Track it? `p9c0`
```

Reply in chat or Matrix (same rules as triage verbs):

| Reply | Effect |
|---|---|
| `useful p3a1` | Kept as useful. A proposed thread or hypothesis becomes tracked; a proposed prediction outcome is confirmed. |
| `wrong p3a1` | Kept as wrong. A proposed thread is dropped; a prediction outcome is flipped; an "answered" question reopens; an "overdue" commitment closes. |
| `not now p3a1` | Comes back in three days. |

After 20 useful/wrong verdicts, provocations are ranked by a model trained on them instead of the built-in weights. A subject shown in the last week is ranked lower.

## Notes are the interface

Everything lands under `Pawn/` (`emergence.vault_root`). Pawn rewrites the sections it manages and keeps any section you add.

| Note | Managed sections | You control |
|---|---|---|
| `Pawn/Threads/{slug}.md` | State, Open questions, Decisions, Timeline | `status:` (`active`, `dismissed`, `parked`, `done`), your own sections. Coworker `file` still appends here. |
| `Pawn/Questions/{slug}.md` | State, Evidence, Timeline | `status:`; write a new note with `pawn: question` to ask one |
| `Pawn/Decisions/{date}-{slug}.md` | Decision, Assumptions (with health), Timeline | `status:`, `council: true` to convene the council |
| `Pawn/Hypotheses/{slug}.md` | Connection, Evidence | `status: active` to keep, `dismissed` to drop |
| `Pawn/Commitments.md` | Overdue, I owe, Owed to me / others, Done recently | — |
| `Pawn/Predictions.md` | Calibration, Open, Resolved | — |
| `Pawn/People/{name}.md` | Recurring concerns, Owes, Is owed, Seen in | — |
| `Pawn/Retros/{period}.md` | Summary, Aged well/badly, Watch next, Priority mirror, Calibration, History | — |

A `status:` you change in a note is applied on the next `sync`; a status changed elsewhere (a verdict, the API) is written back to the note.

`People/{speaker_id}.md` (the Speakers gallery bios) only gets open loops when coworker autonomy allows `people_refresh`; otherwise the digest stays under `Pawn/People/`. Tone trends are not computed.

## Meeting prep

When emergence is on, meeting prep (Day tab) gets a ledger block for the attendees: what you owe them, what they owe, their recent concerns, decisions with shaky assumptions they own.

## Learning

- Labels from triage and compare notes train the extraction ranker and few-shot examples ([COWORKER.md](COWORKER.md#learning-from-your-verdicts)).
- Provocation verdicts train the provocation ranker.
- The Portal **Brainmap** ([PORTAL.md](PORTAL.md#brainmap)) records strings you tie, cut, confirm or snip as `link` labels, and accepting or burning a card as a `subject` label. A cut or snipped pair is remembered by subject and observation text, so routing does not tie it again after a re-process.
- Ghosts proposed by a board chat ([PORTAL.md](PORTAL.md#boards)) that you accept or burn are labelled the same way, with surface `board_chat`.
- Prompt search promotes a rewritten prompt only when it beats the live one on your labels by 0.02 F1; winners are in `{state_dir}/emergence/prompt_overrides.json` and tag observations (`decision.v2+s1`). Delete the file to go back.
- For a model of your own, export the labels and fine-tune outside Pawn:

```bash
pawn-server emergence prompt-search --type decision
pawn-server emergence export --out labels.jsonl
```

## Commands

| Command | What it does |
|---|---|
| `pawn-server emergence night [--job …]` | Run the night shift now |
| `pawn-server emergence subjects [--kind thread] [--status active,proposed]` | List subjects |
| `pawn-server emergence status <id> <status>` | Accept, drop or close a subject (id prefix ok) |
| `pawn-server emergence provocations` | Recent provocations and verdicts |
| `pawn-server emergence verdict <pid> useful\|wrong\|not_now` | Verdict from the shell |
| `pawn-server emergence council <decision-id>` | Run the council now |
| `pawn-server emergence retro --kind week\|month\|quarter` | (Re)write a retrospective |
| `pawn-server emergence prompt-search --type T` | Prompt search for one kind |
| `pawn-server emergence export --out F` | Labels as JSONL |
| `pawn-server emergence render` | Re-render every note |

## HTTP

| Endpoint | |
|---|---|
| `GET /v1/emergence/subjects?kind=&status=` | List subjects |
| `GET /v1/emergence/subjects/{id}` | One subject, its observations and history |
| `POST /v1/emergence/subjects/{id}/status` `{"status": "active"}` | Accept / drop / close |
| `POST /v1/emergence/subjects/{id}/council` | Convene the council (202) |
| `GET /v1/emergence/provocations?days=14` | Recent provocations |
| `POST /v1/emergence/provocations/{pid}/verdict` `{"verdict": "useful"}` | Verdict |
| `POST /v1/emergence/night` `{"jobs": ["synthesis"]}` | Start the night shift (202) |
| `POST /v1/emergence/labels` | A label, e.g. `missed` |

## Settings

All under `emergence:` (env `PAWN_EMERGENCE__*`). The extraction settings are in [COWORKER.md](COWORKER.md#observation-layer-emergence).

| Setting | Default | Meaning |
|---|---|---|
| `directed` | `true` | Run the directed pass after each source |
| `directed_max_subjects` | 12 | Subjects offered per window |
| `route_max_distance` / `route_margin` | 0.32 / 0.04 | Embedding routing without a model call |
| `route_ambiguous_distance` | 0.45 | Up to here, one batched call decides |
| `cluster_distance` / `cluster_min_observations` / `cluster_min_sources` | 0.35 / 4 / 2 | Emergent thread threshold |
| `proposed_expiry_days` | 21 | Unaccepted proposals expire |
| `decision_review_days` | 30 | Review reminder for decisions |
| `commitment_days` | 7 | Undated commitments are overdue after this |
| `night_cron` / `night_max_calls` / `night_max_minutes` | `30 1 * * *` / 400 / 300 | Night shift schedule and budget |
| `night_jobs` | all but `prompt_search` | Which jobs run (a custom list needs `boards` added to work question boards) |
| `brief_max_per_day` | 3 | Provocations per morning |
| `dream_pairs` | 12 | Pairs examined per night (0 turns dreaming off) |
| `dream_max_hypotheses` | 2 | Hypotheses kept per night |
| `board_directed_slots` | 3 | Board questions the directed pass always offers, beyond the nearest subjects |
| `board_backfill_days` / `board_max_candidates` | 30 / 24 | How far back, and how many observations, the `boards` job judges per board |
| `board_dream_pairs` / `board_dream_max` | 6 / 1 | Dreaming among one board's evidence |

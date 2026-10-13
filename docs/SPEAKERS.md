# Speakers gallery

Curated people + manual voiceprints for Pawn diarization.

## Why this exists

Older Pawn builds auto-stored every unmatched diarization turn into the
`embeddings` table (`--store-new` default). That made the gallery grow
noisy and caused wrong nearest-neighbour hits to propagate across sessions.

The Speakers gallery separates two jobs:

1. **Anonymous diarization** — who spoke when (`SPEAKER_XX` turns).
2. **Identification** — match a cluster against *approved* enrollments only.

Runtime diarization **never** writes enrollments. Training is an explicit
step after you have confirmed the person and the audio quality.

## Quick start

```bash
# Migration
alembic upgrade head

# Create people
pawn-diarize speakers create Davide
pawn-diarize speakers create Alice

# Process audio as usual (unknowns stay SPEAKER_XX)
pawn-diarize transcribe-diarize chunk.wav --session monday-standup

# After reviewing the transcript, enroll a good span
pawn-diarize speakers enroll -s Davide --session monday-standup --from SPEAKER_00

# Rematch that session (and later ones) against the gallery
pawn-diarize reidentify --session monday-standup

# Optional: wipe the old auto-accumulated embeddings soup
pawn-diarize speakers purge-legacy-embeddings --confirm
```

## Commands

| Command | Purpose |
|---------|---------|
| `speakers list\|show\|create\|rename\|deactivate` | People registry |
| `speakers enroll` | Approve a voiceprint (`--session/--from` or `--audio`) |
| `speakers enrollments` | List / `--remove` enrollments |
| `speakers reembed` | Rebuild enrollments after an embedding-model change |
| `speakers purge-legacy-embeddings --confirm` | Clear old `embeddings` rows |
| `reidentify --session` | Rematch labels from session centroids |
| `rediarize --session --confirm` | Re-run diarization + identify (keeps ASR text) |
| `retranscribe --session --confirm` | Wipe session + full ASR + diarize from stored S3 audio |

Agent CliTools: `speakers_list`, `speakers_show`, `speakers_update`,
`speaker_enroll`, `session_reidentify`, plus vault bio tools
`people_show` / `people_ensure` / `people_append` (see [PEOPLE.md](PEOPLE.md)).
The agent must confirm quality with you before `speaker_enroll`.

## Obsidian plugin (Sessions → Train / Reprocess)

The Pawn pane **Sessions** tab is primarily for navigating recordings
(default scope **All**, plus speaker chips and search). **Train** on a card
opens a modal that wraps the same gallery operations over HTTP. **Reprocess**
queues CLI-parity `rediarize` (default) or `retranscribe` on the diarize
worker (ASR backend + Whisper language, thresholds); the vault transcript
refreshes when the worker finishes.

| Action | Endpoint |
|--------|----------|
| List / detail | `GET /v1/sessions`, `GET /v1/sessions/{id}` |
| Create person | `POST /v1/speakers` |
| Enroll label centroid | `POST /v1/sessions/{id}/enroll` |
| Rematch gallery | `POST /v1/sessions/{id}/reidentify` |
| Display rename | `POST /v1/sessions/{id}/relabel` |
| Reprocess | `POST /v1/sessions/{id}/reprocess` (`confirm`, `mode`, `backend`, `language`, thresholds) |

Enrollment still uses **session_state centroids** (no browser audio). Confirm
from duration + transcript sample; quality gates (`min_enrollment_seconds`,
pairwise, max enrollments) apply — the UI may offer **force** after a gate
error. Relabel is display-only and does not create a voiceprint.

## Config (`pawnai.yaml`)

```yaml
models:
  diarization_backend: pyannote   # or nemotron
  diarization_model: pyannote/speaker-diarization-community-1
  embedding_model: nvidia/speakerverification_en_titanet_large

speakers:
  identify_threshold: 0.7
  identify_margin: 0.05
  max_enrollments_per_speaker: 5
  min_enrollment_seconds: 1.5
  min_enrollment_pairwise: 0.55
  auto_enroll: false
```

Matching accepts a hit only when `best >= threshold` **and**
`best - second_best >= margin` (open-set reject otherwise).

## Models

- **Embeddings (default):** NeMo TitaNet-Large. Falls back to
  `pyannote/embedding` if TitaNet cannot load. After switching models, run
  `speakers reembed` (requires stored source audio spans).
- **Diarization (default):** pyannote community-1, preferring exclusive
  speaker diarization for ASR alignment. Set `diarization_backend: nemotron`
  for NVIDIA Sortformer:
  - Prefer `nvidia/Nemotron-3-Diarization` when NeMo Speech has RoPE support
    in `TransformerEncoder` (GitHub `NVIDIA-NeMo/Speech` main; PyPI
    `nemo-toolkit==3.0.0` does **not**).
  - Without RoPE, Pawn falls back to `nvidia/diar_sortformer_4spk-v1`.
  - To use Nemotron-3 on an existing venv:
    `uv pip install 'nemo-toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git'`

## Tables

- `speakers` — people
- `speaker_enrollments` — approved voiceprints (JSONB vectors)
- `session_speaker_map` — per-session local label → person

Legacy `embeddings` is no longer written by diarize; purge when ready.

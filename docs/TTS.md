# Text-to-Speech (TTS)

The `POST /v1/audio/speech` endpoint provides OpenAI-compatible TTS powered by
[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) (82M parameters, 24 kHz output).

## Supported Languages

| BCP-47 | Alias accepted | Language | Extra dependency |
|--------|---------------|----------|-----------------|
| `en` / `en-us` | — | American English | — |
| `en-gb` | — | British English | — |
| `es` | — | Spanish | — |
| `fr` / `fr-fr` | — | French | — |
| `hi` | — | Hindi | — |
| `it` | — | Italian | — |
| `pt` / `pt-br` | — | Portuguese (BR) | — |
| `ja` | — | Japanese | `pip install misaki[ja]` |
| `zh` | — | Mandarin Chinese | `pip install misaki[zh]` |

## Voices

Voices are language-specific. OpenAI voice aliases are mapped to the closest
Kokoro equivalent; native Kokoro IDs can also be passed directly.

### American English (`en` / `en-us`)

| OpenAI alias | Kokoro ID | Description |
|---|---|---|
| `alloy` | `af_heart` | American female, warm |
| `nova` | `af_bella` | American female |
| `shimmer` | `af_sarah` | American female |
| `echo` | `am_echo` | American male |
| `onyx` | `am_onyx` | American male |
| `fable` | `bm_george` | British male (fallback) |

Additional native voices: `af_nicole`, `af_sky`, `am_adam`, `am_eric`,
`am_fenrir`, `am_liam`, `am_michael`, `am_puck`, `am_santa`.

### British English (`en-gb`)

Native voices: `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`.

### Other languages

Each language ships with its own set of voices. Pass the native Kokoro ID
directly (e.g. `if_sara` for Italian, `ff_siwis` for French).

## Configuration (`pawnai.yaml`)

```yaml
models:
  tts_language: en          # default BCP-47 language code
  tts_voice: af_heart       # default voice (OpenAI alias or native Kokoro ID)
  tts_device: cpu           # "cpu" or "cuda" (see note below)
  tts_idle_timeout_minutes: 10
  hf_cache_dir: /data/hf_cache  # optional; overrides HF_HUB_CACHE for all HF model downloads
```

`hf_cache_dir` sets `HF_HUB_CACHE` at startup and controls where Kokoro model
weights, voice files, and all other HuggingFace Hub downloads are stored.
Default (when unset) is `~/.cache/huggingface/hub/`.

The setting applies globally — it affects transcription models, diarization
models, and embedding models too, not just TTS.

> **CUDA note:** Kokoro requires a PyTorch build that matches your GPU's
> compute capability. RTX 40-series and earlier work with the standard PyTorch
> wheel. RTX 50-series (Blackwell, sm_120) requires a nightly or future
> PyTorch release — use `tts_device: cpu` in the meantime.

## API

### Request

```
POST /v1/audio/speech
Authorization: Bearer <token>
Content-Type: application/json
```

```json
{
  "model": "tts-1",
  "input": "Hello, world.",
  "voice": "alloy",
  "language": "en",
  "response_format": "wav",
  "speed": 1.0
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | — | Accepted for OpenAI compat, ignored |
| `input` | string | — | Text to synthesise (required) |
| `voice` | string | config `tts_voice` | OpenAI alias or native Kokoro ID |
| `language` | string | config `tts_language` | BCP-47 language code |
| `response_format` | string | `wav` | `wav`, `mp3`, `opus`, `aac`, `flac`, `pcm` |
| `speed` | float | `1.0` | Range 0.25 – 4.0 |

### Response

Binary audio in the requested format. Content-Type header reflects the format:

| `response_format` | `Content-Type` |
|---|---|
| `wav` | `audio/wav` |
| `mp3` | `audio/mpeg` |
| `opus` | `audio/ogg` |
| `aac` | `audio/aac` |
| `flac` | `audio/flac` |
| `pcm` | `audio/pcm` |

### Examples

```bash
# English WAV
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello world","voice":"alloy","response_format":"wav"}' \
  -o out.wav -w "\nHTTP %{http_code}\n"

# Italian MP3
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Ciao mondo","language":"it","response_format":"mp3"}' \
  -o out.mp3 -w "\nHTTP %{http_code}\n"

# Speed control
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Fast speech","speed":1.5,"response_format":"wav"}' \
  -o fast.wav -w "\nHTTP %{http_code}\n"
```

## Model lifecycle

Kokoro pipelines are loaded lazily — one per language, on first use. They are
kept warm as long as requests arrive within `tts_idle_timeout_minutes`. After
that, all cached pipelines are released from memory. The next request reloads
them transparently.

```
first request (lang=en) → load English pipeline → timer armed
request (lang=it)       → load Italian pipeline → timer reset
...idle for 10 min...   → both pipelines unloaded
next request            → reload on demand
```

# Pawn Diarize CLI Architecture

## Single Entrypoint Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  User Commands                                              │
│  $ pawn-diarize [command]                                      │
│  $ python -m pawn-diarize [command]                            │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  🎯 pawn-diarize/__main__.py::main()                           │
│     Single Entry Point                                      │
│     - Handles keyboard interrupts                           │
│     - Manages error reporting                               │
│     - Ensures proper exit codes                             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  CLI Application Router (Typer)                             │
│  pawn-diarize/cli/commands.py                                  │
│  - diarize <audio>                                          │
│  - transcribe <audio>                                       │
│  - embed <audio> --speaker-id ID                            │
│  - search <speaker_id>                                      │
│  - status                                                   │
└────────┬──────────────────┬──────────┬──────────┬───────────┘
         │                  │          │          │
    ┌────▼────┐     ┌───────▼──┐  ┌───▼────┐ ┌──▼────┐
    │diarize  │     │transcribe│  │embed   │ │search │
    │command  │     │command   │  │command │ │command│
    └────┬────┘     └───┬──────┘  └───┬────┘ └──┬────┘
         │              │             │         │
    ┌────▼──────────────▼──────┬──────▼─────────▼┐
    │    Core Business Logic    │  pawn-diarize/   │
    │    pawn-diarize/core/        │  utils/ (CLI) │
    ├───────────────────────────┤               │
    │ • diarization.py          │ • Rich output │
    │ • transcription.py        │ • Progress    │
    │ • embeddings.py           │ • Console     │
    │ • config.py               │               │
    └───────────────────────────┴───────────────┘
         │
    ┌────▼───────────────────────┐
    │   External Services        │
    ├────────────────────────────┤
    │ • Pyannote (diarization)   │
    │ • Nvidia NeMo (ASR)        │
    │ • LanceDB (embeddings)     │
    │ • HuggingFace Models       │
    └────────────────────────────┘
```

## Module Responsibilities

### Entry Point: `pawn-diarize/__main__.py`
```python
main() → app()
```
- ✅ Catches KeyboardInterrupt
- ✅ Handles exceptions gracefully  
- ✅ Sets exit codes properly
- ✅ Displays user-friendly error messages

### CLI Layer: `pawn-diarize/cli/`
- **commands.py**: Command definitions using Typer decorators
- **utils.py**: CLI-specific utilities (Rich console, progress bars)
- **__init__.py**: Exports console for rich output

### Core Layer: `pawn-diarize/core/`
- **config.py**: Configuration class, paths, constants
- **diarization.py**: DiarizationEngine for speaker identification
- **transcription.py**: TranscriptionEngine for speech-to-text
- **embeddings.py**: EmbeddingManager for vector storage
- **__init__.py**: Public API exports

### Utilities: `pawn-diarize/utils/`
- **__init__.py**: General helper functions
- Audio file discovery, validation, etc.

### Tests: `tests/`
- Unit tests for all modules
- CLI integration tests
- Fixture setup in conftest.py

## Installation Options

### For Development
```bash
pip install -e ".[dev]"
```

### For Production
```bash
pip install .
# Then run
pawn-diarize --help
```

### From Git
```bash
git clone <repo>
cd pawn-diarize
pip install -e ".[dev]"
python -m pawn-diarize status
```

## Code Flow Example: Transcribe Command

```
$ pawn-diarize transcribe audio.wav
         ↓
__main__.py:main()
         ↓
app() [Typer router]
         ↓
commands.transcribe(audio_path="audio.wav", with_timestamps=True)
         ↓
core/transcription.py:TranscriptionEngine
    → _initialize_model() [lazy load]
    → transcribe([audio_path])
         ↓
NeMo Parakeet Model (external)
         ↓
results → Rich console output
         ↓
sys.exit(0)
```

## Package Metadata Flow

```
pawn-diarize/__init__.py
  ├─ __version__ = "1.0.0"
  ├─ __author__ = "Pawn Diarize Contributors"
  └─ __description__ = "..."
         ↓
pyproject.toml [tool.setuptools.dynamic]
  ├─ reads version from __init__.py
  ├─ defines entry point: pawn-diarize = pawn-diarize.__main__:main
  └─ lists dependencies
         ↓
Installation: pip install -e .
  ├─ Creates command: pawn-diarize
  └─ Installs dependencies
```

## Dependency Layers

```
pawn-diarize/cli/commands.py
    ↓
pawn-diarize/core/[diarization|transcription|embeddings].py
    ↓ (only imports from)
pawn-diarize/core/config.py
    (no dependencies between core modules)
    ↓
External: torch, pyannote, nemo, lancedb
```

## Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Separation of Concerns**: CLI layer separate from core logic
3. **Lazy Loading**: Models only load when needed
4. **Type Safety**: Full type hints for IDE support
5. **Error Handling**: Graceful failures with informative messages
6. **Testability**: Core logic independent of CLI framework
7. **Extensibility**: Easy to add new commands or core functionality

## Running the Application

### Basic Usage
```bash
# Show help
python -m pawn-diarize --help

# Show status
python -m pawn-diarize status

# Diarize audio
python -m pawn-diarize diarize meeting.wav

# Transcribe
python -m pawn-diarize transcribe speech.wav

# Store embeddings
python -m pawn-diarize embed audio.wav -s speaker_001

# Search similar
python -m pawn-diarize search speaker_001
```

### With Configuration
```bash
# Use custom database
python -m pawn-diarize diarize audio.wav --db-path ./my_db

# No timestamps
python -m pawn-diarize transcribe audio.wav --no-timestamps

# Limit search results
python -m pawn-diarize search speaker_001 --limit 10
```

### Development
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=pawn-diarize

# Format code
black pawn-diarize tests
isort pawn-diarize tests

# Type check
mypy pawn-diarize
```

---

**This architecture ensures Pawn Diarize is:**
- 🎯 Easy to use with a single entry point
- 🏗️ Well-organized with clear separation of concerns
- 📦 Professional and distributable
- 🧪 Testable and maintainable
- 🚀 Scalable for future features

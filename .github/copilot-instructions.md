# GitHub Copilot Instructions - OpenBrain Project

## Project Overview

**OpenBrain** is a professional Python CLI application for speaker diarization, transcription, and speaker embedding management. This document provides project-specific guidelines and architecture for development.

## Project Structure

```
/workspaces/parakeet/
├── openbrain/                          # Main package
│   ├── __init__.py                     # Package metadata & public API
│   ├── __main__.py                     # 🎯 SINGLE CLI ENTRYPOINT
│   ├── core/                           # Core business logic
│   │   ├── __init__.py
│   │   ├── config.py                   # Configuration management
│   │   ├── diarization.py              # Speaker diarization engine
│   │   ├── transcription.py            # Transcription engine
│   │   └── embeddings.py               # Speaker embedding management
│   ├── cli/                            # CLI layer
│   │   ├── __init__.py
│   │   ├── commands.py                 # All CLI commands (Typer-based)
│   │   └── utils.py                    # CLI utilities (Rich console, progress)
│   └── utils/                          # General utilities
│       └── __init__.py                 # Helper functions
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── conftest.py                     # Shared test fixtures
│   ├── test_cli.py                     # CLI integration tests
│   ├── test_core.py                    # Core functionality tests
│   └── test_utils.py                   # Utility tests
├── docs/                               # Documentation (📍 NEW)
│   ├── README.md                       # User guide and command reference
│   ├── ARCHITECTURE.md                 # System architecture & design
│   ├── DEVELOPMENT.md                  # Development workflow
│   ├── PROJECT_ORGANIZATION.md         # Organization summary
│   └── SETUP_COMPLETE.md               # Setup checklist
├── pyproject.toml                      # Modern Python project config
├── setup.py                            # Setup script (backward compatibility)
├── MANIFEST.in                         # Package manifest
├── .gitignore                          # Git ignore rules
├── audio/                              # Audio files directory
├── speakers_db/                        # LanceDB vector database
└── old/                                # Legacy code (ignored by git)
```

## Key Architectural Principles

### 1. Single Entry Point
All CLI commands route through `openbrain/__main__.py:main()`:
```python
def main() -> None:
    """Main entry point function for the CLI."""
    try:
        app()  # Typer app from cli/commands.py
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)
```

### 2. Separation of Concerns
- **Core Layer** (`openbrain/core/`): Pure business logic, no CLI dependencies
- **CLI Layer** (`openbrain/cli/`): Command definitions, argument parsing, formatting
- **Utils Layer** (`openbrain/utils/`): Reusable helper functions

### 3. CLI Framework: Typer
Commands are defined using Typer decorators in `openbrain/cli/commands.py`:
```python
@app.command()
def diarize(
    audio_path: str = typer.Argument(..., help="Path to audio file"),
    db_path: str = typer.Option("speakers_db", help="Database path"),
) -> None:
    """Perform speaker diarization on audio file."""
    pass
```

### 4. Rich Output
Use Rich library for formatted terminal output:
```python
from openbrain.cli.utils import console

console.print("[green]✓ Success[/green]")
console.print("[red]✗ Error[/red]")
with console.status("[bold green]Processing..."):
    # long operation
    pass
```

## CLI Commands

### Available Commands
- **diarize**: Speaker diarization and embedding extraction
- **transcribe**: Speech-to-text transcription
- **embed**: Extract speaker embeddings
- **search**: Find similar speakers
- **status**: System information

### Adding New Commands

1. Add command function to `openbrain/cli/commands.py`:
```python
@app.command()
def new_command(
    required_arg: str = typer.Argument(...),
    optional_opt: str = typer.Option("default"),
) -> None:
    """Short description of the command."""
    # Implementation here
    pass
```

2. Add corresponding core logic to appropriate module in `openbrain/core/`

3. Add tests in `tests/test_cli.py` and `tests/test_core.py`

## Core Module Guidelines

### config.py
- Configuration management and constants
- Paths to databases and models
- Environment variable handling

### diarization.py
- `DiarizationEngine` class for speaker identification
- Embedding extraction using pyannote.audio
- Speaker database management

### transcription.py
- `TranscriptionEngine` class for ASR
- Nvidia NeMo Parakeet model integration
- Timestamp handling

### embeddings.py
- `EmbeddingManager` class for vector storage
- LanceDB integration
- Similarity search functionality

## Development Workflow

### Installation
```bash
cd /workspaces/parakeet
pip install -e ".[dev]"
```

### Running Commands
```bash
# CLI entry point
openbrain --help
openbrain status
openbrain diarize audio.wav

# Python module
python -m openbrain status
python -m openbrain transcribe audio.wav
```

### Code Quality
```bash
# Format code
black openbrain tests
isort openbrain tests

# Lint
flake8 openbrain tests

# Type check
mypy openbrain

# Run tests
pytest
pytest --cov=openbrain
```

## Type Hints & IDE Support

All functions should have complete type hints:
```python
from typing import Optional, List, Dict, Any

def process_audio(
    audio_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Process audio and return transcript.
    
    Args:
        audio_path: Path to audio file
        config: Optional configuration dictionary
    
    Returns:
        List of transcript strings
    
    Raises:
        FileNotFoundError: If audio file not found
        ValueError: If audio format unsupported
    """
    pass
```

## Documentation Standards

### Module Docstrings
```python
"""Module for transcription functionality.

This module provides ASR capabilities using Nvidia NeMo
with the Parakeet model.
"""
```

### Function Docstrings
```python
def transcribe(audio_path: str, timestamps: bool = True) -> str:
    """Transcribe audio to text.
    
    Args:
        audio_path: Path to the audio file
        timestamps: Include word-level timestamps
    
    Returns:
        Transcribed text
    
    Example:
        >>> result = transcribe("speech.wav")
        >>> print(result)
    """
    pass
```

## Testing Guidelines

### Test Structure
- Unit tests in `tests/test_core.py`
- CLI integration tests in `tests/test_cli.py`
- Shared fixtures in `tests/conftest.py`

### Test Example
```python
import pytest
from openbrain.core import diarization

@pytest.fixture
def sample_audio():
    """Provide sample audio for tests."""
    return "tests/fixtures/sample.wav"

def test_diarize(sample_audio):
    """Test speaker diarization."""
    engine = diarization.DiarizationEngine()
    result = engine.diarize(sample_audio)
    assert result is not None
    assert "speakers" in result
```

## Dependencies Management

### pyproject.toml Structure
```toml
[project]
name = "openbrain"
version = "1.0.0"  # Or dynamic from __init__.py
dependencies = [
    "typer[all]>=0.9.0",
    "rich>=13.0.0",
    "pyannote.audio>=2.0",
    "nemo_toolkit[asr]>=1.15",
    "lancedb>=0.1",
    "torch>=1.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "isort>=5.0",
    "mypy>=1.0",
]

[project.scripts]
openbrain = "openbrain.__main__:main"
```

## Common Patterns

### Lazy Loading Models
```python
class TranscriptionEngine:
    def __init__(self):
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            self._model = load_expensive_model()
        return self._model
```

### Error Handling in CLI
```python
try:
    result = core.diarization.diarize(audio_path)
except FileNotFoundError:
    console.print("[red]Error: Audio file not found[/red]")
    sys.exit(1)
except ValueError as e:
    console.print(f"[red]Invalid input: {e}[/red]")
    sys.exit(2)
```

### Configuration Management
```python
from pathlib import Path
from openbrain.core.config import AppConfig

config = AppConfig()
db_path = Path(config.get("db_path", "speakers_db"))
db_path.mkdir(parents=True, exist_ok=True)
```

## Import Organization

### Good Imports
```python
# Standard library first
import sys
from pathlib import Path
from typing import Optional, List

# Third-party second
import typer
from rich.console import Console

# Local imports last
from openbrain.core import diarization
from openbrain.cli.utils import console
```

### Avoid Circular Imports
- Don't import from parent `__init__.py` in child modules
- Import specific classes/functions, not entire modules when possible
- Use `TYPE_CHECKING` for forward references

## Performance Considerations

### Model Initialization
- Use lazy loading (properties with caching)
- Initialize once and reuse in commands

### Database Operations
- Use LanceDB indexing for efficient similarity search
- Batch operations when possible
- Cache frequently accessed embeddings

### Memory Management
- Process audio in chunks if needed
- Clean up large objects after use
- Monitor GPU memory if using CUDA

## Debugging Tips

### Enable Verbose Output
```python
from openbrain.cli.utils import console

console.print("[debug][yellow]Debug message[/yellow][/debug]")
```

### Check System Status
```bash
openbrain status
```

### Inspect Database
```python
from openbrain.core.embeddings import EmbeddingManager

manager = EmbeddingManager()
speakers = manager.list_speakers()
print(speakers)
```

## Documentation Files

- [docs/README.md](docs/README.md): User guide, installation, command reference
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): System design, data flow
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): Development workflow
- [docs/PROJECT_ORGANIZATION.md](docs/PROJECT_ORGANIZATION.md): Organization summary
- [docs/SETUP_COMPLETE.md](docs/SETUP_COMPLETE.md): Setup checklist

## Next Steps for Development

1. **Extend Core Functionality**
   - Implement missing methods in core modules
   - Add more sophisticated embeddings features
   - Expand CLI commands

2. **Improve Testing**
   - Increase test coverage
   - Add integration tests
   - Add performance benchmarks

3. **Documentation**
   - Add API documentation
   - Create usage examples
   - Document configuration options

4. **Distribution**
   - Build package: `python -m build`
   - Upload to PyPI when ready
   - Create GitHub releases

---

**Last Updated**: February 2026
**Project**: OpenBrain
**Status**: Production Ready

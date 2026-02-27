# PawnAI Development Workflow

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Running the Application

```bash
# Run as CLI command (after installation)
pawnai --help
pawnai status

# Run as Python module
python -m pawnai --help
python -m pawnai status
```

## Development Tasks

```bash
# Format code
black pawnai tests
isort pawnai tests

# Lint
flake8 pawnai tests

# Type checking
mypy pawnai

# Run tests
pytest

# Run tests with coverage
pytest --cov=pawnai --cov-report=html
```

## Building and Distributing

```bash
# Build distribution packages
python -m build

# Upload to PyPI (requires credentials)
python -m twine upload dist/*
```

## Project Organization

- **pawnai/**: Main package directory
  - **__main__.py**: Single CLI entrypoint
  - **__init__.py**: Package metadata and public API
  - **core/**: Core business logic (diarization, transcription, embeddings)
  - **cli/**: Command definitions and CLI utilities
  - **utils/**: Helper functions

- **tests/**: Test suite
- **pyproject.toml**: Modern Python project configuration
- **setup.py**: Backward compatibility (optional)
- **MANIFEST.in**: Package manifest

## Key Design Patterns

1. **Single Entrypoint**: All CLI commands go through `pawnai/__main__.py`
2. **Lazy Loading**: Models are loaded on first use, not at startup
3. **Separation of Concerns**: Core logic separated from CLI layer
4. **Type Hints**: Full type annotations for IDE support and safety
5. **Rich Output**: Human-friendly terminal output with colors and formatting

## Common Issues

**ImportError when running commands**
- Ensure you've installed in editable mode: `pip install -e .`
- Check that you're in the correct virtual environment

**Model download fails**
- Set HuggingFace token: `export HF_TOKEN="your_token"`
- Check internet connection and API availability

**Out of memory**
- Models are large (~2GB); ensure sufficient RAM
- Use CPU mode: `export DEVICE="cpu"`

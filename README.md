# PawnAI

A command-line application for speaker diarization, transcription, and speaker embedding management. PawnAI provides a unified interface for audio analysis, speaker identification, and voice clustering.

## Features

- **Speaker Diarization**: Identify and separate multiple speakers in audio
- **Audio Transcription**: Convert speech to text using Nvidia Parakeet model
- **Speaker Embeddings**: Extract and manage speaker voice embeddings
- **Speaker Search**: Find similar speakers based on voice characteristics
- **LanceDB Integration**: Persistent vector database for embeddings
- **GPU Support**: Accelerated processing on CUDA-enabled devices
- **Environment Configuration**: Flexible `.env` file support with custom config paths
- **Secure Credentials**: Built-in .gitignore protection for sensitive tokens

## Installation

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd pawnai

# Install in development mode with dependencies
pip install -e ".[dev]"
```

### System Requirements

- Python 3.10+
- CUDA 11.0+ (optional, for GPU acceleration)
- 8GB+ RAM recommended
- 500MB+ disk space for models

## Quick Start

### 1. Configure Credentials

Save your HuggingFace token in a `.env` file:

```bash
# Copy the example config
cp .env.example .env

# Edit .env and add your token (get it from https://huggingface.co/settings/tokens)
# HF_TOKEN=hf_your_token_here
```

### 2. Verify Installation

```bash
pawnai status
```

### 3. Basic Commands

```bash
# Show system status and available commands
pawnai status

# Diarize an audio file (identify speakers)
pawnai diarize path/to/audio.wav

# Label speakers for future recognition
pawnai label -f audio.wav -s SPEAKER_00 -n "John Doe"

# Transcribe audio to text
pawnai transcribe path/to/audio.wav

# Extract speaker embeddings
pawnai embed path/to/audio.wav --speaker-id speaker_001

# Search for similar speakers
pawnai search speaker_001
```

### Using Python Module

```bash
# Run as Python module
python -m pawnai status
python -m pawnai transcribe audio.wav
```

## Commands Reference

### `diarize`

Perform speaker diarization on an audio file. Identifies and separates different speakers with timestamps showing when each speaker talks. **Automatically recognizes speakers from the database and replaces generic labels with actual names.** Only stores embeddings for new/unknown speakers to avoid database clutter.

```bash
pawnai diarize <audio_path> [OPTIONS]

Arguments:
  audio_path     Path to the audio file to diarize

Options:
  --output TEXT, -o TEXT       Output file path (format inferred from extension: .txt or .json)
  --config TEXT                Path to .env configuration file
  --db-path TEXT               Path to speaker database (default: speakers_db)
  --threshold FLOAT, -t FLOAT  Similarity threshold for speaker matching (0-1, default: 0.7)
  --store-new/--no-store       Store embeddings for unknown speakers (default: enabled)
```

Example:
```bash
pawnai diarize meeting.wav
pawnai diarize meeting.wav -o speakers.json
pawnai diarize meeting.wav --no-store  # Don't save new speakers
pawnai diarize meeting.wav -o speakers.txt --db-path ./my_speakers
pawnai diarize meeting.wav --config ./custom.env  # Use custom config file
pawnai diarize meeting.wav -t 0.8  # Stricter matching (higher threshold)
```

Features:
- **Automatic Recognition**: Searches database for known speakers (0.7 similarity threshold)
- **Smart Storage**: Only stores embeddings for new/unknown speakers (prevents duplicates)
- **Name Replacement**: Displays actual names instead of SPEAKER_00, SPEAKER_01, etc.
- **Custom Configuration**: Load credentials from custom `.env` file

Output includes:
- List of detected speakers (with actual names if recognized)
- Timeline of speaker segments with start/end times
- Duration of each speaking segment
- Matched speakers (original label → actual name)
- New/unknown speakers that were stored

### `transcribe`

Convert speech to text using the Nvidia Parakeet model.

```bash
pawnai transcribe <audio_path> [OPTIONS]

Arguments:
  audio_path  Path to the audio file to transcribe

Options:
  --output TEXT, -o TEXT          Output file path (format inferred from extension: .txt or .json)
  --config TEXT                   Path to .env configuration file
  --timestamps/--no-timestamps    Include word-level timestamps (default: enabled)
  --device TEXT, -d TEXT          Device to use: cuda or cpu (default: cuda)
  --chunk-duration FLOAT, -c      Split audio into chunks of N seconds (helps avoid OOM on long files)
```

Example:
```bash
pawnai transcribe speech.wav
pawnai transcribe speech.wav --no-timestamps
pawnai transcribe speech.wav -o transcript.txt
pawnai transcribe speech.wav -o transcript.json
pawnai transcribe large.mp3 --device cpu
pawnai transcribe large.mp3 -c 300 -o output.txt
pawnai transcribe large.mp3 --config ./prod.env  # Use custom config
pawnai transcribe huge.mp3 --device cpu -c 600
```

### `embed`

Extract speaker embeddings from audio and store in the database.

```bash
pawnai embed <audio_path> --speaker-id SPEAKER_ID [OPTIONS]

Arguments:
  audio_path      Path to the audio file to process

Options:
  --speaker-id    Unique speaker identifier (required)
  -s              Short form of --speaker-id
  --config TEXT   Path to .env configuration file
  --db-path TEXT  Path to speaker database (default: speakers_db)
```

Example:
```bash
pawnai embed person1.wav --speaker-id john_doe
pawnai embed person2.wav -s alice_smith --db-path ./voices
pawnai embed person3.wav -s bob --config ./custom.env
```

### `search`

Search for speakers with similar voice characteristics.

```bash
pawnai search <speaker_id> [OPTIONS]

Arguments:
  speaker_id      Speaker ID to search similar speakers for

Options:
  --config TEXT   Path to .env configuration file
  --db-path TEXT  Path to speaker database (default: speakers_db)
  --limit INT     Maximum number of results (default: 5)
```

Example:
```bash
pawnai search john_doe
pawnai search alice_smith --limit 10 --db-path ./voices
pawnai search bob --config ./custom.env
```

### `label`

Assign human-readable names to speakers for automatic recognition in future diarizations.

```bash
pawnai label [OPTIONS]

Options:
  --file TEXT, -f TEXT     Audio file containing the speaker
  --speaker TEXT, -s TEXT  Speaker label (e.g., SPEAKER_00)
  --name TEXT, -n TEXT     Human-readable name for the speaker
  --config TEXT            Path to .env configuration file
  --list, -l               List all speaker name mappings
  --db-path TEXT           Path to speaker database (default: speakers_db)
```

Example:
```bash
# Label a speaker in a file
pawnai label -f audio.wav -s SPEAKER_00 -n "John Doe"

# List all labeled speakers
pawnai label --list

# Use custom config
pawnai label --list --config ./custom.env
pawnai label -f audio.wav -s SPEAKER_00 -n "Alice Smith" --config ./prod.env
```

**Workflow:**
1. Run `diarize` on an audio file → Detects speakers and stores embeddings for unknown ones
2. Run `label` to assign names → Associates names with speaker embeddings
3. Run `diarize` on new audio → Automatically recognizes and labels known speakers
4. New speakers are automatically stored for future recognition (use `--no-store` to disable)

### `status`

Display system information and available commands.

```bash
pawnai status [--config TEXT]

Options:
  --config TEXT  Path to .env configuration file
```

Shows:
- Device type (CPU/GPU)
- CUDA availability
- Available GPU information
- List of available commands

Example:
```bash
pawnai status
pawnai status --config ./custom.env
```

## Configuration

### Environment Setup with .env File

PawnAI uses a `.env` file to manage environment variables. This keeps sensitive credentials out of your code and shell history.

**Setup:**

1. Copy the example configuration:
```bash
cp .env.example .env
```

2. Edit `.env` and add your HuggingFace token:
```bash
# .env file
HF_TOKEN=hf_your_actual_token_here
DEVICE=auto  # Optional: auto, cuda, or cpu
```

3. The `.env` file is automatically loaded when you run any command:
```bash
pawnai status
pawnai diarize audio.wav
```

**Important**: `.env` is in `.gitignore` and won't be committed to git (protects your credentials).

### Custom Configuration File

You can specify a custom `.env` file using the `--config` flag on any command:

```bash
# Use a specific config file
pawnai diarize audio.wav --config ./configs/prod.env
pawnai transcribe audio.wav --config /path/to/custom.env
pawnai transcribe-diarize audio.wav --config ~/.config/pawnai/.env

# Works with all commands
pawnai embed audio.wav --speaker-id john --config ./custom.env
pawnai search alice --config ./custom.env
pawnai label --list --config ./custom.env
pawnai status --config ./custom.env
```

This is useful for:
- **Multiple environments**: Production, staging, development configs
- **Different tokens**: Separate tokens for different users
- **CI/CD pipelines**: Pass config path as environment variable
- **Docker containers**: Mount config from external volume

### Environment Variables

```bash
# Required: Hugging Face API token for model access
HF_TOKEN=hf_your_token_here

# Optional: Device selection (auto, cuda, cpu) - default: auto
DEVICE=auto

# Optional: Database path - default: speakers_db (from CLI)
# DB_PATH=./my_speakers_db

# Optional: Audio directory - default: audio (from CLI)
# AUDIO_DIR=./audio_files
```

Get your HuggingFace token: https://huggingface.co/settings/tokens

### Configuration Precedence

1. `.env` file in current directory (auto-loaded)
2. Custom `.env` file via `--config` flag (overrides default)
3. CLI options like `--db-path` (take highest precedence)

## Project Structure

```
pawnai/
├── __init__.py           # Package metadata and public API
├── __main__.py           # Single CLI entrypoint
├── core/
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── diarization.py    # Speaker diarization engine
│   ├── transcription.py  # Audio transcription engine
│   └── embeddings.py     # Speaker embedding management
├── cli/
│   ├── __init__.py
│   ├── commands.py       # CLI command definitions
│   └── utils.py          # CLI utilities
└── utils/
    └── __init__.py       # General utilities
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black pawnai tests
isort pawnai tests

# Run linting
flake8 pawnai tests
mypy pawnai

# Run tests
pytest

# Generate coverage report
pytest --cov=pawnai --cov-report=html
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_transcription.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=pawnai
```

## Technologies

- **pyannote.audio**: Speaker diarization and embedding extraction
- **Nvidia NeMo**: ASR transcription with Parakeet model
- **LanceDB**: Vector database for embeddings
- **Typer**: CLI framework with auto-generated help
- **Rich**: Beautiful terminal output formatting
- **python-dotenv**: Environment variable management from .env files
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities

## Performance Tips

- **GPU Acceleration**: Models automatically use CUDA if available
- **Batch Processing**: Process multiple files for efficiency
- **Model Caching**: Models are loaded once and reused
- **Database Indexing**: LanceDB automatically indexes embeddings

## Troubleshooting

### HuggingFace Token Error

If you see "HuggingFace token not found" errors:

1. Create a `.env` file from the example:
```bash
cp .env.example .env
```

2. Add your token:
```
HF_TOKEN=hf_your_actual_token_here
```

3. Or use a custom config file:
```bash
pawnai status --config ./my_config.env
pawnai diarize audio.wav --config /path/to/config.env
```

Get your token: https://huggingface.co/settings/tokens

### Models Not Loading

```bash
# Ensure HuggingFace token is set correctly
cat .env | grep HF_TOKEN

# Check system status
pawnai status
```

### CUDA Issues

```bash
# Use CPU instead (can specify via .env or command)
export DEVICE="cpu"
pawnai transcribe audio.wav

# Or in custom .env file
echo "DEVICE=cpu" >> custom.env
pawnai transcribe audio.wav --config ./custom.env
```

### Out of Memory

- Reduce batch size or process files individually
- Use CPU mode for lower memory consumption
- Ensure no other GPU processes are running

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check existing documentation
- Review example commands

## Acknowledgments

- pyannote.audio for speaker diarization models
- Nvidia NeMo for transcription models
- LanceDB for vector database functionality

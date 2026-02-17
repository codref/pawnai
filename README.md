# PawnAI

A command-line application for speaker diarization, transcription, and speaker embedding management. PawnAI provides a unified interface for audio analysis, speaker identification, and voice clustering.

## Features

- **Speaker Diarization**: Identify and separate multiple speakers in audio
- **Audio Transcription**: Convert speech to text using Nvidia Parakeet model
- **Combined Transcription & Diarization**: Transcribe and label speakers in one command
- **Multi-file Processing**: Process multiple files as one conversation with speaker alignment
- **Session Accumulation**: Process long conversations in parts across multiple invocations
- **Conversation Analysis**: Generate summaries, extract keywords, or build knowledge graphs
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
# Show system status
pawnai status

# Perform speaker identification
pawnai diarize meeting.wav -o speakers.json

# Transcribe and identify speakers together
pawnai transcribe-diarize meeting.wav -o transcript.txt

# Label a speaker for future recognition
pawnai label -f audio.wav -s SPEAKER_00 -n "John Doe"

# Transcribe audio to text
pawnai transcribe audio.wav -o transcript.txt

# Analyze a conversation (summaries, keywords, knowledge graphs)
pawnai analyze result.json --mode summary

# Extract speaker embeddings for database
pawnai embed speaker_audio.wav --speaker-id john_doe

# Search for similar speakers
pawnai search john_doe
```

### 4. Advanced: Multi-File & Session Processing

**Process multiple audio files as one conversation:**

```bash
# Diarize multiple files with speaker alignment
pawnai diarize part1.wav part2.wav part3.wav -o full_speakers.json

# Transcribe multiple files together
pawnai transcribe-diarize chunk1.wav chunk2.wav -o transcript.txt
```

**Process a long conversation in parts over time (session mode):**

```bash
# First part - creates session
pawnai transcribe-diarize part1.wav --session meeting.json

# Second part - appends to session
pawnai transcribe-diarize part2.wav --session meeting.json

# Final part - add remaining audio and save output
pawnai transcribe-diarize part3.wav --session meeting.json -o final_transcript.txt
```

### Using Python Module

```bash
# Run as Python module
python -m pawnai status
python -m pawnai transcribe audio.wav
```

## Commands Reference

### `diarize`

Perform speaker diarization on an audio file or multiple files. Identifies and separates different speakers with timestamps showing when each speaker talks. **Automatically recognizes speakers from the database and replaces generic labels with actual names.** Only stores embeddings for new/unknown speakers to avoid database clutter.

```bash
pawnai diarize <audio_path> [<audio_path>...] [OPTIONS]

Arguments:
  audio_path(s)  One or more audio files to diarize (multiple treated as one conversation)

Options:
  --output TEXT, -o TEXT       Output file path (format inferred from extension: .txt or .json)
  --config TEXT                Path to .env configuration file
  --db-path TEXT               Path to speaker database (default: speakers_db)
  --threshold FLOAT, -t FLOAT  Similarity threshold for speaker matching (0-1, default: 0.7)
  --store-new/--no-store       Store embeddings for unknown speakers (default: enabled)
```

Example:
```bash
# Single file
pawnai diarize meeting.wav
pawnai diarize meeting.wav -o speakers.json
pawnai diarize meeting.wav --no-store  # Don't save new speakers

# Multiple files (ordered chunks of same conversation)
pawnai diarize part1.wav part2.wav part3.wav -o full_diarization.json
pawnai diarize chunk1.wav chunk2.wav -t 0.8  # Stricter matching
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

### `transcribe-diarize`

Transcribe audio with automatic speaker diarization and labeling in one command. Combines transcription with speaker identification.

When multiple files are given they are treated as ordered chunks of the same conversation. Each file is diarized independently, and speaker labels are aligned across files using embedding similarity so the same person gets the same label throughout.

**Session Support**: Use `--session` to accumulate results across multiple invocations. Perfect for processing long conversations in parts.

```bash
pawnai transcribe-diarize <audio_path> [OPTIONS]

Arguments:
  audio_path(s)  One or more audio files to process

Options:
  --output TEXT, -o TEXT              Output file path (.txt or .json)
  --config TEXT                       Path to .env configuration file
  --session TEXT, -s TEXT             Path to session JSON file for accumulative processing
  --db-path TEXT                      Path to speaker database (default: speakers_db)
  --threshold FLOAT, -t FLOAT         Similarity threshold for speaker matching (0-1, default: 0.7)
  --store-new/--no-store              Store embeddings for unknown speakers (default: enabled)
  --device TEXT, -d TEXT              Device to use: cuda or cpu (default: cuda)
  --chunk-duration FLOAT, -c FLOAT    Split audio into chunks (helps avoid OOM)
  --cross-threshold FLOAT, -x FLOAT   Threshold for matching speakers across files (0-1, default: 0.85)
  --no-timestamps                     Hide timestamps in output
```

**Examples:**

Single file:
```bash
pawnai transcribe-diarize meeting.wav
pawnai transcribe-diarize meeting.wav -o transcript.txt
pawnai transcribe-diarize meeting.wav -o transcript.json
```

Multiple files (ordered chunks of same conversation):
```bash
pawnai transcribe-diarize part1.wav part2.wav part3.wav -o full_transcript.txt
pawnai transcribe-diarize part1.wav part2.wav -x 0.9  # Stricter cross-file matching
```

Session accumulation (process files separately over time):
```bash
# First session - creates conv.json if it doesn't exist
pawnai transcribe-diarize part1.wav --session conv.json

# Second session - appends to existing conversation
pawnai transcribe-diarize part2.wav --session conv.json

# Third session - adds more and saves full output
pawnai transcribe-diarize part3.wav --session conv.json -o full_transcript.txt
```

**Session Workflow:**
1. First call with `--session file.json` creates or initializes the session
2. Subsequent calls append new audio while preserving speaker labels
3. Speaker state and timestamps are loaded and continued from previous sessions
4. Final output contains accumulated results from all sessions

**Features:**
- **Combined Analysis**: Single command for transcription + diarization
- **Multi-file Support**: Process multiple files as one conversation (no concatenation)
- **Session Persistence**: Accumulate results across separate command invocations
- **Cross-file Speaker Alignment**: Automatically matches the same speaker across files
- **Automatic Recognition**: Uses speaker database for identification

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

### `analyze`

Analyze diarization results to generate summaries, extract keywords, or build knowledge graphs. Accepts diarization output or processes audio directly.

This command uses AI to extract insights from transcribed conversations with speaker labels.

```bash
pawnai analyze <input_file> [OPTIONS]

Arguments:
  input_file     Path to diarization JSON/text file, or an audio file to process first

Options:
  --output TEXT, -o TEXT       Output file path (.txt, .json, or .csv)
  --mode TEXT                  Analysis mode: 'summary' (default) or 'graph'
  --model TEXT, -m TEXT        Copilot model to use for analysis (default: gpt-4o)
  --db-path TEXT               Path to speaker database (default: speakers_db)
  --device TEXT, -d TEXT       Device for audio processing: cuda or cpu (default: cuda)
```

**Analysis Modes:**

**Summary Mode** (default):
- Structured analysis with conversation summary
- Key topics and keywords extraction
- Per-speaker highlights and insights
- Sentiment analysis

**Graph Mode**:
- Extracts knowledge graph triples (subject, relation, object)
- Perfect for visualizing conversations as networks
- Useful for semantic analysis and relationship mapping

**Examples:**

Analyze existing diarization output:
```bash
pawnai analyze result.json
pawnai analyze transcript.txt --mode graph
pawnai analyze result.json --mode summary -o analysis.txt
```

Process audio directly and analyze:
```bash
pawnai analyze meeting.wav
pawnai analyze meeting.wav --mode graph -o graph.json
pawnai analyze meeting.wav --mode summary -o analysis.txt
```

Output to different formats:
```bash
pawnai analyze result.json --mode graph -o graph.json   # JSON with triples
pawnai analyze result.json --mode graph -o graph.csv    # CSV table format
pawnai analyze result.json --mode summary -o analysis.txt  # Text summary

# Custom model
pawnai analyze result.json --model gpt-4o --mode graph
```

**Output Formats:**

Summary Mode:
- **Text** (.txt): Formatted text report with sections
- **JSON** (.json): Structured JSON with model metadata

Graph Mode:
- **JSON** (.json): Array of {subject, relation, object} triples
- **CSV** (.csv): Three-column table format
- **Text** (.txt): Tab-separated values

**Features:**
- **AI-Powered**: Uses Copilot models for intelligent analysis
- **Multi-format Input**: Works with diarization JSON, text files, or raw audio
- **Flexible Output**: Save as text, JSON, or CSV
- **Knowledge Graphs**: Extract semantic relationships from conversations
- **Speaker Insights**: Per-speaker analysis and sentiment

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

## Multi-File Processing & Session Management

PawnAI supports processing multiple audio files as a single conversation, with intelligent speaker tracking. Two main approaches are available:

### Multi-File Processing (Batch)

Process multiple files in a single command as chunks of one conversation:

```bash
# Diarization
pawnai diarize part1.wav part2.wav part3.wav -o full_diarization.json

# Transcription with diarization
pawnai transcribe-diarize chunk1.wav chunk2.wav chunk3.wav -o transcript.txt

# Key feature: Speaker labels are consistent across all files
```

**How it works:**
1. Each file is processed independently
2. Speaker embeddings from each file are compared
3. Same speaker gets the same label across all files
4. Results are merged into a single output
5. **Cross-file threshold** (`-x`, default 0.85) controls matching strictness

**Use Cases:**
- Recording broken into multiple files due to technical limitations
- Long meetings split into separate audio files
- Processing multiple audio segments of the same conversation

### Session Accumulation (Streaming)

Process a long conversation in parts over time, with state preserved between invocations:

```bash
# Session 1: Start a new conversation session
pawnai transcribe-diarize part1.wav --session meeting.json

# Session 2: Append to existing session (picks up where it left off)
pawnai transcribe-diarize part2.wav --session meeting.json

# Session 3: Add more and export final output
pawnai transcribe-diarize part3.wav --session meeting.json -o final_transcript.txt
```

**Session File Format** (JSON, auto-created):
```json
{
  "speaker_embeddings": { "SPEAKER_00": [...], "SPEAKER_01": [...] },
  "speaker_names": { "SPEAKER_00": "John", "SPEAKER_01": "Alice" },
  "segments": [ 
    { "speaker": "SPEAKER_00", "text": "Hello...", "start": 0.0, "end": 2.5 },
    { "speaker": "SPEAKER_01", "text": "Hi there...", "start": 2.6, "end": 5.0 }
  ],
  "time_cursor": 5.0
}
```

**How it works:**
1. **First call**: Creates session file with initial speaker state
2. **Subsequent calls**: Load existing session, process new audio, align speakers
3. **Time cursor**: Automatically tracks position for proper timestamp continuity
4. **Speaker persistence**: Same speakers matched across all sessions
5. **Final export**: Use `-o` flag to save complete transcript

**Use Cases:**
- Long conversations processed in parts as they arrive
- Real-time transcription with periodic output
- Distributed file processing across multiple machines
- Processing files from different sources over time

**Session vs Batch Comparison:**

| Feature | Batch | Session |
|---------|-------|---------|
| **Command** | Single command | Multiple commands |
| **All files provided** | Yes | No, incremental |
| **State persistence** | None | In JSON file |
| **Speaker alignment** | Cross-file threshold | Embeddings + state |
| **Timeline** | Merged timestamps | Continuous cursor |
| **Use case** | Pre-recorded chunks | Streaming arrival |

### Threshold Options

**`--threshold` / `-t`** (default: 0.7):
- Controls speaker matching within a file
- Range: 0.0 - 1.0
- Higher = stricter matching (fewer false positives)
- Used for diarize and transcribe-diarize

**`--cross-threshold` / `-x`** (default: 0.85):
- Controls speaker matching across multiple files
- Only used when processing multiple files
- Higher = stricter cross-file alignment
- Use higher values (0.85-0.95) for very strict matching

```bash
# More permissive matching
pawnai transcribe-diarize part1.wav part2.wav -t 0.5 -x 0.70

# Stricter matching
pawnai transcribe-diarize part1.wav part2.wav -t 0.85 -x 0.95
```

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

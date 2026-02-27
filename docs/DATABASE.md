# Database Integration Guide

## Overview

PawnAI now stores all data and metadata in SQL databases (SQLite or PostgreSQL) while using LanceDB exclusively for vector embeddings. This provides better data management, querying capabilities, and production scalability.

## Database Architecture

### SQLite (Default)
- **Best for**: Local development, single-user, small to medium datasets
- **Advantages**: No setup required, single file, simple
- **Storage**: `pawnai.db` file in workspace root

### PostgreSQL (Production)
- **Best for**: Production, multi-user, large datasets, advanced queries
- **Advantages**: Better concurrency, JSONB indexing, connection pooling
- **Setup**: Docker Compose (see below)

### LanceDB (Vector Embeddings)
- **Purpose**: Store and search speaker embedding vectors
- **Location**: `speakers_db/` directory
- **Use**: Cosine similarity search for speaker matching

## Quick Start

### 1. Initialize Database

```bash
# SQLite (default - no setup needed)
pawnai init-db

# View status
pawnai status
```

### 2. Configuration

Edit `.pawnai.yml`:

```yaml
database:
  # SQLite (default)
  url: sqlite:///pawnai.db
  
  # Or use PostgreSQL
  # url: postgresql://pawnai:pawnai@localhost:5432/pawnai
```

### 3. Use PostgreSQL (Optional)

```bash
# Start PostgreSQL
docker-compose up -d

# Update .pawnai.yml to use PostgreSQL URL

# Initialize database
pawnai init-db

# Verify
docker-compose ps
pawnai status
```

## Database Schema

### Tables

1. **conversations** - Group related audio files (meetings, projects)
2. **audio_files** - Track uploaded/processed audio with metadata
3. **speakers** - Global speaker records with unique IDs
4. **speaker_segments** - Diarization results (who spoke when)
5. **transcripts** - Full transcription results with word timestamps
6. **sessions** - Incremental processing sessions
7. **speaker_names** - Human-readable labels for speakers

### Key Features

- **Global Speaker IDs**: Speakers identified across all audio files
- **Full Transcripts**: Text + word/segment/character timestamps stored
- **Session Management**: Database-backed (replaces JSON files)
- **Metadata Rich**: Audio duration, sample rate, file sizes, timestamps

## Usage Examples

### Basic Workflow

```bash
# 1. Initialize database
pawnai init-db

# 2. Process audio with diarization
pawnai diarize audio.wav

# 3. Transcribe audio
pawnai transcribe audio.wav

# 4. Combined transcription + diarization
pawnai transcribe-diarize audio.wav

# 5. Label speakers
pawnai label SPEAKER_01 "John Doe"
```

### Data is Automatically Stored

All commands now automatically store results in the database:

- **Diarization**: Creates AudioFile, Speaker, SpeakerSegment records + embeddings in LanceDB
- **Transcription**: Creates AudioFile and Transcript records
- **Combined**: Creates all above + links segments to transcripts

### Sessions

Sessions are now stored in the database (not JSON files):

```bash
# Start a new session
pawnai transcribe-diarize file1.wav --session my-session

# Continue the session with more files
pawnai transcribe-diarize file2.wav file3.wav --session my-session
```

## Docker Compose (PostgreSQL)

### Start PostgreSQL

```bash
docker-compose up -d
```

### With pgAdmin (Database UI)

```bash
docker-compose --profile admin up -d
```

Access pgAdmin at http://localhost:5050:
- Email: `admin@pawnai.local`
- Password: `admin`

### Stop Services

```bash
docker-compose down
```

### Reset Database (WARNING: Deletes all data!)

```bash
# Docker method
docker-compose down -v

# Or via CLI
pawnai init-db --reset
```

## Environment Variables

Override database URL via environment:

```bash
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
pawnai init-db
```

## Migration Notes

### From Old LanceDB-Only Setup

The existing LanceDB data in `speakers_db/` remains untouched but contains old metadata. New data uses the SQL + LanceDB architecture:

- **Old**: LanceDB stores embeddings + speaker names + metadata
- **New**: SQL stores all metadata, LanceDB only stores embeddings

**No automatic migration** - this is a fresh start approach. Old data continues to work but won't appear in SQL queries.

## Installation

### SQLite (Built-in)

```bash
pip install -e .
```

### PostgreSQL (Optional)

```bash
pip install -e ".[postgresql]"
```

## Troubleshooting

### PostgreSQL Connection Failed

```bash
# Check if Docker is running
docker-compose ps

# Check logs
docker-compose logs postgres

# Restart
docker-compose restart

# Verify connection
pawnai status
```

### Database Locked (SQLite)

SQLite uses WAL mode for better concurrency, but if issues persist:
- Ensure no other processes are accessing `pawnai.db`
- Consider using PostgreSQL for multi-user scenarios

### Reset Everything

```bash
# Remove SQLite database
rm pawnai.db

# Or reset from CLI
pawnai init-db --reset

# For PostgreSQL
docker-compose down -v
docker-compose up -d
pawnai init-db
```

## Performance Tips

### SQLite
- Good for < 10GB data
- Single user or low concurrency
- Fast for read-heavy workloads

### PostgreSQL
- Better for > 10GB data
- Multiple concurrent users
- JSONB columns indexed for fast queries
- Connection pooling enabled (5 connections)

## Next Steps

1. **Query Data**: Use DatabaseManager directly in Python scripts
2. **Custom Reports**: SQL queries on transcripts and segments
3. **Speaker Analytics**: Track speaker participation across sessions
4. **Conversation Management**: Group and search audio files
5. **Alembic Migrations**: Add schema versioning (future)

## Support

For issues or questions:
1. Check `pawnai status` for configuration
2. Review `pawnai init-db` output
3. Check logs: `docker-compose logs` (PostgreSQL)
4. Verify schema: inspect `pawnai.db` or query PostgreSQL

---

**Database Type**: SQLite or PostgreSQL
**Vector Storage**: LanceDB
**Configuration**: `.pawnai.yml`
**CLI**: `pawnai init-db`, `pawnai status`

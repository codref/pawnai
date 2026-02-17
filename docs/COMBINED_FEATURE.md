# Combined Transcription and Diarization

## Overview

I've implemented a new combined feature that merges transcription and diarization into a single process. This creates a transcript with speaker labels showing who said what.

## How It Works

The combined process happens in three steps:

1. **Transcription**: Converts speech to text with word-level timestamps
2. **Diarization**: Identifies speakers and their speaking segments
3. **Merging**: Matches timestamps to assign speaker labels to transcript text

## New Files Created

### `/workspaces/parakeet/pawnai/core/combined.py`

This module provides two main functions:

- **`transcribe_with_diarization()`**: Main function that runs both transcription and diarization, then merges the results
- **`format_transcript_with_speakers()`**: Formats the merged results as readable text with speaker labels

### Key Features

- Automatically recognizes known speakers from the database
- Labels new/unknown speakers
- Matches word timestamps with speaker segments
- Supports database speaker matching with configurable threshold
- Optional storage of new speaker embeddings

## New CLI Command

### `transcribe-diarize`

```bash
python -m pawnai transcribe-diarize audio.wav
```

#### Options

- `-o, --output`: Output file (.txt or .json)
- `--db-path`: Path to speaker database (default: speakers_db)
- `-t, --threshold`: Speaker matching threshold 0-1 (default: 0.7)
- `--store-new/--no-store`: Store embeddings for unknown speakers
- `-d, --device`: cuda or cpu (default: cuda)
- `-c, --chunk-duration`: Split audio into chunks (for large files)
- `--no-timestamps`: Hide timestamps in text output

#### Examples

```bash
# Basic usage - output to console
python -m pawnai transcribe-diarize audio.wav

# Save to text file
python -m pawnai transcribe-diarize audio.wav -o transcript.txt

# Save to JSON
python -m pawnai transcribe-diarize audio.wav -o result.json

# Stricter speaker matching
python -m pawnai transcribe-diarize audio.wav -t 0.8

# Don't store new speakers
python -m pawnai transcribe-diarize audio.wav --no-store

# Process large file on CPU with chunks
python -m pawnai transcribe-diarize large.mp3 --device cpu -c 300
```

## Output Format

### Text Output Example

```
============================================================
TRANSCRIPT WITH SPEAKER DIARIZATION
============================================================

Detected 2 speaker(s): John Doe, SPEAKER_01

Matched speakers:
  SPEAKER_00 → John Doe

New/Unknown speakers: SPEAKER_01

------------------------------------------------------------

[00:00.00] John Doe:
  [00:00.00 → 00:03.45] Hello everyone, welcome to today's meeting.
  [00:03.50 → 00:05.20] Let me introduce our agenda.

[00:05.25] SPEAKER_01:
  [00:05.25 → 00:08.10] Thank you for having me here today.
  [00:08.15 → 00:10.50] I'm excited to discuss the project.
```

### JSON Output Structure

```json
{
  "text": "Full transcript text...",
  "speakers": ["John Doe", "SPEAKER_01"],
  "num_speakers": 2,
  "segments": [
    {
      "speaker": "John Doe",
      "start": 0.0,
      "end": 3.45,
      "duration": 3.45,
      "text": "Hello everyone, welcome to today's meeting.",
      "words": [
        {"word": "Hello", "start": 0.0, "end": 0.42},
        {"word": "everyone", "start": 0.45, "end": 0.89},
        ...
      ],
      "num_words": 7
    },
    ...
  ],
  "matched_speakers": {
    "SPEAKER_00": "John Doe"
  },
  "new_speakers": ["SPEAKER_01"]
}
```

## Merging Algorithm

The merging process works by:

1. For each speaker segment (from diarization):
   - Get the segment's start and end time
   - Find all words whose midpoint falls within this time range
   - Combine these words into the segment's text

2. This ensures:
   - Words are accurately attributed to the correct speaker
   - Overlapping segments are handled properly
   - Empty segments are preserved for timing information

## Usage Examples

### Example 1: Basic Transcript with Speakers

```bash
python -m pawnai transcribe-diarize audio/260216230838_01.mp3
```

This will:
- Transcribe the audio
- Identify speakers
- Match against database (if speakers are known)
- Display transcript with speaker labels in console

### Example 2: Save Professional Transcript

```bash
python -m pawnai transcribe-diarize audio/260216230838_01.mp3 -o meeting_transcript.txt
```

Creates a formatted text file with:
- Header showing all speakers
- Speaker changes clearly marked
- Timestamps for each segment
- Easy-to-read format

### Example 3: Process Interview with JSON Output

```bash
python -m pawnai transcribe-diarize interview.wav -o interview.json
```

Creates a JSON file with:
- Full structured data
- Word-level timestamps
- Speaker metadata
- Easy to parse programmatically

### Example 4: Large File Processing

```bash
python -m pawnai transcribe-diarize large_podcast.mp3 --device cpu -c 600 -o podcast.txt
```

For large files:
- Uses CPU to avoid GPU memory limits
- Splits into 10-minute chunks
- Processes sequentially
- Saves to text file

## Testing the Implementation

You can test the new command with any audio file:

```bash
# Quick test with a short file
python -m pawnai transcribe-diarize audio/2086-149220-0033.wav

# Test with MP3 file and speaker matching
python -m pawnai transcribe-diarize audio/260216230838_01.mp3 -t 0.75

# Test with labeled speakers
# First, label a speaker (if you haven't already)
python -m pawnai diarize audio/260216230838_01.mp3
python -m pawnai label -f audio/260216230838_01.mp3 -s SPEAKER_00 -n "Alice"

# Then run combined transcription
python -m pawnai transcribe-diarize audio/260216230838_02.mp3 -o transcript.txt
# If the same speaker appears, they'll be recognized as "Alice"
```

## Implementation Details

### File Changes

1. **Created**: `/workspaces/parakeet/pawnai/core/combined.py`
   - New module for combined functionality

2. **Modified**: `/workspaces/parakeet/pawnai/core/__init__.py`
   - Added exports for `transcribe_with_diarization` and `format_transcript_with_speakers`

3. **Modified**: `/workspaces/parakeet/pawnai/cli/commands.py`
   - Added `transcribe-diarize` command
   - Updated `status` command to list new command

### Code Architecture

The implementation follows the project's architectural principles:

- **Separation of Concerns**: Core logic in `combined.py`, CLI in `commands.py`
- **Single Entry Point**: Command registered through Typer app in `__main__.py`
- **Lazy Loading**: Models only loaded when command is executed
- **Type Hints**: Full type annotations for IDE support
- **Error Handling**: Comprehensive error handling with user-friendly messages

## Benefits

1. **Time Saving**: Single command instead of running two separate processes
2. **Automatic Matching**: Timestamps are automatically aligned
3. **Speaker Recognition**: Known speakers are automatically identified
4. **Flexible Output**: Console, text, or JSON formats
5. **Professional Results**: Clean, readable transcripts with speaker labels

## Next Steps

To use this in production:

1. Label your known speakers using `pawnai label`
2. Run `transcribe-diarize` on new audio files
3. Known speakers will be automatically recognized
4. New speakers will be stored for future recognition
5. Export to your preferred format (TXT or JSON)

## Notes

- The merging algorithm uses word midpoints to determine speaker attribution
- Empty segments (where no words overlap) are preserved for timing accuracy
- The system respects the `similarity_threshold` for speaker matching
- New speakers are automatically stored (unless `--no-store` is used)

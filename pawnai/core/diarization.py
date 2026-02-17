"""Speaker diarization engine using pyannote.audio."""

from typing import Optional, List, Dict, Any, Union, Tuple
import tempfile
import warnings
import os
import torch
import numpy as np
import torchaudio
import soundfile as sf
from pathlib import Path
from pyannote.audio import Pipeline, Model, Inference
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from .config import (
    HUGGINGFACE_TOKEN,
    DIARIZATION_MODEL,
    EMBEDDING_MODEL,
    DEVICE_TYPE,
)

warnings.filterwarnings("ignore")


class DiarizationEngine:
    """Engine for speaker diarization and embedding extraction."""

    def __init__(self, device: Optional[str] = None):
        """Initialize the diarization engine.

        Args:
            device: Device to use ("cuda", "cpu", or "auto"). Defaults to auto-detect.
        """
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.diarization_pipeline: Optional[Pipeline] = None
        self.embedding_model: Optional[Inference] = None

    def _preprocess_audio(self, audio_path: str) -> Union[str, Dict[str, Any]]:
        """Preprocess audio file to ensure compatibility with pyannote.audio.
        
        For MP3/M4A/AAC/FLAC files, loads into memory and returns as dict to avoid
        sample count mismatch errors and CUDA graph compilation issues.
        For other formats (WAV, etc.), returns the file path unchanged.
        
        This approach differs from transcription preprocessing:
        - Transcription creates temporary files only when needed (stereo->mono or chunking)
        - Diarization loads MP3s into memory to avoid torchaudio sample count issues
        - Both ultimately pass data to their respective models in compatible formats
        
        The in-memory approach eliminates temporary file creation while solving
        the "expected X samples instead of Y samples" error that occurs when
        pyannote.audio tries to chunk MP3 files.

        Args:
            audio_path: Path to audio file

        Returns:
            Either a file path string (for WAV, FLAC, etc.) or a dict with 
            'waveform', 'sample_rate', and 'uri' for in-memory processing (MP3, AAC, M4A)
        """
        audio_path_obj = Path(audio_path)
        
        # Only preprocess MP3, AAC, and FLAC files due to potential torchaudio sample count issues
        if audio_path_obj.suffix.lower() in ['.mp3', '.m4a', '.aac', '.flac']:
            print(f"Loading {audio_path_obj.suffix} file into memory for compatibility...")
            
            try:
                # Load audio with torchaudio into memory
                waveform, sample_rate = torchaudio.load(str(audio_path))
                
                # Resample to 16kHz if necessary (standard for speech processing)
                target_sr = 16000
                if sample_rate != target_sr:
                    resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                    waveform = resampler(waveform)
                    sample_rate = target_sr
                    print(f"Resampled to {target_sr}Hz")
                
                # Return as dict for in-memory processing (avoids temporary files)
                audio_dict = {
                    'waveform': waveform,
                    'sample_rate': sample_rate,
                    'uri': audio_path_obj.stem
                }
                print(f"Audio loaded into memory: {waveform.shape} at {sample_rate}Hz")
                
                return audio_dict
            
            except Exception as e:
                print(f"Warning: Could not load {audio_path_obj.suffix} file into memory: {e}")
                print("Attempting to process original file directly...")
                return str(audio_path)
        
        return str(audio_path)

    def _concatenate_to_temp_file(
        self, audio_paths: List[str]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Stream-concatenate multiple audio files into one temporary WAV on disk.

        Files are processed one at a time: each is resampled to 16 kHz and
        mixed to mono, then written sequentially to an ``sf.SoundFile`` opened
        in "w+" mode.  Only one file's waveform is held in RAM at a time, so
        peak memory usage is bounded by the largest single file, not the total.

        Args:
            audio_paths: Ordered list of audio file paths (same conversation)

        Returns:
            Tuple of:
                - ``tmp_path``:  absolute path to the temporary WAV file
                - ``offsets``:   list of ``{path, start, end}`` dicts mapping
                                 each source file to its position (seconds) in
                                 the concatenated stream
        """
        TARGET_SR = 16000
        offsets: List[Dict[str, Any]] = []
        cursor = 0.0

        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix="_concat.wav"
        )
        tmp_path = tmp.name
        tmp.close()

        with sf.SoundFile(
            tmp_path, mode="w", samplerate=TARGET_SR, channels=1, subtype="PCM_16"
        ) as out_f:
            for path in audio_paths:
                waveform, sr = torchaudio.load(str(path))
                if sr != TARGET_SR:
                    waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
                # Mix to mono and convert to float32 numpy
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                audio_np = waveform.squeeze(0).numpy()  # (N,)
                duration = len(audio_np) / TARGET_SR
                offsets.append({"path": str(path), "start": cursor, "end": cursor + duration})
                cursor += duration
                out_f.write(audio_np)
                del waveform, audio_np  # free memory before loading next file

        print(
            f"Wrote concatenated audio to temp file: {tmp_path} "
            f"({cursor:.1f}s total from {len(audio_paths)} files)"
        )
        return tmp_path, offsets

    def _diarize_multiple_files(
        self,
        audio_paths: List[str],
        cross_file_threshold: float = 0.85,
        prior_speaker_embeddings: Optional[Dict[str, Any]] = None,
        time_cursor: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Diarize each file independently and align speakers across files.

        No audio concatenation is performed.  Each file is diarized on its own;
        speaker embeddings from each file are compared against all globally-seen
        speakers so that the same person gets the same global label regardless of
        which file they appear in.

        Supports incremental sessions: pass ``prior_speaker_embeddings`` and
        ``time_cursor`` to seed speaker state from a previous call so that
        speakers are recognised across sessions and all new segments carry
        correctly offset timestamps.

        Args:
            audio_paths: Ordered list of audio file paths.
            cross_file_threshold: Cosine-similarity threshold for merging two
                local speaker labels into one global label (0-1).
            prior_speaker_embeddings: Per-speaker state from a previous session.
                Mapping of global_label → {"embedding": [...], "total_duration": float}.
                When provided the global speaker pool is pre-seeded so speakers
                from past calls are recognised without re-processing old audio.
            time_cursor: Seconds of audio already processed in previous calls.
                All new segment timestamps are shifted by this value.

        Returns:
            Tuple of:
                - segments: Time-offset segments with globally consistent labels,
                            sorted by start time.
                - speaker_embeddings: Mapping of global label →
                                      list of {embedding, start, end} dicts
                                      (current call's data only; excludes synthetic
                                       prior-session entries).
                - chunk_offsets: List of {path, start, end} for each source file
                                 using global time positions.
        """
        TARGET_SR = 16000
        all_segments: List[Dict[str, Any]] = []
        # global_label → list of {embedding, start, end}
        global_speaker_embeddings: Dict[str, List[Dict[str, Any]]] = {}
        chunk_offsets: List[Dict[str, Any]] = []
        # time_cursor is a running counter; initialised from the parameter

        # --------------------------------------------------------------------
        # Seed global speaker state from a prior session
        # --------------------------------------------------------------------
        if prior_speaker_embeddings:
            for label, info in prior_speaker_embeddings.items():
                emb = np.array(info["embedding"])
                emb = emb / np.linalg.norm(emb)
                # Represent the prior session as a single synthetic segment
                # whose "duration" equals total_duration so that the weight
                # given to it in any subsequent averaging is proportional to
                # how much real audio it was derived from.
                # The "synthetic" flag prevents these entries from being stored
                # in the speaker database as if they were new audio.
                global_speaker_embeddings[label] = [{
                    "embedding": emb,
                    "start": 0.0,
                    "end": info["total_duration"],
                    "synthetic": True,
                }]
            # Set counter past the highest SPEAKER_NN number already in use
            nums = [
                int(lbl.split("_")[-1])
                for lbl in prior_speaker_embeddings
                if lbl.startswith("SPEAKER_") and lbl.split("_")[-1].isdigit()
            ]
            global_counter = (max(nums) + 1) if nums else len(prior_speaker_embeddings)
        else:
            global_counter = 0

        for file_idx, path in enumerate(audio_paths):
            print(f"  Diarizing file {file_idx + 1}/{len(audio_paths)}: {path}")

            # Preprocess handles format conversion; keep waveform for embedding extraction
            processed_audio = self._preprocess_audio(path)

            if isinstance(processed_audio, dict):
                waveform = processed_audio["waveform"]
                sample_rate = processed_audio["sample_rate"]
            else:
                waveform, sample_rate = torchaudio.load(str(processed_audio))
                if sample_rate != TARGET_SR:
                    waveform = torchaudio.transforms.Resample(sample_rate, TARGET_SR)(waveform)
                    sample_rate = TARGET_SR

            file_duration = waveform.shape[1] / sample_rate
            chunk_offsets.append({
                "path": str(path),
                "start": time_cursor,
                "end": time_cursor + file_duration,
            })

            # Run diarization pipeline on this file
            diarization_output = self.diarization_pipeline(processed_audio)
            diarization = diarization_output.speaker_diarization

            # ------------------------------------------------------------------
            # Extract per-segment embeddings and accumulate per local speaker
            # ------------------------------------------------------------------
            local_speaker_embeddings: Dict[str, List[Dict[str, Any]]] = {}

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                seg_start = turn.start
                seg_end = turn.end
                if seg_end - seg_start < 0.5:
                    continue

                start_idx = int(seg_start * sample_rate)
                end_idx = int(seg_end * sample_rate)
                segment_audio = waveform[:, start_idx:end_idx]

                try:
                    if segment_audio.device != torch.device("cpu"):
                        segment_audio = segment_audio.cpu()

                    embedding = self.embedding_model({
                        "waveform": segment_audio,
                        "sample_rate": sample_rate,
                    })
                    if not isinstance(embedding, np.ndarray):
                        embedding = (
                            embedding.numpy()
                            if hasattr(embedding, "numpy")
                            else np.array(embedding)
                        )
                    embedding = embedding / np.linalg.norm(embedding)

                    local_speaker_embeddings.setdefault(speaker, []).append({
                        "embedding": embedding,
                        "start": seg_start,
                        "end": seg_end,
                    })
                except Exception as e:
                    print(f"Warning: Could not extract embedding at {seg_start:.2f}s: {e}")

            # ------------------------------------------------------------------
            # Compute duration-weighted mean embedding per local speaker
            # ------------------------------------------------------------------
            def _mean_emb(emb_list: List[Dict[str, Any]]) -> np.ndarray:
                durations = np.array([e["end"] - e["start"] for e in emb_list])
                weights = durations / durations.sum()
                stacked = np.stack([e["embedding"].flatten() for e in emb_list])
                mean = np.average(stacked, axis=0, weights=weights)
                return mean / np.linalg.norm(mean)

            local_mean_embeddings: Dict[str, np.ndarray] = {
                lbl: _mean_emb(lst)
                for lbl, lst in local_speaker_embeddings.items()
            }

            # ------------------------------------------------------------------
            # Cross-file speaker alignment
            # ------------------------------------------------------------------
            local_to_global: Dict[str, str] = {}

            if not global_speaker_embeddings:
                # First file — assign new global labels directly
                for local_label in local_mean_embeddings:
                    global_label = f"SPEAKER_{global_counter:02d}"
                    global_counter += 1
                    local_to_global[local_label] = global_label
                    global_speaker_embeddings[global_label] = list(
                        local_speaker_embeddings.get(local_label, [])
                    )
                    print(f"    {local_label} → {global_label} (new)")
            else:
                # Subsequent files — compute current global mean embeddings once
                global_labels = list(global_speaker_embeddings.keys())
                global_means = [_mean_emb(global_speaker_embeddings[gl]) for gl in global_labels]

                for local_label, local_mean in local_mean_embeddings.items():
                    similarities = [
                        float(np.dot(local_mean.flatten(), gm.flatten()))
                        for gm in global_means
                    ]
                    best_idx = int(np.argmax(similarities))
                    best_sim = similarities[best_idx]

                    if best_sim >= cross_file_threshold:
                        best_global = global_labels[best_idx]
                        local_to_global[local_label] = best_global
                        # Merge embeddings into global pool
                        global_speaker_embeddings[best_global].extend(
                            local_speaker_embeddings.get(local_label, [])
                        )
                        print(
                            f"    {local_label} → {best_global} "
                            f"(same speaker, similarity={best_sim:.3f})"
                        )
                    else:
                        global_label = f"SPEAKER_{global_counter:02d}"
                        global_counter += 1
                        local_to_global[local_label] = global_label
                        global_speaker_embeddings[global_label] = list(
                            local_speaker_embeddings.get(local_label, [])
                        )
                        print(
                            f"    {local_label} → {global_label} "
                            f"(new speaker, best_sim={best_sim:.3f})"
                        )

            # ------------------------------------------------------------------
            # Build time-offset segments with global labels
            # ------------------------------------------------------------------
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.end - turn.start < 0.5:
                    continue
                global_label = local_to_global.get(speaker, speaker)
                all_segments.append({
                    "speaker": global_label,
                    "original_label": speaker,
                    "start": turn.start + time_cursor,
                    "end": turn.end + time_cursor,
                    "duration": turn.end - turn.start,
                    "source_file": str(path),
                })

            time_cursor += file_duration
            del waveform  # free before loading next file

        all_segments.sort(key=lambda x: x["start"])
        return all_segments, global_speaker_embeddings, chunk_offsets

    def _initialize_models(self) -> None:
        """Initialize diarization and embedding models (lazy loading)."""
        if self.diarization_pipeline is not None and self.embedding_model is not None:
            return  # Already initialized

        print(f"Using device: {self.device}")
        print("Initializing diarization pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL, token=HUGGINGFACE_TOKEN
        ).to(self.device)

        print("Initializing embedding model...")
        self.embedding_model = Inference(
            Model.from_pretrained(
                EMBEDDING_MODEL, token=HUGGINGFACE_TOKEN
            ).to(self.device),
            window="whole",
        )
        print("Models initialized successfully")

    def diarize(
        self,
        audio_path: Union[str, List[str]],
        db_path: Optional[str] = None,
        similarity_threshold: float = 0.7,
        store_new_speakers: bool = True,
        cross_file_threshold: float = 0.85,
        prior_speaker_embeddings: Optional[Dict[str, Any]] = None,
        time_cursor: float = 0.0,
    ) -> Dict[str, Any]:
        """Perform speaker diarization on audio file(s) with database lookup.

        Supports incremental session processing: pass ``prior_speaker_embeddings``
        and ``time_cursor`` from a saved session to recognise speakers across
        separate CLI invocations without re-processing old audio.

        Each new file is diarized independently and speaker labels are aligned
        using embedding similarity.  No audio concatenation is created.

        Args:
            audio_path: Path to audio file, or ordered list of paths.
            db_path: Path to LanceDB database (None to skip database lookup).
            similarity_threshold: Minimum cosine similarity to match a speaker
                against the database (0-1).
            store_new_speakers: Whether to store embeddings for unknown speakers.
            cross_file_threshold: Cosine-similarity threshold for aligning
                speaker labels across files (0-1).
            prior_speaker_embeddings: Per-speaker state from a previous session.
                Mapping of global_label → {"embedding": [...], "total_duration": float}.
                When provided all new speakers are compared against these so that
                the same person receives the same label across sessions.
            time_cursor: Seconds of audio already processed in previous calls.
                All new segment timestamps are shifted by this value.

        Returns:
            Dictionary containing:
                - speakers: List of unique speaker IDs (with names if found)
                - segments: List of dicts with speaker, start, end times
                - num_speakers: Total number of speakers detected
                - matched_speakers: Dict mapping global labels to matched names
                - new_speakers: List of speaker labels that were not matched
                - chunk_offsets: list of {path, start, end} when multiple files given
                - session_speaker_embeddings: Serialisable per-speaker embeddings
                  (global_label → {"embedding": [...], "total_duration": float})
                  suitable for persisting in a session file and passing back as
                  prior_speaker_embeddings on the next incremental call.
                - new_time_cursor: Updated total seconds processed; pass as
                  time_cursor on the next incremental call.
        """
        import lancedb
        import os

        self._initialize_models()

        # Normalise to list
        audio_paths: List[str] = (
            [audio_path] if isinstance(audio_path, str) else list(audio_path)
        )

        # ------------------------------------------------------------------
        # STEP 1: Diarize and collect segments + per-speaker embeddings
        # ------------------------------------------------------------------
        segments: List[Dict[str, Any]] = []
        speaker_embeddings: Dict[str, List[Dict[str, Any]]] = {}
        chunk_offsets: Optional[List[Dict[str, Any]]] = None
        file_duration: float = 0.0  # set in single-file branch for new_time_cursor

        # Route single file with prior state through the multi-file path so that
        # prior speaker embeddings and the time offset are handled uniformly.
        use_multi_file = len(audio_paths) > 1 or bool(prior_speaker_embeddings)

        if use_multi_file:
            resume_note = f", resuming from t={time_cursor:.1f}s" if time_cursor > 0 else ""
            print(
                f"Diarizing {len(audio_paths)} file(s) independently "
                f"(cross-file threshold={cross_file_threshold}{resume_note})…"
            )
            segments, speaker_embeddings, chunk_offsets = self._diarize_multiple_files(
                audio_paths,
                cross_file_threshold=cross_file_threshold,
                prior_speaker_embeddings=prior_speaker_embeddings,
                time_cursor=time_cursor,
            )
            speakers: set = {seg["speaker"] for seg in segments}
        else:
            # ----- single-file baseline path (no prior state) -----
            processed_audio = self._preprocess_audio(audio_paths[0])

            if isinstance(processed_audio, dict):
                print(f"Diarizing in-memory audio from: {audio_paths[0]}")
                waveform = processed_audio["waveform"]
                sample_rate = processed_audio["sample_rate"]
            else:
                print(f"Diarizing: {processed_audio}")
                waveform, sample_rate = torchaudio.load(str(processed_audio))

            file_duration = waveform.shape[1] / sample_rate

            diarization_output = self.diarization_pipeline(processed_audio)
            diarization = diarization_output.speaker_diarization

            speakers = set()
            print("Extracting speaker embeddings...")
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = turn.start
                end_time = turn.end
                if end_time - start_time < 0.5:
                    continue

                start_idx = int(start_time * sample_rate)
                end_idx = int(end_time * sample_rate)
                segment_audio = waveform[:, start_idx:end_idx]

                try:
                    if segment_audio.device != torch.device("cpu"):
                        segment_audio = segment_audio.cpu()

                    embedding = self.embedding_model({
                        "waveform": segment_audio,
                        "sample_rate": sample_rate,
                    })
                    if not isinstance(embedding, np.ndarray):
                        embedding = (
                            embedding.numpy()
                            if hasattr(embedding, "numpy")
                            else np.array(embedding)
                        )
                    embedding = embedding / np.linalg.norm(embedding)

                    speaker_embeddings.setdefault(speaker, []).append({
                        "embedding": embedding,
                        "start": start_time,
                        "end": end_time,
                    })
                except Exception as e:
                    print(f"Warning: Could not extract embedding at {start_time:.2f}s: {e}")

                speakers.add(speaker)

        # ------------------------------------------------------------------
        # STEP 2: Database lookup — match each speaker against known embeddings
        # ------------------------------------------------------------------
        try:
            db = None
            embeddings_table = None
            names_table = None
            if db_path:
                try:
                    db = lancedb.connect(db_path)
                    existing_tables = db.table_names()
                    if "embeddings" in existing_tables:
                        embeddings_table = db.open_table("embeddings")
                    if "speaker_names" in existing_tables:
                        names_table = db.open_table("speaker_names")
                except Exception as e:
                    print(f"Warning: Could not connect to database: {e}")

            matched_speakers: Dict[str, str] = {}
            new_speakers: List[str] = []

            if embeddings_table is not None:
                print("Searching database for speaker matches...")
                for speaker_label, emb_list in speaker_embeddings.items():
                    if not emb_list:
                        continue

                    # Duration-weighted mean embedding as the query vector
                    durations = np.array([e["end"] - e["start"] for e in emb_list])
                    weights = durations / durations.sum()
                    stacked = np.stack([e["embedding"].flatten() for e in emb_list])
                    mean_embedding = np.average(stacked, axis=0, weights=weights)
                    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
                    query_embedding = mean_embedding.tolist()

                    try:
                        search_results = (
                            embeddings_table.search(query_embedding)
                            .metric("cosine")
                            .limit(1)
                            .to_pandas()
                        )

                        if not search_results.empty:
                            best_match = search_results.iloc[0]
                            similarity = 1 - best_match["_distance"]

                            if similarity >= similarity_threshold:
                                matched_label = best_match["local_speaker_label"]
                                matched_file = best_match["audio_file"]

                                speaker_name = None
                                if names_table is not None:
                                    names_df = names_table.to_pandas()
                                    mask = (
                                        (names_df["audio_file"] == matched_file)
                                        & (names_df["local_speaker_label"] == matched_label)
                                    )
                                    if mask.any():
                                        speaker_name = names_df[mask].iloc[0]["speaker_name"]

                                if speaker_name:
                                    print(
                                        f"  ✓ Matched {speaker_label} → '{speaker_name}' "
                                        f"(similarity={similarity:.3f})"
                                    )
                                    matched_speakers[speaker_label] = speaker_name
                                else:
                                    matched_id = best_match["id"]
                                    print(
                                        f"  Found match for {speaker_label} but no name assigned "
                                        f"(id: {matched_id}, similarity={similarity:.3f})"
                                    )
                            else:
                                new_speakers.append(speaker_label)
                        else:
                            new_speakers.append(speaker_label)
                    except Exception as e:
                        print(f"Warning: Could not search for {speaker_label}: {e}")
                        new_speakers.append(speaker_label)
            else:
                new_speakers = list(speaker_embeddings.keys())

            # ------------------------------------------------------------------
            # STEP 3: Store embeddings for new/unknown speakers
            # ------------------------------------------------------------------
            if store_new_speakers and new_speakers and db_path and db:
                print(f"Storing embeddings for {len(new_speakers)} new speaker(s)...")
                records_to_add = []
                # For multi-file runs the "source file" is the first file by convention;
                # the per-file origin is captured in source_file field of each segment.
                source_file = audio_paths[0]

                for speaker_label in new_speakers:
                    if speaker_label not in speaker_embeddings:
                        continue
                    for idx, emb_data in enumerate(speaker_embeddings[speaker_label]):
                        records_to_add.append({
                            "id": f"{os.path.basename(source_file)}_{speaker_label}_{idx}",
                            "audio_file": str(source_file),
                            "local_speaker_label": speaker_label,
                            "start_time": float(emb_data["start"]),
                            "end_time": float(emb_data["end"]),
                            "embedding": emb_data["embedding"].flatten().tolist(),
                        })

                if records_to_add:
                    try:
                        if embeddings_table is None:
                            embeddings_table = db.create_table("embeddings", data=records_to_add)
                            print(f"  Created embeddings table with {len(records_to_add)} records")
                        else:
                            embeddings_table.add(records_to_add)
                            print(f"  Added {len(records_to_add)} embeddings to database")
                    except Exception as e:
                        print(f"Warning: Could not store embeddings: {e}")

            # ------------------------------------------------------------------
            # STEP 4: Build final segment list, applying matched names
            # ------------------------------------------------------------------
            if use_multi_file:
                # Segments already built by _diarize_multiple_files; just rename speakers
                for seg in segments:
                    seg["speaker"] = matched_speakers.get(seg["speaker"], seg["speaker"])
            else:
                # Build segments from single-file diarization annotation
                for turn, _, speaker in diarization.itertracks(yield_label=True):  # type: ignore[union-attr]
                    if turn.end - turn.start < 0.5:
                        continue
                    display_name = matched_speakers.get(speaker, speaker)
                    segments.append({
                        "speaker": display_name,
                        "original_label": speaker,
                        "start": turn.start + time_cursor,
                        "end": turn.end + time_cursor,
                        "duration": turn.end - turn.start,
                    })
                segments.sort(key=lambda x: x["start"])

            # Final speaker list with matched names applied
            final_speakers = sorted({
                matched_speakers.get(s, s) for s in speakers
            })

            # ------------------------------------------------------------------
            # Build session_speaker_embeddings for incremental use
            # The DW-mean is computed over all entries, including any synthetic
            # prior-session entry, so history is correctly blended with new data.
            # ------------------------------------------------------------------
            session_speaker_embeddings: Dict[str, Any] = {}
            for label, emb_list in speaker_embeddings.items():
                real_entries = [e for e in emb_list if not e.get("synthetic", False)]
                if not real_entries and not emb_list:
                    continue
                # Use all entries (including synthetic prior) for the mean so
                # that prior history is weighted proportionally.
                entries_for_mean = emb_list if emb_list else real_entries
                durations = np.array([e["end"] - e["start"] for e in entries_for_mean])
                weights = durations / durations.sum() if durations.sum() > 0 else durations
                stacked = np.stack([e["embedding"].flatten() for e in entries_for_mean])
                mean_emb = np.average(stacked, axis=0, weights=weights)
                norm = np.linalg.norm(mean_emb)
                if norm > 0:
                    mean_emb = mean_emb / norm
                total_dur = float(durations.sum())
                # Key by matched/display name so sessions carry human-readable labels
                display_label = matched_speakers.get(label, label)
                session_speaker_embeddings[display_label] = {
                    "embedding": mean_emb.tolist(),
                    "total_duration": total_dur,
                }

            # Updated time cursor for the next incremental call
            if chunk_offsets:
                new_time_cursor = chunk_offsets[-1]["end"]
            else:
                new_time_cursor = time_cursor + file_duration

            result: Dict[str, Any] = {
                "speakers": final_speakers,
                "segments": segments,
                "num_speakers": len(speakers),
                "matched_speakers": matched_speakers,
                "new_speakers": new_speakers,
                "session_speaker_embeddings": session_speaker_embeddings,
                "new_time_cursor": new_time_cursor,
            }

            # Attach chunk offsets when multiple files were used
            if chunk_offsets is not None:
                result["chunk_offsets"] = chunk_offsets

            return result

        except Exception:
            raise


    def extract_embeddings(self, audio_path: Union[str, List[str]]) -> np.ndarray:
        """Extract a speaker embedding from one or more audio files.

        For a single file the embedding is extracted directly.
        For multiple files each file is processed individually and the
        duration-weighted average of the per-file embeddings is returned.
        This avoids loading all files into memory simultaneously.

        Args:
            audio_path: Path (or ordered list of paths) to audio file(s)

        Returns:
            1-D normalised embedding array
        """
        self._initialize_models()

        audio_paths: List[str] = [audio_path] if isinstance(audio_path, str) else list(audio_path)

        if len(audio_paths) == 1:
            processed_audio = self._preprocess_audio(audio_paths[0])
            embedding = self.embedding_model(processed_audio)
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            return embedding / np.linalg.norm(embedding)

        # Multiple files: weighted average of per-file embeddings
        print(f"Extracting embeddings from {len(audio_paths)} files (per-file average)…")
        embeddings_list: List[np.ndarray] = []
        durations: List[float] = []

        for path in audio_paths:
            processed = self._preprocess_audio(path)
            emb = self.embedding_model(processed)
            if not isinstance(emb, np.ndarray):
                emb = np.array(emb)
            emb = emb / np.linalg.norm(emb)
            embeddings_list.append(emb.flatten())

            # Measure duration for weighting
            waveform, sr = torchaudio.load(str(path))
            durations.append(waveform.shape[1] / sr)
            del waveform  # free immediately

        weights = np.array(durations)
        weights = weights / weights.sum()
        mean_emb = np.average(np.stack(embeddings_list), axis=0, weights=weights)
        return mean_emb / np.linalg.norm(mean_emb)

    def cluster_speakers(
        self, embeddings: List[np.ndarray], eps: float = 0.5, min_samples: int = 2
    ) -> List[int]:
        """Cluster speakers using DBSCAN.

        Args:
            embeddings: List of embedding vectors
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter

        Returns:
            List of cluster labels
        """
        if len(embeddings) == 0:
            return []

        embeddings_array = np.array(embeddings)
        distances = cosine_distances(embeddings_array)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = clustering.fit_predict(distances)
        return labels.tolist()

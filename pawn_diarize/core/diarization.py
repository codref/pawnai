"""Speaker diarization engine with curated Speakers-gallery identification.

Pipeline (two separate jobs — do not conflate them):

1. **Anonymous diarization** — a pluggable backend (pyannote / Nemotron) emits
   ``SPEAKER_XX`` turns for "who spoke when".
2. **Identification** — duration-weighted cluster embeddings are scored against
   the curated Speakers gallery only.  Unknowns stay anonymous; nothing is
   auto-enrolled into the gallery at runtime.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
import re
import tempfile
import warnings
import os
import torch
import numpy as np
import soundfile as sf
from math import gcd
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from .config import (
    HUGGINGFACE_TOKEN,
    DIARIZATION_MODEL,
    EMBEDDING_MODEL,
    DEVICE_TYPE,
)

warnings.filterwarnings("ignore")

_ANON_SPEAKER_RE = re.compile(r"^SPEAKER_\d+$")


def is_anonymous_speaker_label(label: str) -> bool:
    """True for backend-local labels like ``SPEAKER_00`` (not display names)."""
    return bool(_ANON_SPEAKER_RE.match(str(label or "").strip()))


def _load_audio(path: str) -> Tuple[torch.Tensor, int]:
    """Load an audio file into a (channels, frames) float32 tensor.

    Uses soundfile for lossless formats (WAV, FLAC, AIFF, etc.) and
    librosa as a fallback for compressed formats (MP3, M4A, AAC).
    Avoids torchcodec / torchaudio entirely.
    """
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T.copy())  # (channels, frames)
        return waveform, sr
    except Exception:
        import librosa  # lazy import – only needed for MP3/M4A/AAC
        data, sr = librosa.load(str(path), sr=None, mono=False, dtype=np.float32)
        if data.ndim == 1:
            data = data[np.newaxis, :]  # ensure (channels, frames)
        return torch.from_numpy(data), sr


def _resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample waveform tensor from orig_sr to target_sr using scipy."""
    if orig_sr == target_sr:
        return waveform
    from scipy.signal import resample_poly
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    data = waveform.numpy()  # (channels, frames)
    resampled = resample_poly(data, up, down, axis=-1).astype(np.float32)
    return torch.from_numpy(resampled)


class DiarizationEngine:
    """Engine for speaker diarization and embedding extraction.

    Args:
        device: ``cuda`` / ``cpu`` / ``auto``.
        diarization_backend: ``pyannote`` or ``nemotron``.
        diarization_model: HF / NeMo model id for the chosen backend.
        embedding_model: voiceprint extractor id (TitaNet or pyannote).
        hf_token: Hugging Face token for gated pyannote models.
        identify_margin: min gap between best and second-best gallery scores.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        diarization_backend: str = "pyannote",
        diarization_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        hf_token: Optional[str] = None,
        identify_margin: float = 0.05,
    ):
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.diarization_backend_name = diarization_backend
        self.diarization_model_id = diarization_model or DIARIZATION_MODEL
        self.embedding_model_id = embedding_model or EMBEDDING_MODEL
        self.hf_token = hf_token if hf_token is not None else HUGGINGFACE_TOKEN
        self.identify_margin = identify_margin

        # Lazy-loaded in _initialize_models().
        self._backend = None
        self._extractor = None
        # Legacy aliases kept so older call sites / tests keep working.
        self.diarization_pipeline = None
        self.embedding_model = None

    def _initialize_models(self) -> None:
        """Lazy-load the diarization backend and embedding extractor."""
        if self._backend is not None and self._extractor is not None:
            return

        from .diar_backends import (
            build_diarization_backend,
            resolve_diarization_model_id,
            resolve_nemotron_runtime_model_id,
        )
        from .voice_embeddings import build_embedding_extractor

        device_str = str(self.device)
        print(f"Using device: {self.device}")

        resolved_model = resolve_diarization_model_id(
            self.diarization_backend_name, self.diarization_model_id
        )
        if self.diarization_backend_name.lower() == "nemotron":
            resolved_model = resolve_nemotron_runtime_model_id(resolved_model)
        self.diarization_model_id = resolved_model
        print(
            f"Initializing diarization backend={self.diarization_backend_name} "
            f"model={resolved_model}..."
        )
        self._backend = build_diarization_backend(
            self.diarization_backend_name,
            model_id=resolved_model,
            device=device_str,
            hf_token=self.hf_token,
        )
        # Keep a handle for code that still expects the raw pyannote Pipeline.
        self.diarization_pipeline = getattr(self._backend, "_pipeline", None)

        print(f"Initializing embedding model={self.embedding_model_id}...")
        self._extractor = build_embedding_extractor(
            self.embedding_model_id,
            device=device_str,
            hf_token=self.hf_token,
        )
        self.embedding_model = self._extractor  # duck-typed for extract_embeddings
        print("Models initialized successfully")

    def _embed_crop(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        start: float,
        end: float,
    ) -> Optional[np.ndarray]:
        """Extract an L2-normalised embedding for ``[start, end)`` seconds."""
        assert self._extractor is not None
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        if end_idx <= start_idx:
            return None
        segment_audio = waveform[:, start_idx:end_idx]
        if segment_audio.device != torch.device("cpu"):
            segment_audio = segment_audio.cpu()
        try:
            return self._extractor.extract(segment_audio, sample_rate)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not extract embedding at {start:.2f}s: {exc}")
            return None

    def _diarize_turns(self, processed_audio: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the configured backend; returns ``[{speaker, start, end}, ...]``."""
        assert self._backend is not None
        return self._backend.diarize_file(processed_audio)

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

        # Always load into memory so pyannote never calls torchcodec internally.
        # torchcodec is broken with PyTorch ≥ 2.6+cu* builds; we use soundfile/
        # librosa + scipy instead (see module-level _load_audio / _resample).
        try:
            waveform, sample_rate = _load_audio(str(audio_path))

            # Resample to 16 kHz if necessary (standard for speech processing)
            target_sr = 16000
            if sample_rate != target_sr:
                waveform = _resample(waveform, sample_rate, target_sr)
                sample_rate = target_sr
                print(f"Resampled to {target_sr}Hz")

            audio_dict = {
                'waveform': waveform,
                'sample_rate': sample_rate,
                'uri': audio_path_obj.stem,
            }
            print(f"Audio loaded into memory: {waveform.shape} at {sample_rate}Hz")
            return audio_dict

        except Exception as e:
            print(f"Warning: Could not load audio into memory: {e}")
            print("Attempting to process original file directly...")
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
                waveform, sr = _load_audio(str(path))
                if sr != TARGET_SR:
                    waveform = _resample(waveform, sr, TARGET_SR)
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
        cross_file_threshold: float = 0.55,
        prior_speaker_embeddings: Optional[Dict[str, Any]] = None,
        time_cursor: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Diarize each file independently and align speakers across files.

        No audio concatenation.  Each file is diarized on its own; local
        speaker centroids are then clustered globally (average-linkage on
        cosine distance) so the same person keeps one label across chunks.

        Args:
            audio_paths: Ordered list of audio file paths.
            cross_file_threshold: Minimum cosine similarity to merge two local
                centroids into one global speaker (0-1).  Distance threshold for
                clustering is ``1 - cross_file_threshold``.
            prior_speaker_embeddings: Optional prior centroids
                ``label → {embedding, total_duration}``.  After clustering,
                cluster means are matched to these so named speakers (e.g. Tom)
                stick across a rediarize.
            time_cursor: Seconds already processed; new timestamps are offset.

        Returns:
            ``(segments, speaker_embeddings, chunk_offsets)``.
        """
        from sklearn.cluster import AgglomerativeClustering  # noqa: PLC0415

        TARGET_SR = 16000
        all_segments: List[Dict[str, Any]] = []
        global_speaker_embeddings: Dict[str, List[Dict[str, Any]]] = {}
        chunk_offsets: List[Dict[str, Any]] = []

        def _mean_emb(emb_list: List[Dict[str, Any]]) -> np.ndarray:
            durations = np.array([e["end"] - e["start"] for e in emb_list], dtype=np.float64)
            total = float(durations.sum())
            weights = durations / total if total > 0 else np.ones(len(emb_list))
            stacked = np.stack([e["embedding"].flatten() for e in emb_list])
            mean = np.average(stacked, axis=0, weights=weights)
            norm = float(np.linalg.norm(mean))
            return mean / norm if norm > 0 else mean

        # Per-file local clusters awaiting global merge.
        # Each entry: file_idx, local_label, mean, emb_list, turns, file_offset
        pending: List[Dict[str, Any]] = []

        for file_idx, path in enumerate(audio_paths):
            print(f"  Diarizing file {file_idx + 1}/{len(audio_paths)}: {path}")

            processed_audio = self._preprocess_audio(path)
            if isinstance(processed_audio, dict):
                waveform = processed_audio["waveform"]
                sample_rate = processed_audio["sample_rate"]
            else:
                waveform, sample_rate = _load_audio(str(processed_audio))
                if sample_rate != TARGET_SR:
                    waveform = _resample(waveform, sample_rate, TARGET_SR)
                    sample_rate = TARGET_SR

            file_duration = waveform.shape[1] / sample_rate
            file_offset = time_cursor
            chunk_offsets.append(
                {
                    "path": str(path),
                    "start": file_offset,
                    "end": file_offset + file_duration,
                }
            )

            turns = self._diarize_turns(processed_audio)
            local_speaker_embeddings: Dict[str, List[Dict[str, Any]]] = {}
            for turn in turns:
                speaker = turn["speaker"]
                seg_start = float(turn["start"])
                seg_end = float(turn["end"])
                if seg_end - seg_start < 0.5:
                    continue
                embedding = self._embed_crop(waveform, sample_rate, seg_start, seg_end)
                if embedding is None:
                    continue
                local_speaker_embeddings.setdefault(speaker, []).append(
                    {
                        "embedding": embedding,
                        "start": seg_start,
                        "end": seg_end,
                    }
                )

            for local_label, emb_list in local_speaker_embeddings.items():
                pending.append(
                    {
                        "file_idx": file_idx,
                        "local_label": local_label,
                        "mean": _mean_emb(emb_list),
                        "emb_list": emb_list,
                        "turns": [
                            t
                            for t in turns
                            if t["speaker"] == local_label
                            and float(t["end"]) - float(t["start"]) >= 0.5
                        ],
                        "file_offset": file_offset,
                        "source_file": str(path),
                    }
                )

            # Turns with no usable embedding still need a provisional label later.
            embedded_locals = set(local_speaker_embeddings)
            orphan_turns: Dict[str, List[Dict[str, Any]]] = {}
            for turn in turns:
                if turn["speaker"] in embedded_locals:
                    continue
                if float(turn["end"]) - float(turn["start"]) < 0.5:
                    continue
                orphan_turns.setdefault(turn["speaker"], []).append(turn)
            for local_label, turn_list in orphan_turns.items():
                pending.append(
                    {
                        "file_idx": file_idx,
                        "local_label": local_label,
                        "mean": None,
                        "emb_list": [],
                        "turns": turn_list,
                        "file_offset": file_offset,
                        "source_file": str(path),
                    }
                )

            time_cursor += file_duration
            del waveform

        if not pending:
            return [], {}, chunk_offsets

        # ------------------------------------------------------------------
        # Global clustering of embedded locals (agglomerative on cosine)
        # ------------------------------------------------------------------
        embedded = [p for p in pending if p["mean"] is not None]
        unembedded = [p for p in pending if p["mean"] is None]

        def _entry_duration(entry: Dict[str, Any]) -> float:
            return float(
                sum(float(e["end"]) - float(e["start"]) for e in entry["emb_list"])
            )

        # Short/noisy chunk-locals often refuse to merge; cluster on longer
        # voices first, then attach short ones to the nearest centroid.
        min_seed_sec = 2.0
        seeds = [p for p in embedded if _entry_duration(p) >= min_seed_sec]
        shorts = [p for p in embedded if _entry_duration(p) < min_seed_sec]
        if not seeds:
            seeds, shorts = embedded, []

        cluster_ids = np.zeros(len(seeds), dtype=np.int32)
        if len(seeds) == 1:
            cluster_ids[0] = 0
        elif len(seeds) > 1:
            stacked = np.stack([p["mean"] for p in seeds])
            distance_threshold = max(0.0, min(1.0, 1.0 - float(cross_file_threshold)))
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=distance_threshold,
            )
            cluster_ids = clustering.fit_predict(stacked)

        unique_clusters = sorted({int(c) for c in cluster_ids})
        cluster_emb_lists: Dict[int, List[Dict[str, Any]]] = {}
        for idx, entry in enumerate(seeds):
            cid = int(cluster_ids[idx])
            cluster_emb_lists.setdefault(cid, []).extend(entry["emb_list"])
        cluster_means: Dict[int, np.ndarray] = {
            cid: _mean_emb(emb_list) for cid, emb_list in cluster_emb_lists.items()
        }

        # Attach short locals to the nearest existing cluster when similar enough;
        # otherwise mint a new cluster (keeps rare but real speakers).
        next_cid = (max(unique_clusters) + 1) if unique_clusters else 0
        short_cluster_of: Dict[int, int] = {}
        for entry in shorts:
            mean = entry["mean"]
            best_cid, best_sim = None, -1.0
            for cid, cmean in cluster_means.items():
                if cmean.shape != mean.shape:
                    continue
                sim = float(np.dot(mean.flatten(), cmean.flatten()))
                if sim > best_sim:
                    best_sim, best_cid = sim, cid
            if best_cid is not None and best_sim >= cross_file_threshold:
                short_cluster_of[id(entry)] = best_cid
                cluster_emb_lists[best_cid].extend(entry["emb_list"])
                cluster_means[best_cid] = _mean_emb(cluster_emb_lists[best_cid])
            else:
                short_cluster_of[id(entry)] = next_cid
                cluster_emb_lists[next_cid] = list(entry["emb_list"])
                cluster_means[next_cid] = mean
                unique_clusters.append(next_cid)
                next_cid += 1

        # Second pass: merge cluster centroids that are still similar.
        merged_into: Dict[int, int] = {cid: cid for cid in unique_clusters}
        changed = True
        while changed:
            changed = False
            alive = sorted({merged_into[c] for c in unique_clusters})
            best_pair = None
            best_sim = -1.0
            for i, a in enumerate(alive):
                for b in alive[i + 1 :]:
                    ma, mb = cluster_means[a], cluster_means[b]
                    if ma.shape != mb.shape:
                        continue
                    sim = float(np.dot(ma.flatten(), mb.flatten()))
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (a, b)
            if best_pair is not None and best_sim >= cross_file_threshold:
                a, b = best_pair
                keep, drop = (a, b) if a < b else (b, a)
                cluster_emb_lists[keep].extend(cluster_emb_lists.get(drop, []))
                cluster_means[keep] = _mean_emb(cluster_emb_lists[keep])
                for cid in list(merged_into):
                    if merged_into[cid] == drop:
                        merged_into[cid] = keep
                changed = True

        def _final_cid(raw_cid: int) -> int:
            return merged_into.get(raw_cid, raw_cid)

        # Rebuild means after merges.
        final_ids = sorted({_final_cid(c) for c in unique_clusters})
        final_emb_lists: Dict[int, List[Dict[str, Any]]] = {cid: [] for cid in final_ids}
        for idx, entry in enumerate(seeds):
            final_emb_lists[_final_cid(int(cluster_ids[idx]))].extend(entry["emb_list"])
        for entry in shorts:
            final_emb_lists[_final_cid(short_cluster_of[id(entry)])].extend(
                entry["emb_list"]
            )
        final_means = {
            cid: _mean_emb(lst) for cid, lst in final_emb_lists.items() if lst
        }

        # Fold singleton / tiny clusters into the nearest major voice.  These are
        # usually one noisy chunk-local that failed to merge at the main threshold.
        local_counts: Dict[int, int] = {cid: 0 for cid in final_means}
        for idx, _entry in enumerate(seeds):
            local_counts[_final_cid(int(cluster_ids[idx]))] = (
                local_counts.get(_final_cid(int(cluster_ids[idx])), 0) + 1
            )
        for entry in shorts:
            cid = _final_cid(short_cluster_of[id(entry)])
            local_counts[cid] = local_counts.get(cid, 0) + 1

        def _cid_duration(cid: int) -> float:
            return float(
                sum(
                    float(e["end"]) - float(e["start"])
                    for e in final_emb_lists.get(cid, [])
                )
            )

        majors = [
            cid
            for cid in final_means
            if local_counts.get(cid, 0) >= 2 or _cid_duration(cid) >= 5.0
        ]
        tinies = [cid for cid in final_means if cid not in majors]
        absorb_thr = min(float(cross_file_threshold), 0.40)
        if majors and tinies:
            for tiny in list(tinies):
                tmean = final_means[tiny]
                best_major, best_sim = None, -1.0
                for major in majors:
                    mmean = final_means[major]
                    if mmean.shape != tmean.shape:
                        continue
                    sim = float(np.dot(tmean.flatten(), mmean.flatten()))
                    if sim > best_sim:
                        best_sim, best_major = sim, major
                if best_major is not None and best_sim >= absorb_thr:
                    print(
                        f"    absorb cluster {tiny} → cluster {best_major} "
                        f"(singleton/tiny, similarity={best_sim:.3f})"
                    )
                    for cid in list(merged_into):
                        if _final_cid(cid) == tiny:
                            merged_into[cid] = best_major
                    final_emb_lists[best_major].extend(final_emb_lists.get(tiny, []))
                    final_means[best_major] = _mean_emb(final_emb_lists[best_major])
                    final_emb_lists.pop(tiny, None)
                    final_means.pop(tiny, None)
                    local_counts[best_major] = local_counts.get(best_major, 0) + local_counts.get(
                        tiny, 0
                    )
                    local_counts.pop(tiny, None)

        # Match only *named* priors (Tom). Anonymous SPEAKER_XX priors from a
        # previous fragmented rediarize must not pin new clusters to old junk.
        named_prior = {
            label: info
            for label, info in (prior_speaker_embeddings or {}).items()
            if not is_anonymous_speaker_label(str(label))
        }

        def _prior_vec(info: Any) -> Optional[np.ndarray]:
            if not isinstance(info, dict):
                return None
            emb = info.get("embedding")
            if emb is None:
                return None
            vec = np.asarray(emb, dtype=np.float32).flatten()
            norm = float(np.linalg.norm(vec))
            return vec / norm if norm > 0 else None

        cluster_to_global: Dict[int, str] = {}
        used_prior: set = set()
        next_speaker_num = 0

        for cid in sorted(final_means):
            mean = final_means[cid]
            best_label = None
            best_sim = -1.0
            for label, info in named_prior.items():
                if label in used_prior:
                    continue
                pvec = _prior_vec(info)
                if pvec is None or pvec.shape != mean.shape:
                    continue
                sim = float(np.dot(mean.flatten(), pvec.flatten()))
                if sim > best_sim:
                    best_sim = sim
                    best_label = label
            n_locals = sum(
                1
                for idx, entry in enumerate(seeds)
                if _final_cid(int(cluster_ids[idx])) == cid
            ) + sum(
                1
                for entry in shorts
                if _final_cid(short_cluster_of[id(entry)]) == cid
            )
            if best_label is not None and best_sim >= cross_file_threshold:
                cluster_to_global[cid] = best_label
                used_prior.add(best_label)
                print(
                    f"    cluster {cid} → {best_label} "
                    f"(named prior, similarity={best_sim:.3f}, "
                    f"{n_locals} chunk-local voice(s))"
                )
            else:
                global_label = f"SPEAKER_{next_speaker_num:02d}"
                next_speaker_num += 1
                cluster_to_global[cid] = global_label
                print(
                    f"    cluster {cid} → {global_label} "
                    f"({n_locals} chunk-local voice(s))"
                )

        local_to_global_by_file: Dict[Tuple[int, str], str] = {}

        def _assign_entry(entry: Dict[str, Any], raw_cid: int) -> None:
            global_label = cluster_to_global[_final_cid(raw_cid)]
            key = (entry["file_idx"], entry["local_label"])
            local_to_global_by_file[key] = global_label
            global_speaker_embeddings.setdefault(global_label, []).extend(
                entry["emb_list"]
            )
            for turn in entry["turns"]:
                seg_start = float(turn["start"])
                seg_end = float(turn["end"])
                all_segments.append(
                    {
                        "speaker": global_label,
                        "original_label": entry["local_label"],
                        "start": seg_start + entry["file_offset"],
                        "end": seg_end + entry["file_offset"],
                        "duration": seg_end - seg_start,
                        "source_file": entry["source_file"],
                    }
                )

        for idx, entry in enumerate(seeds):
            _assign_entry(entry, int(cluster_ids[idx]))
        for entry in shorts:
            _assign_entry(entry, short_cluster_of[id(entry)])

        # Unembedded turns: map via same-file local label if that local was clustered.
        for entry in unembedded:
            key = (entry["file_idx"], entry["local_label"])
            global_label = local_to_global_by_file.get(key)
            if global_label is None:
                global_label = f"SPEAKER_{next_speaker_num:02d}"
                next_speaker_num += 1
                local_to_global_by_file[key] = global_label
                print(
                    f"    {entry['local_label']}@file{entry['file_idx']} → "
                    f"{global_label} (no embedding; isolated)"
                )
            for turn in entry["turns"]:
                seg_start = float(turn["start"])
                seg_end = float(turn["end"])
                all_segments.append(
                    {
                        "speaker": global_label,
                        "original_label": entry["local_label"],
                        "start": seg_start + entry["file_offset"],
                        "end": seg_end + entry["file_offset"],
                        "duration": seg_end - seg_start,
                        "source_file": entry["source_file"],
                    }
                )

        n_globals = len(global_speaker_embeddings)
        print(
            f"  Cross-file clustering: {len(embedded)} local voice(s) → "
            f"{n_globals} global speaker(s) "
            f"(threshold={cross_file_threshold:.2f})"
        )

        all_segments.sort(key=lambda x: x["start"])
        return all_segments, global_speaker_embeddings, chunk_offsets

    def diarize(
        self,
        audio_path: Union[str, List[str]],
        db_dsn: Optional[str] = None,
        similarity_threshold: float = 0.7,
        store_new_speakers: bool = False,
        cross_file_threshold: float = 0.55,
        prior_speaker_embeddings: Optional[Dict[str, Any]] = None,
        time_cursor: float = 0.0,
        source_map: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
        identify_margin: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform speaker diarization with curated gallery identification.

        Supports incremental session processing: pass ``prior_speaker_embeddings``
        and ``time_cursor`` from a saved session to recognise speakers across
        separate CLI invocations without re-processing old audio.

        Each new file is diarized independently and speaker labels are aligned
        using embedding similarity.  No audio concatenation is created.

        Args:
            audio_path: Path to audio file, or ordered list of paths.
            db_dsn: PostgreSQL DSN (None to skip gallery lookup).
            similarity_threshold: Minimum cosine similarity to accept a gallery hit.
            store_new_speakers: Deprecated / ignored.  Gallery enrollments are
                never written at runtime — use ``speakers enroll`` instead.
            cross_file_threshold: Cosine-similarity threshold for aligning
                speaker labels across files (0-1). Default 0.55.
            prior_speaker_embeddings: Per-speaker state from a previous session.
                Only **named** priors (not ``SPEAKER_XX``) are used to relabel clusters.
            time_cursor: Seconds of audio already processed in previous calls.
            source_map: local temp path → canonical S3 URI map.
            session_id: When set, gallery hits are written to session_speaker_map.
            identify_margin: Override for best-vs-second margin (default: engine).

        Returns:
            Dictionary containing speakers, segments, matched/new speakers,
            session_speaker_embeddings, and new_time_cursor.
        """
        from .speaker_gallery import SpeakerGallery, duration_weighted_mean
        from pawn_core.config import SpeakersConfig

        if store_new_speakers:
            print(
                "Note: --store-new is ignored.  Enroll speakers explicitly via "
                "`pawn-diarize speakers enroll` (gallery is curated, not auto-filled)."
            )

        self._initialize_models()
        margin = (
            self.identify_margin if identify_margin is None else float(identify_margin)
        )

        audio_paths: List[str] = (
            [audio_path] if isinstance(audio_path, str) else list(audio_path)
        )

        # ------------------------------------------------------------------
        # STEP 1: Diarize and collect segments + per-speaker embeddings
        # ------------------------------------------------------------------
        segments: List[Dict[str, Any]] = []
        speaker_embeddings: Dict[str, List[Dict[str, Any]]] = {}
        chunk_offsets: Optional[List[Dict[str, Any]]] = None
        file_duration: float = 0.0
        single_file_turns: List[Dict[str, Any]] = []

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
            if source_map:
                for seg in segments:
                    if "source_file" in seg:
                        seg["source_file"] = source_map.get(
                            seg["source_file"], seg["source_file"]
                        )
            speakers: set = {seg["speaker"] for seg in segments}
        else:
            processed_audio = self._preprocess_audio(audio_paths[0])

            if isinstance(processed_audio, dict):
                print(f"Diarizing in-memory audio from: {audio_paths[0]}")
                waveform = processed_audio["waveform"]
                sample_rate = processed_audio["sample_rate"]
            else:
                print(f"Diarizing: {processed_audio}")
                waveform, sample_rate = _load_audio(str(processed_audio))

            file_duration = waveform.shape[1] / sample_rate
            single_file_turns = self._diarize_turns(processed_audio)

            speakers = set()
            print("Extracting speaker embeddings...")
            for turn in single_file_turns:
                speaker = turn["speaker"]
                start_time = float(turn["start"])
                end_time = float(turn["end"])
                if end_time - start_time < 0.5:
                    continue
                embedding = self._embed_crop(waveform, sample_rate, start_time, end_time)
                if embedding is not None:
                    speaker_embeddings.setdefault(speaker, []).append({
                        "embedding": embedding,
                        "start": start_time,
                        "end": end_time,
                    })
                speakers.add(speaker)

        # ------------------------------------------------------------------
        # STEP 2: Identify against the curated Speakers gallery (no auto-store)
        # ------------------------------------------------------------------
        matched_speakers: Dict[str, str] = {}
        matched_speaker_ids: Dict[str, str] = {}
        matched_scores: Dict[str, float] = {}
        new_speakers: List[str] = []

        gallery: Optional[SpeakerGallery] = None
        if db_dsn:
            try:
                gallery = SpeakerGallery(
                    db_dsn,
                    config=SpeakersConfig(
                        identify_threshold=similarity_threshold,
                        identify_margin=margin,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: Could not open Speakers gallery: {exc}")
                gallery = None

        if gallery is not None:
            enrollment_count = gallery.count_enrollments()
            if enrollment_count == 0:
                print(
                    "Matching speakers against curated gallery... "
                    "(gallery has 0 enrollments — all labels stay anonymous; "
                    "use `speakers create` + `speakers enroll` for named hits)"
                )
            else:
                print(
                    f"Matching speakers against curated gallery "
                    f"({enrollment_count} enrollment(s))..."
                )
            compat_warned = False
            for speaker_label, emb_list in speaker_embeddings.items():
                probe = duration_weighted_mean(emb_list)
                if probe is None:
                    new_speakers.append(speaker_label)
                    continue
                probe_model = getattr(self._extractor, "model_id", None)
                hit = gallery.identify(
                    probe,
                    embedding_model=probe_model,
                    threshold=similarity_threshold,
                    margin=margin,
                )
                if not hit.accepted:
                    # Allow cross-model gallery hits after an embedding upgrade.
                    hit = gallery.identify(
                        probe,
                        embedding_model=None,
                        threshold=similarity_threshold,
                        margin=margin,
                    )
                if hit.accepted and hit.display_name:
                    print(
                        f"  ✓ Matched {speaker_label} → '{hit.display_name}' "
                        f"(score={hit.score:.3f}, "
                        f"margin={hit.score - hit.second_score:.3f})"
                    )
                    matched_speakers[speaker_label] = hit.display_name
                    if hit.speaker_id:
                        matched_speaker_ids[speaker_label] = hit.speaker_id
                    matched_scores[speaker_label] = hit.score
                    if session_id:
                        gallery.upsert_session_map(
                            session_id,
                            speaker_label,
                            speaker_id=hit.speaker_id,
                            display_name=hit.display_name,
                            match_score=hit.score,
                            match_method="gallery",
                        )
                else:
                    if hit.score == 0.0 and hit.second_score == 0.0:
                        if enrollment_count == 0:
                            print(f"  · {speaker_label} unmatched (empty gallery)")
                        else:
                            if not compat_warned:
                                info = gallery.enrollment_compatibility(
                                    probe, embedding_model=probe_model
                                )
                                print(
                                    "  Note: gallery enrollments are incompatible with "
                                    f"runtime embeddings (probe dim={info['probe_dim']}, "
                                    f"model={info['probe_model']!r}; "
                                    f"enrollments dims={dict(info['enrollment_dims'])}, "
                                    f"models={dict(info['enrollment_models'])}). "
                                    "Re-enroll or run `speakers reembed`."
                                )
                                compat_warned = True
                            print(
                                f"  · {speaker_label} unmatched "
                                f"(no compatible enrollments for this embedding dim/model)"
                            )
                    else:
                        print(
                            f"  · {speaker_label} unmatched "
                            f"(best={hit.score:.3f}, second={hit.second_score:.3f})"
                        )
                    new_speakers.append(speaker_label)
        else:
            new_speakers = list(speaker_embeddings.keys())

        # ------------------------------------------------------------------
        # STEP 3: Build final segment list (apply display names)
        # ------------------------------------------------------------------
        if use_multi_file:
            for seg in segments:
                seg["speaker"] = matched_speakers.get(seg["speaker"], seg["speaker"])
        else:
            canonical_path0 = (
                source_map.get(str(audio_paths[0]), str(audio_paths[0]))
                if source_map
                else str(audio_paths[0])
            )
            for turn in single_file_turns:
                start_t = float(turn["start"])
                end_t = float(turn["end"])
                if end_t - start_t < 0.5:
                    continue
                speaker = turn["speaker"]
                display_name = matched_speakers.get(speaker, speaker)
                segments.append({
                    "speaker": display_name,
                    "original_label": speaker,
                    "source_file": canonical_path0,
                    "start": start_t + time_cursor,
                    "end": end_t + time_cursor,
                    "duration": end_t - start_t,
                })
            segments.sort(key=lambda x: x["start"])

        final_speakers = sorted({matched_speakers.get(s, s) for s in speakers})

        # In-session centroids only — never promoted to the gallery automatically.
        session_speaker_embeddings: Dict[str, Any] = {}
        for label, emb_list in speaker_embeddings.items():
            if not emb_list:
                continue
            durations = np.array([e["end"] - e["start"] for e in emb_list])
            weights = durations / durations.sum() if durations.sum() > 0 else durations
            stacked = np.stack([e["embedding"].flatten() for e in emb_list])
            mean_emb = np.average(stacked, axis=0, weights=weights)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            display_label = matched_speakers.get(label, label)
            session_speaker_embeddings[display_label] = {
                "embedding": mean_emb.tolist(),
                "total_duration": float(durations.sum()),
            }

        if chunk_offsets:
            new_time_cursor = chunk_offsets[-1]["end"]
        else:
            new_time_cursor = time_cursor + file_duration

        result: Dict[str, Any] = {
            "speakers": final_speakers,
            "segments": segments,
            "num_speakers": len(speakers),
            "matched_speakers": matched_speakers,
            "matched_speaker_ids": matched_speaker_ids,
            "matched_scores": matched_scores,
            "new_speakers": new_speakers,
            "session_speaker_embeddings": session_speaker_embeddings,
            "new_time_cursor": new_time_cursor,
            "embedding_model": getattr(
                self._extractor, "model_id", self.embedding_model_id
            ),
        }
        if chunk_offsets is not None:
            result["chunk_offsets"] = chunk_offsets
        return result

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
        assert self._extractor is not None

        audio_paths: List[str] = (
            [audio_path] if isinstance(audio_path, str) else list(audio_path)
        )

        if len(audio_paths) == 1:
            waveform, sample_rate = _load_audio(audio_paths[0])
            target_sr = 16000
            if sample_rate != target_sr:
                waveform = _resample(waveform, sample_rate, target_sr)
                sample_rate = target_sr
            return self._extractor.extract(waveform, sample_rate)

        print(f"Extracting embeddings from {len(audio_paths)} files (per-file average)…")
        embeddings_list: List[np.ndarray] = []
        durations: List[float] = []

        for path in audio_paths:
            waveform, sample_rate = _load_audio(str(path))
            target_sr = 16000
            if sample_rate != target_sr:
                waveform = _resample(waveform, sample_rate, target_sr)
                sample_rate = target_sr
            emb = self._extractor.extract(waveform, sample_rate)
            embeddings_list.append(emb.flatten())
            durations.append(waveform.shape[1] / sample_rate)
            del waveform

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

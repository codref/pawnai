"""Pluggable anonymous diarization backends ("who spoke when").

Backends return a list of turns ``{speaker, start, end}`` with local labels
like ``SPEAKER_00``.  Identity matching against the curated gallery happens
*after* this step in :class:`~pawn_diarize.core.diarization.DiarizationEngine`.

Supported backends
------------------
pyannote
    ``pyannote/speaker-diarization-community-1`` (default).  Prefers the
    pipeline's ``exclusive_speaker_diarization`` when present — one active
    speaker at a time, which aligns cleanly with ASR word timestamps.
nemotron
    Prefers ``nvidia/Nemotron-3-Diarization`` (Sortformer, up to 8 speakers).
    That checkpoint needs NeMo Speech with RoPE in ``TransformerEncoder``
    (newer than PyPI ``nemo-toolkit==3.0.0``).  Without RoPE we fall back to
    ``nvidia/diar_sortformer_4spk-v1``, which loads on stock 3.0.0.
"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch


# One diarization turn: anonymous local label + time span in seconds.
Turn = Dict[str, Any]  # keys: speaker (str), start (float), end (float)

DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_NEMOTRON_MODEL = "nvidia/Nemotron-3-Diarization"
# Works on PyPI nemo-toolkit 3.0.0 (no RoPE). Used when Nemotron-3 cannot load.
FALLBACK_SORTFORMER_MODEL = "nvidia/diar_sortformer_4spk-v1"


class DiarizationBackend(ABC):
    """Produce anonymous SPEAKER_XX turns for one audio input."""

    name: str

    @abstractmethod
    def diarize_file(
        self,
        audio: Union[str, Dict[str, Any]],
    ) -> List[Turn]:
        """Diarize a file path or in-memory ``{waveform, sample_rate, uri}`` dict."""


class PyannoteBackend(DiarizationBackend):
    """pyannote.audio Pipeline wrapper."""

    name = "pyannote"

    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-community-1",
        device: Optional[torch.device] = None,
        hf_token: Optional[str] = None,
        prefer_exclusive: bool = True,
    ) -> None:
        from pyannote.audio import Pipeline  # noqa: PLC0415

        self.model_id = model_id
        self.prefer_exclusive = prefer_exclusive
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._pipeline = Pipeline.from_pretrained(model_id, token=hf_token).to(device)

    def diarize_file(self, audio: Union[str, Dict[str, Any]]) -> List[Turn]:
        output = self._pipeline(audio)
        # community-1 exposes exclusive_speaker_diarization for STT alignment.
        annotation = None
        if self.prefer_exclusive and hasattr(output, "exclusive_speaker_diarization"):
            annotation = output.exclusive_speaker_diarization
        if annotation is None and hasattr(output, "speaker_diarization"):
            annotation = output.speaker_diarization
        if annotation is None:
            # Older pyannote returned the annotation directly.
            annotation = output

        turns: List[Turn] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                {
                    "speaker": str(speaker),
                    "start": float(turn.start),
                    "end": float(turn.end),
                }
            )
        return turns


class NemotronBackend(DiarizationBackend):
    """NVIDIA Sortformer / Nemotron diarization wrapper."""

    name = "nemotron"

    def __init__(
        self,
        model_id: str = DEFAULT_NEMOTRON_MODEL,
        device: Optional[torch.device] = None,
    ) -> None:
        from nemo.collections.asr.models import SortformerEncLabelModel  # noqa: PLC0415

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = model_id
        self.model_id = resolve_nemotron_runtime_model_id(model_id)
        self._device = device
        try:
            self._model = SortformerEncLabelModel.from_pretrained(self.model_id)
        except Exception as exc:
            if (
                self.model_id != FALLBACK_SORTFORMER_MODEL
                and _is_rope_unsupported_error(exc)
            ):
                print(
                    f"Note: loading {self.model_id!r} failed (NeMo lacks RoPE "
                    f"support); falling back to {FALLBACK_SORTFORMER_MODEL!r}.\n"
                    f"  Install NeMo Speech from GitHub for Nemotron-3:\n"
                    f"    uv pip install "
                    f"'nemo-toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git'"
                )
                self.model_id = FALLBACK_SORTFORMER_MODEL
                self._model = SortformerEncLabelModel.from_pretrained(self.model_id)
            else:
                raise
        self._model = self._model.to(device)
        self._model.eval()
        _configure_sortformer_offline(self._model)
        if self.model_id != requested:
            print(
                f"Using Sortformer model={self.model_id} "
                f"(requested {requested})"
            )

    def diarize_file(self, audio: Union[str, Dict[str, Any]]) -> List[Turn]:
        import os
        import tempfile

        import soundfile as sf  # noqa: PLC0415

        tmp_path: Optional[str] = None
        try:
            if isinstance(audio, dict):
                waveform = audio["waveform"]
                sample_rate = int(audio["sample_rate"])
                if hasattr(waveform, "numpy"):
                    data = waveform.numpy()
                else:
                    data = waveform
                if getattr(data, "ndim", 1) > 1:
                    data = data.mean(axis=0)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp_path = tmp.name
                tmp.close()
                sf.write(tmp_path, data, sample_rate, subtype="PCM_16")
                path = tmp_path
            else:
                path = str(audio)

            # Prefer diarize() when available; fall back to transcribe API.
            # Pass a bare path (not [path]): NeMo wraps singles, and for a
            # one-file batch ``results.extend`` flattens to List[str] lines.
            # quiet_nemo_loggers: ASR's _quiet_nemo may have left NeMo handlers
            # bound to a closed /dev/null; rebind + raise level so Lhotse
            # "ignored keys" warnings don't spam Logging error traces.
            from pawn_core.transcription import (  # noqa: PLC0415
                _rebind_nemo_stream_handlers,
                quiet_nemo_loggers,
            )

            _rebind_nemo_stream_handlers()
            with quiet_nemo_loggers():
                if hasattr(self._model, "diarize"):
                    raw = self._model.diarize(audio=path, batch_size=1)
                else:
                    raw = self._model.transcribe([path], batch_size=1)

            turns = _normalize_nemotron_output(raw)
            if not turns:
                # NeMo returns [[]] (or []) when postprocessing finds no speech.
                emptyish = (
                    raw in ([], [[]])
                    or (
                        isinstance(raw, list)
                        and len(raw) == 1
                        and isinstance(raw[0], list)
                        and len(raw[0]) == 0
                    )
                )
                if emptyish:
                    print(f"    Note: no speech detected in {path}")
                else:
                    print(
                        f"    Warning: NeMo returned no parseable turns for {path} "
                        f"(raw type={type(raw).__name__}, "
                        f"sample={_raw_preview(raw)!r})"
                    )
            return turns
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def _raw_preview(raw: Any, limit: int = 120) -> str:
    text = repr(raw)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _parse_one_nemotron_turn(item: Any) -> Optional[Turn]:
    """Parse one NeMo/Sortformer turn into ``{speaker, start, end}``."""
    if item is None:
        return None

    if isinstance(item, dict):
        spk = item.get("speaker", item.get("label", 0))
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
        except (TypeError, ValueError):
            return None
        if end <= start:
            return None
        return {"speaker": _speaker_label(spk), "start": start, "end": end}

    if isinstance(item, (list, tuple)) and len(item) >= 3:
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            return None
        if end <= start:
            return None
        return {
            "speaker": _speaker_label(item[2]),
            "start": start,
            "end": end,
        }

    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        # Docstring-style: "[begin_seconds, end_seconds, speaker_index]"
        if text.startswith("[") and text.endswith("]"):
            try:
                return _parse_one_nemotron_turn(ast.literal_eval(text))
            except (SyntaxError, ValueError):
                return None
        # generate_diarization_output_lines: "start end speaker_N"
        parts = text.split()
        if len(parts) >= 3:
            try:
                start, end = float(parts[0]), float(parts[1])
            except ValueError:
                return None
            if end <= start:
                return None
            return {
                "speaker": _speaker_label(parts[2]),
                "start": start,
                "end": end,
            }
    return None


def _normalize_nemotron_output(raw: Any) -> List[Turn]:
    """Normalise NeMo Sortformer / Nemotron ``diarize()`` outputs.

    Observed shapes (NeMo Speech / Sortformer):
    - Flat ``List[str]`` after one-file ``results.extend``:
      ``["0.50 3.12 speaker_0", ...]`` (from ``generate_diarization_output_lines``)
    - ``List[List[str]]`` for a multi-file batch before flatten
    - ``List[List[float]]`` triples ``[[start, end, spk], ...]``
    - Optional ``(lines, tensors)`` when ``include_tensor_outputs=True``
    """
    if raw is None:
        return []

    # (lines, tensors)
    if isinstance(raw, tuple) and raw:
        raw = raw[0]

    items: List[Any]
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        first = raw[0]
        # [[start, end, spk], ...] — do NOT unwrap the triple itself
        if first and isinstance(first[0], (int, float)):
            items = raw
        else:
            # [[lines_file0...], [lines_file1...]] or single-file wrapper
            items = []
            for file_items in raw:
                if isinstance(file_items, list):
                    items.extend(file_items)
                else:
                    items.append(file_items)
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]

    turns: List[Turn] = []
    for item in items:
        turn = _parse_one_nemotron_turn(item)
        if turn is not None:
            turns.append(turn)

    # Hypotheses with .speaker_timestamps (list of per-speaker [start,end] spans)
    if not turns and hasattr(raw, "speaker_timestamps"):
        for spk_idx, spans in enumerate(raw.speaker_timestamps):
            for span in spans:
                if isinstance(span, (list, tuple)) and len(span) >= 2:
                    start, end = float(span[0]), float(span[1])
                    if end > start:
                        turns.append(
                            {
                                "speaker": _speaker_label(spk_idx),
                                "start": start,
                                "end": end,
                            }
                        )
    return turns


_SPEAKER_NUM_RE = re.compile(r"^(?:speaker[_-]?)?(\d+)$", re.IGNORECASE)


def _speaker_label(spk: Any) -> str:
    """Normalise a speaker index/name to ``SPEAKER_XX``."""
    text = str(spk).strip()
    if text.startswith("SPEAKER_"):
        return text
    match = _SPEAKER_NUM_RE.match(text)
    if match:
        return f"SPEAKER_{int(match.group(1)):02d}"
    try:
        idx = int(spk)
        return f"SPEAKER_{idx:02d}"
    except (TypeError, ValueError):
        return f"SPEAKER_{text}"


def nemo_supports_rope() -> bool:
    """True when this NeMo build's TransformerEncoder accepts ``rope``."""
    try:
        from nemo.collections.asr.modules import transformer_encoder as te  # noqa: PLC0415

        supported = getattr(te, "_SUPPORTED_SELF_ATTENTION_MODELS", ())
        return "rope" in supported
    except Exception:
        return False


def _needs_rope(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return "nemotron-3" in mid or mid.endswith("/nemotron-3-diarization")


def _is_rope_unsupported_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "rope" in text and (
        "not supported" in text or "self_attention_model" in text
    )


def resolve_nemotron_runtime_model_id(model_id: str) -> str:
    """Swap Nemotron-3 for Sortformer 4spk when this NeMo lacks RoPE.

    PyPI ``nemo-toolkit==3.0.0`` only allows abs_pos/rel_pos/no_pos, so loading
    ``nvidia/Nemotron-3-Diarization`` fails during encoder instantiate.  GitHub
    ``NVIDIA-NeMo/Speech`` main adds RoPE.
    """
    mid = (model_id or "").strip() or DEFAULT_NEMOTRON_MODEL
    if _needs_rope(mid) and not nemo_supports_rope():
        print(
            f"Note: {mid!r} needs NeMo TransformerEncoder RoPE support "
            f"(newer than PyPI nemo-toolkit 3.0.0); using "
            f"{FALLBACK_SORTFORMER_MODEL!r} instead.\n"
            f"  For Nemotron-3: uv pip install "
            f"'nemo-toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git'\n"
            f"  Or set models.diarization_model: {FALLBACK_SORTFORMER_MODEL}"
        )
        return FALLBACK_SORTFORMER_MODEL
    return mid


def _configure_sortformer_offline(model: Any) -> None:
    """Apply NVIDIA's recommended offline-style Sortformer streaming knobs when present."""
    modules = getattr(model, "sortformer_modules", None)
    if modules is None:
        return
    # Values from the Nemotron-3 / Sortformer offline recipe (80 ms frames).
    for attr, value in (
        ("spkcache_len", 264),
        ("fifo_len", 40),
        ("chunk_len", 340),
        ("chunk_right_context", 40),
        ("spkcache_update_period", 300),
    ):
        if hasattr(modules, attr):
            setattr(modules, attr, value)
    checker = getattr(model, "_check_streaming_parameters", None)
    if callable(checker):
        try:
            checker()
        except Exception:
            pass


def resolve_diarization_model_id(backend: str, model_id: Optional[str] = None) -> str:
    """Pick a model id that matches *backend*.

    ``models.diarization_model`` in yaml often stays on the pyannote default
    even after switching ``diarization_backend: nemotron``.  Feeding that
    pyannote id into NeMo ``SortformerEncLabelModel.from_pretrained`` downloads
    the wrong Hub repo and then fails looking for ``model_config.yaml``.

    Runtime NeMo capability (RoPE) is handled later by
    :func:`resolve_nemotron_runtime_model_id`.
    """
    name = (backend or "pyannote").lower().strip()
    mid = (model_id or "").strip()

    if name == "nemotron":
        if not mid or mid.startswith("pyannote/") or "speaker-diarization" in mid:
            if mid and mid != DEFAULT_NEMOTRON_MODEL:
                print(
                    f"Note: diarization_backend=nemotron ignores incompatible "
                    f"diarization_model={mid!r}; using {DEFAULT_NEMOTRON_MODEL!r}. "
                    f"Set models.diarization_model to a Nemotron/Sortformer id "
                    f"to silence this."
                )
            return DEFAULT_NEMOTRON_MODEL
        return mid

    # pyannote (default)
    if not mid or mid.startswith("nvidia/") or "nemotron" in mid.lower():
        if mid and mid != DEFAULT_PYANNOTE_MODEL:
            print(
                f"Note: diarization_backend=pyannote ignores incompatible "
                f"diarization_model={mid!r}; using {DEFAULT_PYANNOTE_MODEL!r}."
            )
        return DEFAULT_PYANNOTE_MODEL
    return mid


def build_diarization_backend(
    backend: str,
    *,
    model_id: str,
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> DiarizationBackend:
    """Factory used by :class:`DiarizationEngine`."""
    torch_device: Optional[torch.device] = None
    if device and device != "auto":
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = (backend or "pyannote").lower()
    resolved = resolve_diarization_model_id(name, model_id)
    if name == "nemotron":
        return NemotronBackend(
            model_id=resolved,
            device=torch_device,
        )
    return PyannoteBackend(
        model_id=resolved,
        device=torch_device,
        hf_token=hf_token,
        prefer_exclusive=True,
    )

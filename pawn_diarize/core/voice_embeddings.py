"""Voice embedding extractors for the curated Speakers gallery.

Two backends are supported:

* ``titanet`` — NeMo ``nvidia/speakerverification_en_titanet_large`` (default)
* ``pyannote`` — legacy ``pyannote/embedding`` (512-d)

Callers should go through :func:`build_embedding_extractor` so the model id
from config picks the right implementation.  Both return L2-normalised
``numpy`` vectors; dimensions differ by model, which is fine because gallery
enrollments store ``embedding_dim`` alongside the vector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of *vec* (safe for zero vectors)."""
    flat = np.asarray(vec, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(flat))
    if norm < 1e-12:
        return flat
    return flat / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (assumed already L2-normalised)."""
    return float(np.dot(np.asarray(a).flatten(), np.asarray(b).flatten()))


class EmbeddingExtractor(ABC):
    """Extract a single speaker embedding from a waveform crop."""

    model_id: str
    dim: int

    @abstractmethod
    def extract(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """Return an L2-normalised embedding for *waveform* (channels, frames)."""


class PyannoteEmbeddingExtractor(EmbeddingExtractor):
    """Legacy pyannote.audio whole-window embedding (512-d)."""

    def __init__(
        self,
        model_id: str = "pyannote/embedding",
        device: Optional[torch.device] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        from pyannote.audio import Inference, Model  # noqa: PLC0415

        self.model_id = model_id
        self.dim = 512
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._inference = Inference(
            Model.from_pretrained(model_id, token=hf_token).to(device),
            window="whole",
        )

    def extract(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        # pyannote expects mono (1, frames).
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        emb = self._inference({"waveform": waveform, "sample_rate": sample_rate})
        return _l2_normalize(np.asarray(emb, dtype=np.float32))


class TitaNetEmbeddingExtractor(EmbeddingExtractor):
    """NeMo TitaNet-Large speaker verification embeddings (~192-d)."""

    def __init__(
        self,
        model_id: str = "nvidia/speakerverification_en_titanet_large",
        device: Optional[torch.device] = None,
    ) -> None:
        import nemo.collections.asr as nemo_asr  # noqa: PLC0415

        self.model_id = model_id
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        # from_pretrained accepts both NGC short names and HF ids.
        self._model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_id
        )
        self._model = self._model.to(device)
        self._model.eval()
        # TitaNet-Large decoder emb size is 192; fall back to a probe if needed.
        self.dim = int(getattr(self._model.decoder, "emb_sizes", 192) or 192)

    def extract(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        import soundfile as sf  # noqa: PLC0415
        import tempfile
        import os

        # Mix to mono float32 at the model's expected rate (16 kHz).
        if waveform.ndim == 1:
            audio = waveform.detach().cpu().numpy()
        else:
            audio = waveform.mean(dim=0).detach().cpu().numpy()
        if sample_rate != 16000:
            from pawn_diarize.core.diarization import _resample  # noqa: PLC0415

            resampled = _resample(
                torch.from_numpy(audio).unsqueeze(0), sample_rate, 16000
            )
            audio = resampled.squeeze(0).numpy()
            sample_rate = 16000

        # NeMo's get_embedding API is file-oriented; write a short temp WAV.
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp.name
        tmp.close()
        try:
            sf.write(tmp_path, audio, sample_rate, subtype="PCM_16")
            with torch.no_grad():
                emb = self._model.get_embedding(tmp_path)
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            return _l2_normalize(np.asarray(emb, dtype=np.float32))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def is_titanet_model(model_id: str) -> bool:
    """True when *model_id* should use the NeMo TitaNet extractor."""
    mid = (model_id or "").lower()
    return "titanet" in mid or mid.startswith("nvidia/speakerverification")


def build_embedding_extractor(
    model_id: str,
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> EmbeddingExtractor:
    """Build the extractor matching *model_id*.

    Falls back to pyannote when TitaNet cannot be imported, so a missing NeMo
    speaker package does not hard-crash identification.
    """
    torch_device: Optional[torch.device] = None
    if device and device != "auto":
        torch_device = torch.device(device)
    elif device == "auto" or device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_titanet_model(model_id):
        try:
            return TitaNetEmbeddingExtractor(model_id=model_id, device=torch_device)
        except Exception as exc:  # noqa: BLE001 — surface and fall back
            print(
                f"Warning: could not load TitaNet ({exc}); "
                "falling back to pyannote/embedding"
            )
            return PyannoteEmbeddingExtractor(
                model_id="pyannote/embedding",
                device=torch_device,
                hf_token=hf_token,
            )

    return PyannoteEmbeddingExtractor(
        model_id=model_id or "pyannote/embedding",
        device=torch_device,
        hf_token=hf_token,
    )

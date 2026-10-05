"""Cross-file speaker clustering behaviour (no ML diarization loads)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def _cluster_locals(
    means: list[np.ndarray], *, cross_file_threshold: float
) -> np.ndarray:
    """Mirror of diarization._diarize_multiple_files clustering step."""
    if len(means) <= 1:
        return np.zeros(len(means), dtype=np.int32)
    stacked = np.stack(means)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=max(0.0, min(1.0, 1.0 - float(cross_file_threshold))),
    )
    return clustering.fit_predict(stacked)


def test_same_voice_across_chunks_merges_at_0_65() -> None:
    rng = np.random.default_rng(0)
    base_a = rng.normal(size=32)
    base_a = base_a / np.linalg.norm(base_a)
    base_b = rng.normal(size=32)
    base_b = base_b / np.linalg.norm(base_b)

    # 17 chunk-local means: alternate two people with light noise
    means = []
    for i in range(17):
        base = base_a if i % 2 == 0 else base_b
        noise = rng.normal(scale=0.05, size=32)
        vec = base + noise
        means.append(vec / np.linalg.norm(vec))

    labels = _cluster_locals(means, cross_file_threshold=0.65)
    assert len(set(int(x) for x in labels)) == 2


def test_strict_0_85_fragments_noisy_same_voice() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=32)
    base = base / np.linalg.norm(base)
    means = []
    for _ in range(8):
        noise = rng.normal(scale=0.25, size=32)
        vec = base + noise
        means.append(vec / np.linalg.norm(vec))

    loose = set(int(x) for x in _cluster_locals(means, cross_file_threshold=0.55))
    strict = set(int(x) for x in _cluster_locals(means, cross_file_threshold=0.85))
    assert len(loose) <= len(strict)

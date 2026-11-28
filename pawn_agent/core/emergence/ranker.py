"""Relevance ranker: logistic regression over label embeddings (numpy).

Trained on ``em_labels`` (accept = 1, reject = 0). Features are the
embedding of ``label_text(type, text, quote)``, a one-hot of the type, the
quote match score and whether an owner is named. Weights live in
``{state_dir}/emergence/ranker.json`` with cross-validated AUC; a model is
retrained when the label count changes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pawn_agent.core.emergence.types import OBSERVATION_TYPES

logger = logging.getLogger(__name__)

_L2 = 1.0
_EPOCHS = 400
_LR = 0.5
_FOLDS = 5


def _features(
    embedding: list[float], type_name: str, quote_score: Any, has_owner: Any
) -> np.ndarray:
    vec = np.asarray(embedding, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    onehot = np.array([1.0 if type_name == t else 0.0 for t in OBSERVATION_TYPES])
    try:
        score = float(quote_score) if quote_score is not None else 1.0
    except (TypeError, ValueError):
        score = 1.0
    extra = np.array([score, 1.0 if has_owner else 0.0])
    return np.concatenate([vec, onehot, extra])


def row_features(row: dict[str, Any]) -> np.ndarray:
    feats = row.get("features") or {}
    return _features(
        row["embedding"], row["type"], feats.get("quote_score"), feats.get("has_owner")
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """L2-regularized logistic regression by gradient descent; class-balanced."""
    n, d = x.shape
    w = np.zeros(d)
    b = 0.0
    pos = max(1.0, float(y.sum()))
    neg = max(1.0, float(n - y.sum()))
    weights = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    for _ in range(_EPOCHS):
        p = _sigmoid(x @ w + b)
        err = (p - y) * weights
        grad_w = x.T @ err / n + _L2 * w / n
        grad_b = float(err.mean())
        w -= _LR * grad_w
        b -= _LR * grad_b
    return w, b


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC (probability a random positive outranks a random negative)."""
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def cross_val_auc(x: np.ndarray, y: np.ndarray, folds: int = _FOLDS, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    scores = np.zeros(len(y))
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        if len(np.unique(y[train])) < 2:
            return float("nan")
        w, b = fit(x[train], y[train])
        scores[test] = _sigmoid(x[test] @ w + b)
    return auc(scores, y)


def model_path(cfg: Any) -> Path:
    from pawn_agent.core.sallm_factory import resolve_state_dir  # noqa: PLC0415

    return resolve_state_dir(cfg) / "emergence" / "ranker.json"


def train(
    cfg: Any,
    rows: Optional[list[dict[str, Any]]] = None,
    *,
    label_total: Optional[int] = None,
) -> dict[str, Any]:
    """Fit on all training labels and save. Returns the model metadata.

    ``label_total`` (all label rows, any verdict) is stored so
    :func:`ensure_fresh` can tell when to retrain.
    """
    from pawn_agent.core.emergence.labels import training_rows  # noqa: PLC0415

    emergence = getattr(cfg, "emergence", None)
    if rows is None:
        rows = training_rows(
            cfg.db_dsn, include_bulk=bool(getattr(emergence, "train_on_bulk", False))
        )
    y = np.array([1.0 if r["verdict"] == "accept" else 0.0 for r in rows])
    meta: dict[str, Any] = {
        "labels": len(rows),
        "accept": int(y.sum()),
        "reject": int(len(y) - y.sum()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "label_total": label_total,
    }
    min_labels = int(getattr(emergence, "ranker_min_labels", 30) or 30)
    if len(rows) < min_labels or meta["accept"] < 5 or meta["reject"] < 5:
        meta["status"] = "too_few_labels"
        meta["needed"] = min_labels
        _save(cfg, meta, None, None)
        return meta
    x = np.stack([row_features(r) for r in rows])
    meta["auc"] = round(cross_val_auc(x, y), 3)
    w, b = fit(x, y)
    meta["status"] = "trained"
    _save(cfg, meta, w, b)
    logger.info("emergence ranker trained: %s", meta)
    return meta


def _save(cfg: Any, meta: dict[str, Any], w: Optional[np.ndarray], b: Optional[float]) -> None:
    path = model_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(meta)
    if w is not None:
        data["w"] = [round(float(v), 6) for v in w]
        data["b"] = float(b or 0.0)
    path.write_text(json.dumps(data))


def load(cfg: Any) -> Optional[dict[str, Any]]:
    path = model_path(cfg)
    if not path.exists():
        return None
    try:
        return dict(json.loads(path.read_text()))
    except Exception as exc:
        logger.warning("emergence ranker unreadable: %s", exc)
        return None


def ensure_fresh(cfg: Any) -> Optional[dict[str, Any]]:
    """Retrain when the label count changed; return the current model."""
    from pawn_agent.core.emergence.labels import label_counts  # noqa: PLC0415

    model = load(cfg)
    try:
        total = label_counts(cfg.db_dsn)["total"]
    except Exception as exc:
        logger.debug("emergence labels unavailable: %s", exc)
        return model
    if model is None or model.get("label_total") != total:
        train(cfg, label_total=total)
        model = load(cfg)
    return model


def score(model: Optional[dict[str, Any]], obs: dict[str, Any]) -> Optional[float]:
    """Probability the user wants this observation, or None without a model."""
    if not model or "w" not in model or not obs.get("embedding"):
        return None
    fields = obs.get("fields") or {}
    has_owner = any(fields.get(k) for k in ("owner", "debtor", "asked_by"))
    x = _features(obs["embedding"], obs["type"], obs.get("quote_score"), has_owner)
    w = np.asarray(model["w"])
    if w.shape[0] != x.shape[0]:
        return None
    return round(float(_sigmoid(np.array([x @ w + float(model.get("b", 0.0))]))[0]), 3)

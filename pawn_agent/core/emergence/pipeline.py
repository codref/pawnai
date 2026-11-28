"""Observe one source: window → triage → typed extraction → verification.

``observe_source`` returns every observation (live and rejected) and,
when ``persist`` is true, stores the run. ``project_items`` turns live
actionable observations into the dicts the coworker pipeline files as Items.
"""

from __future__ import annotations

import asyncio
import copy
import difflib
import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from pawn_agent.core.coworker.extract import (
    Window,
    locate_quote,
    merge_items,
    split_annotations,
    start_seconds_at,
    window_source,
)
from pawn_agent.core.emergence import passes, prompts
from pawn_agent.core.emergence.types import ITEM_KINDS, OWNER_FIELD, enabled_types

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "phase1.v1"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)

# When one statement yields items of several kinds, the earlier kind wins.
KIND_PRIORITY: tuple[str, ...] = (
    "commitment",
    "decision",
    "block",
    "contradiction",
    "open_question",
)
# Items this far apart (seconds) are never the same statement.
_SAME_STATEMENT_SECONDS = 120.0
_SAME_STATEMENT_OVERLAP = 0.6


@dataclass
class WindowResult:
    window: Window
    id: str
    content_hash: str
    start_sec: Optional[float]
    end_sec: Optional[float]
    triage: dict[str, float]
    triage_ok: bool
    embedding: Optional[list[float]] = None


class _FewShot:
    """Window embeddings and per-type example retrieval for one run.

    Disables itself (no examples, no errors) when the label table is empty
    or unavailable, so a fresh install pays nothing.
    """

    def __init__(self, cfg: Any, settings: dict[str, Any]):
        self.cfg = cfg
        self.n_accept = settings["few_shot_accept"]
        self.n_reject = settings["few_shot_reject"]
        self.used = 0
        self.active = True
        try:
            from pawn_agent.core.emergence.labels import label_counts  # noqa: PLC0415

            self.active = label_counts(cfg.db_dsn)["total"] > 0
        except Exception as exc:
            logger.debug("emergence few-shot off: %s", exc)
            self.active = False

    async def embed_window(self, text: str) -> Optional[list[float]]:
        if not self.active:
            return None
        from pawn_agent.core.emergence.labels import embed_many  # noqa: PLC0415

        return (await asyncio.to_thread(embed_many, self.cfg, [text[:6000]]))[0]

    async def examples(
        self, type_name: str, vector: Optional[list[float]]
    ) -> Optional[dict[str, list[dict[str, str]]]]:
        if not self.active or not vector:
            return None
        from pawn_agent.core.emergence.labels import few_shot_examples  # noqa: PLC0415

        try:
            found = await asyncio.to_thread(
                few_shot_examples,
                self.cfg.db_dsn,
                type_name,
                vector,
                n_accept=self.n_accept,
                n_reject=self.n_reject,
            )
        except Exception as exc:
            logger.warning("emergence few-shot lookup failed: %s", exc)
            return None
        if found["accept"] or found["reject"]:
            self.used += 1
            return found
        return None


async def _embed_and_rank(
    cfg: Any, live: list[dict[str, Any]], settings: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Embed live observations and score them with the ranker (best effort)."""
    from pawn_agent.core.emergence import labels  # noqa: PLC0415

    texts = [labels.label_text(o["type"], o["text"], o.get("quote") or "") for o in live]
    vectors = await asyncio.to_thread(labels.embed_many, cfg, texts)
    for obs, vector in zip(live, vectors):
        obs["embedding"] = vector
    if not settings["ranker"]:
        return None
    from pawn_agent.core.emergence import ranker  # noqa: PLC0415

    try:
        model = await asyncio.to_thread(ranker.ensure_fresh, cfg)
    except Exception as exc:
        logger.warning("emergence ranker unavailable: %s", exc)
        return None
    for obs in live:
        obs["rank_score"] = ranker.score(model, obs)
    return model


def _ranker_summary(model: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not model:
        return {"status": "off"}
    return {k: model.get(k) for k in ("status", "labels", "accept", "reject", "auc") if k in model}


@dataclass
class ObserveResult:
    source_kind: str
    source_ref: str
    run_id: str
    content_hash: str
    model: str
    windows: list[WindowResult] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def live(self) -> list[dict[str, Any]]:
        return [o for o in self.observations if o["status"] == "live"]

    @property
    def rejected(self) -> list[dict[str, Any]]:
        return [o for o in self.observations if o["status"] == "rejected"]


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:32]


def model_cfg(cfg: Any) -> tuple[Any, str]:
    """``cfg`` pointed at ``emergence.fast_model`` when set, and its id."""
    emergence = getattr(cfg, "emergence", None)
    fast = str(getattr(emergence, "fast_model", "") or "").strip()
    if not fast:
        return cfg, str(getattr(cfg, "chat_model_id", "") or "")
    from pawn_agent.utils.model_catalog import apply_model_selection  # noqa: PLC0415

    effective = copy.copy(cfg)
    apply_model_selection(effective, fast)
    return effective, fast


def _settings(cfg: Any) -> dict[str, Any]:
    emergence = getattr(cfg, "emergence", None)
    return {
        "window_chars": int(getattr(emergence, "window_chars", 12000) or 12000),
        "window_overlap": int(getattr(emergence, "window_overlap", 1500) or 0),
        "threshold": float(getattr(emergence, "triage_threshold", 0.35)),
        "quote_min": float(getattr(emergence, "quote_min_score", 0.8)),
        "types": enabled_types(getattr(emergence, "types", None)),
        "few_shot": bool(getattr(emergence, "few_shot", True)),
        "few_shot_accept": int(getattr(emergence, "few_shot_accept", 3) or 0),
        "few_shot_reject": int(getattr(emergence, "few_shot_reject", 2) or 0),
        "ranker": bool(getattr(emergence, "ranker", True)),
        "always_types": {
            str(t).strip().lower() for t in (getattr(emergence, "always_types", None) or [])
        },
        # An empty list means no type thinks (unlike ``types``).
        "think_types": {
            str(t).strip().lower() for t in (getattr(emergence, "think_types", None) or [])
        },
    }


def verify(obs: dict[str, Any], window: Window, body: str, quote_min: float) -> dict[str, Any]:
    """Locate the quote; set ``quote_score``, ``start_sec`` and ``status``."""
    quote = obs.get("quote") or ""
    offset, score = locate_quote(window.text, quote)
    obs["quote_score"] = round(score, 3)
    obs["start_sec"] = (
        start_seconds_at(body, window.char_start + offset) if offset is not None else None
    )
    if not quote.strip():
        obs["status"], obs["reject_reason"] = "rejected", "no_quote"
    elif score < quote_min:
        obs["status"], obs["reject_reason"] = "rejected", "quote_not_found"
    else:
        obs["status"], obs["reject_reason"] = "live", None
    return obs


async def observe_source(
    cfg: Any,
    *,
    source_kind: str,
    source_ref: str,
    source_text: str,
    persist: bool = True,
) -> ObserveResult:
    """Run the Phase 1 passes over one source.

    Model call failures are logged and skipped; the last error is raised
    only when every call failed.
    """
    settings = _settings(cfg)
    run_cfg, model_id = model_cfg(cfg)
    result = ObserveResult(
        source_kind=source_kind,
        source_ref=source_ref,
        run_id=str(uuid.uuid4()),
        content_hash=_hash(source_text),
        model=model_id,
    )
    annotations, body = split_annotations(source_text or "")
    windows = window_source(
        body, max_chars=settings["window_chars"], overlap_chars=settings["window_overlap"]
    )
    types = settings["types"]
    started = time.monotonic()
    calls = {"triage": 0, "typed": 0, "failed": 0}
    last_exc: Optional[Exception] = None
    raw_obs: list[dict[str, Any]] = []
    shots = _FewShot(cfg, settings) if settings["few_shot"] else None

    for window in windows:
        part = prompts.part_note(window.index, len(windows), annotations)
        end = window.char_start + max(0, len(window.text) - 1)
        wres = WindowResult(
            window=window,
            id=str(uuid.uuid4()),
            content_hash=_hash(window.text),
            start_sec=start_seconds_at(body, window.char_start),
            end_sec=start_seconds_at(body, end),
            triage={},
            triage_ok=False,
        )
        result.windows.append(wres)
        if shots is not None:
            wres.embedding = await shots.embed_window(window.text)
        if not types:
            continue
        calls["triage"] += 1
        try:
            wres.triage, wres.triage_ok = await passes.triage(run_cfg, types, window.text, part)
        except passes.PassError as exc:
            calls["failed"] += 1
            last_exc = exc
            logger.error("emergence triage failed on window %d: %s", window.index + 1, exc)
            wres.triage = {t: 1.0 for t in types}
        for type_name in types:
            if (
                type_name not in settings["always_types"]
                and wres.triage.get(type_name, 0.0) < settings["threshold"]
            ):
                continue
            calls["typed"] += 1
            try:
                examples = (
                    await shots.examples(type_name, wres.embedding) if shots is not None else None
                )
                found = await passes.extract_type(
                    run_cfg,
                    type_name,
                    window.text,
                    part,
                    think=type_name in settings["think_types"],
                    examples=examples,
                )
            except passes.PassError as exc:
                calls["failed"] += 1
                last_exc = exc
                logger.error(
                    "emergence %s failed on window %d: %s", type_name, window.index + 1, exc
                )
                continue
            for obs in found:
                obs["id"] = str(uuid.uuid4())
                obs["window"] = window.index
                obs["window_id"] = wres.id
                obs["model"] = model_id
                raw_obs.append(verify(obs, window, body, settings["quote_min"]))

    total_calls = calls["triage"] + calls["typed"]
    if total_calls and calls["failed"] == total_calls and last_exc is not None:
        raise last_exc

    live = merge_items([o for o in raw_obs if o["status"] == "live"], kind_key="type")
    rejected = [o for o in raw_obs if o["status"] == "rejected"]
    result.observations = live + rejected
    ranker_meta = await _embed_and_rank(cfg, live, settings) if live else None
    skipped = sum(
        1 for w in result.windows if w.triage and max(w.triage.values()) < settings["threshold"]
    )
    result.stats = {
        "windows": len(result.windows),
        "windows_skipped": skipped,
        "triage_calls": calls["triage"],
        "typed_calls": calls["typed"],
        "failed_calls": calls["failed"],
        "raw": len(raw_obs),
        "live": len(live),
        "rejected": len(rejected),
        "duplicates_merged": len(raw_obs) - len(rejected) - len(live),
        "seconds": round(time.monotonic() - started, 1),
        "model": model_id,
        "think_types": sorted(settings["think_types"]),
        "always_types": sorted(settings["always_types"]),
        "few_shot_prompts": shots.used if shots is not None else 0,
        "ranker": _ranker_summary(ranker_meta),
        "pipeline_version": PIPELINE_VERSION,
    }
    logger.info("emergence %s/%s: %s", source_kind, source_ref, result.stats)
    if persist:
        from pawn_agent.core.emergence import store  # noqa: PLC0415

        store.save_run(cfg.db_dsn, result, pipeline_version=PIPELINE_VERSION)
    return result


def _due(value: Any) -> str:
    text = str(value or "").strip()
    return text if _DATE_RE.match(text) else ""


def quote_overlap(a: str, b: str) -> float:
    """Share of the shorter quote's words that appear, in order, in the other."""
    wa = [w.lower() for w in _WORD_RE.findall(a or "")]
    wb = [w.lower() for w in _WORD_RE.findall(b or "")]
    if not wa or not wb:
        return 0.0
    matcher = difflib.SequenceMatcher(None, wa, wb, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / min(len(wa), len(wb))


def _same_statement(a: dict[str, Any], b: dict[str, Any]) -> bool:
    sa, sb = a.get("start_sec"), b.get("start_sec")
    if sa is not None and sb is not None and abs(sa - sb) > _SAME_STATEMENT_SECONDS:
        return False
    return quote_overlap(a.get("quote") or "", b.get("quote") or "") >= _SAME_STATEMENT_OVERLAP


def _priority(item: dict[str, Any]) -> int:
    kind = item.get("kind") or ""
    rank = KIND_PRIORITY.index(kind) if kind in KIND_PRIORITY else len(KIND_PRIORITY)
    # A commitment with nobody owing it is weaker than a decision.
    if kind == "commitment" and not item.get("owner"):
        rank = KIND_PRIORITY.index("decision") + 0.5
    return int(rank * 2)


def merge_statements(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one item per statement across kinds.

    The model often files "I'll share the doc" as both a decision and a
    commitment. Items whose quotes mostly overlap within two minutes are one
    statement; the higher-priority kind is kept and the others' observation
    ids are recorded in ``merged_observation_ids``.
    """
    ordered = sorted(items, key=_priority)
    kept: list[dict[str, Any]] = []
    for item in ordered:
        twin = next((k for k in kept if _same_statement(k, item)), None)
        if twin is None:
            kept.append(item)
            continue
        merged = twin.setdefault("merged_observation_ids", [])
        if item.get("observation_id"):
            merged.append(item["observation_id"])
        if not twin.get("owner") and item.get("owner"):
            twin["owner"] = item["owner"]
    kept.sort(key=lambda it: (it["start_sec"] is None, it["start_sec"] or 0.0))
    return kept


def project_items(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Live actionable observations → coworker item dicts (time order).

    One statement filed under several kinds yields one item
    (:func:`merge_statements`).
    """
    items: list[dict[str, Any]] = []
    for obs in observations:
        if obs.get("status") != "live":
            continue
        kind = ITEM_KINDS.get(obs.get("type") or "")
        if not kind:
            continue
        fields = obs.get("fields") or {}
        owner = str(fields.get(OWNER_FIELD.get(obs["type"], ""), "") or "").strip()
        items.append(
            {
                "kind": kind,
                "text": obs.get("text") or "",
                "context": obs.get("context") or "",
                "owner": owner,
                "due": _due(fields.get("due")),
                "quote": obs.get("quote") or "",
                "start_sec": obs.get("start_sec"),
                "window": obs.get("window"),
                "quote_score": obs.get("quote_score"),
                "observation_id": obs.get("id"),
                "observation_type": obs.get("type"),
                "rank_score": obs.get("rank_score"),
            }
        )
    return merge_statements(items)

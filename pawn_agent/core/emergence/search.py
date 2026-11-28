"""Prompt search (level 4) and label export for external fine-tuning (level 5).

Prompt search tries rewritten definitions/rules for one observation type on
the sessions the user has labelled in compare notes, scores each variant
against the labels (wanted lines found = hits, unwanted lines found = false
alarms, wanted lines missed = misses), and promotes a variant only when its
F1 beats the live prompt by ``min_gain``. Winners live in
``{state_dir}/emergence/prompt_overrides.json`` and tag observations
(``decision.v2+s1``).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import select

from pawn_agent.core.coworker.extract import claims_similar, split_annotations, window_source
from pawn_agent.core.emergence import passes, prompts
from pawn_agent.core.emergence.llm import ask_json
from pawn_agent.utils.db import EmLabel, _get_session

logger = logging.getLogger(__name__)


def overrides_path(cfg: Any) -> Path:
    from pawn_agent.core.sallm_factory import resolve_state_dir  # noqa: PLC0415

    return resolve_state_dir(cfg) / "emergence" / "prompt_overrides.json"


def load_overrides(cfg: Any) -> dict[str, dict]:
    path = overrides_path(cfg)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("prompt overrides unreadable: %s", exc)
        return {}


def install_overrides(cfg: Any) -> None:
    try:
        prompts.set_overrides(load_overrides(cfg))
    except Exception as exc:
        logger.debug("prompt overrides not installed: %s", exc)


def review_labels(dsn: str, type_name: str) -> dict[str, list[dict[str, Any]]]:
    """Review-note labels of one type, grouped by source ref."""
    with _get_session(dsn) as db:
        rows = db.scalars(
            select(EmLabel).where(
                EmLabel.type == type_name,
                EmLabel.source_ref.is_not(None),
                EmLabel.verdict.in_(("accept", "reject", "missed")),
                EmLabel.surface.in_(("review", "api", "inbox", "note", "chat")),
            )
        ).all()
        out: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            out.setdefault(str(r.source_ref), []).append(
                {"text": r.text, "quote": r.quote or "", "positive": r.verdict != "reject"}
            )
        return out


def score(found: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, float]:
    """Hits / false alarms / misses of extracted texts against labels."""
    hits = alarms = misses = 0
    for label in gold:
        matched = any(
            claims_similar(label["text"], f.get("text") or "", 0.6)
            or (label["quote"] and claims_similar(label["quote"], f.get("quote") or "", 0.6))
            for f in found
        )
        if label["positive"]:
            hits += matched
            misses += not matched
        else:
            alarms += matched
    denom = 2 * hits + alarms + misses
    return {
        "hits": hits,
        "false_alarms": alarms,
        "misses": misses,
        "f1": round(2 * hits / denom, 3) if denom else 0.0,
    }


_VARIANT_PROMPT = """\
You are improving an extraction prompt for "{name}" observations.

Current definition: {definition}
Current rules:
{rules}

It extracted these that the user did NOT want:
{alarms}

It missed these that the user DID want:
{misses}

Write {n} alternative versions of the definition and rules that would fix
these mistakes without overfitting to the exact wording. Return JSON:
{{"variants": [{{"definition": "...", "rules": ["...", "..."]}}]}}
"""


async def _run_spec(
    cfg: Any, type_name: str, spec: Any, windows_by_source: dict[str, list[Any]], budget: Any
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for ref, windows in windows_by_source.items():
        found: list[dict[str, Any]] = []
        for w in windows:
            if budget is not None:
                budget.spend()
            part = prompts.part_note(w.index, len(windows), "")
            try:
                found += await passes.extract_type(
                    cfg, type_name, w.text, part, spec=spec, think=False
                )
            except passes.PassError as exc:
                logger.warning("prompt search call failed: %s", exc)
        out[ref] = found
    return out


def _total(
    results: dict[str, list[dict[str, Any]]], gold: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    agg: dict[str, float] = {"hits": 0, "false_alarms": 0, "misses": 0}
    for ref, labels in gold.items():
        s = score(results.get(ref, []), labels)
        for k in agg:
            agg[k] += s[k]
    denom = 2 * agg["hits"] + agg["false_alarms"] + agg["misses"]
    agg["f1"] = round(2 * agg["hits"] / denom, 3) if denom else 0.0
    return agg


async def prompt_search(
    cfg: Any,
    store: Any,
    type_name: str,
    *,
    variants: int = 2,
    min_gain: float = 0.02,
    max_sources: int = 5,
    budget: Any = None,
) -> dict[str, Any]:
    """Search, report to ``Pawn/Reviews/prompt-search-…``, maybe promote."""
    from pawn_agent.core.coworker.pipeline import session_source_text  # noqa: PLC0415
    from pawn_agent.core.emergence.vault_notes import write_plain  # noqa: PLC0415

    install_overrides(cfg)
    gold = review_labels(cfg.db_dsn, type_name)
    gold = {ref: rows for ref, rows in gold.items() if any(r["positive"] for r in rows)}
    gold = dict(list(gold.items())[:max_sources])
    if not gold:
        return {"status": "no_labels", "type": type_name}
    em = getattr(cfg, "emergence", None)
    windows_by_source: dict[str, list[Any]] = {}
    for ref in gold:
        try:
            _, body = split_annotations(session_source_text(cfg, ref, store))
        except Exception as exc:
            logger.warning("prompt search: no transcript for %s: %s", ref, exc)
            continue
        windows_by_source[ref] = window_source(
            body,
            max_chars=int(getattr(em, "window_chars", 12000) or 12000),
            overlap_chars=int(getattr(em, "window_overlap", 1500) or 0),
        )
    gold = {ref: rows for ref, rows in gold.items() if ref in windows_by_source}
    if not gold:
        return {"status": "no_transcripts", "type": type_name}
    current = prompts.effective_spec(type_name)
    base_results = await _run_spec(cfg, type_name, current, windows_by_source, budget)
    base = _total(base_results, gold)
    alarms, misses = [], []
    for ref, labels in gold.items():
        found = base_results.get(ref, [])
        for label in labels:
            hit = any(claims_similar(label["text"], f.get("text") or "", 0.6) for f in found)
            if label["positive"] and not hit:
                misses.append(label["text"])
            if not label["positive"] and hit:
                alarms.append(label["text"])
    data = await ask_json(
        cfg,
        _VARIANT_PROMPT.format(
            name=type_name,
            definition=current.definition,
            rules="\n".join(f"- {r}" for r in current.rules) or "(none)",
            alarms="\n".join(f"- {a}" for a in alarms[:12]) or "(none)",
            misses="\n".join(f"- {m}" for m in misses[:12]) or "(none)",
            n=variants,
        ),
        think=True,
        budget=budget,
    )
    tried = []
    for v in (data or {}).get("variants", [])[:variants] if isinstance(data, dict) else []:
        if not isinstance(v, dict) or not v.get("definition"):
            continue
        spec = prompts.TypeSpec(
            name=current.name,
            version=current.version,
            definition=str(v["definition"]),
            fields=current.fields,
            rules=tuple(str(r) for r in v.get("rules") or []),
        )
        result = _total(await _run_spec(cfg, type_name, spec, windows_by_source, budget), gold)
        tried.append((spec, result))
    best_spec, best = max(tried, key=lambda pair: pair[1]["f1"], default=(None, None))
    promoted = bool(best_spec and best and best["f1"] >= base["f1"] + min_gain)
    if promoted and best_spec is not None and best is not None:
        overrides = load_overrides(cfg)
        n = int(str((overrides.get(type_name) or {}).get("tag") or "s0")[1:] or 0) + 1
        overrides[type_name] = {
            "definition": best_spec.definition,
            "rules": list(best_spec.rules),
            "tag": f"s{n}",
            "f1": best["f1"],
            "baseline_f1": base["f1"],
            "promoted": date.today().isoformat(),
        }
        path = overrides_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(overrides, indent=2))
        prompts.set_overrides(overrides)
    lines = [
        f"| live prompt | {base['f1']} | {base['hits']} | {base['false_alarms']} | {base['misses']} |"
    ]
    for i, (spec, result) in enumerate(tried, 1):
        lines.append(
            f"| variant {i} | {result['f1']} | {result['hits']} | {result['false_alarms']} | {result['misses']} |"
        )
    report = "\n".join(
        ["| Prompt | F1 | Hits | False alarms | Misses |", "|---|---|---|---|---|"] + lines
    )
    sections = [
        ("Result", report),
        ("Outcome", "Promoted variant." if promoted else "Kept the live prompt."),
    ]
    for i, (spec, _) in enumerate(tried, 1):
        sections.append(
            (f"Variant {i}", spec.definition + "\n\n" + "\n".join(f"- {r}" for r in spec.rules))
        )
    key = f"{getattr(cfg.coworker, 'reviews_dir', 'Pawn/Reviews').rstrip('/')}/prompt-search-{type_name}-{date.today():%Y-%m-%d}.md"
    try:
        write_plain(
            cfg,
            store,
            key,
            meta={"pawn": "prompt-search", "type": type_name},
            title=f"Prompt search: {type_name}",
            sections=sections,
        )
    except Exception as exc:
        logger.warning("prompt search report not written: %s", exc)
    return {
        "status": "promoted" if promoted else "kept",
        "type": type_name,
        "baseline": base,
        "variants": [r for _, r in tried],
        "report": key,
    }


def export_labels(dsn: str, path: Path, *, include_bulk: bool = False) -> int:
    """Write labels as JSONL (``type, text, quote, context, verdict, surface, source_ref``)."""
    with _get_session(dsn) as db:
        # Strings and cards (brainmap) judge structure, not extraction: not training data here.
        stmt = select(EmLabel).where(EmLabel.target_kind.not_in(("link", "subject")))
        if not include_bulk:
            stmt = stmt.where(EmLabel.surface != "bulk")
        rows = db.scalars(stmt.order_by(EmLabel.created_at)).all()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for r in rows:
                fh.write(
                    json.dumps(
                        {
                            "type": r.type,
                            "text": r.text,
                            "quote": r.quote,
                            "context": r.context,
                            "verdict": r.verdict,
                            "surface": r.surface,
                            "source_ref": r.source_ref,
                            "start_sec": r.start_sec,
                            "extractor": r.extractor,
                        }
                    )
                    + "\n"
                )
        return len(rows)

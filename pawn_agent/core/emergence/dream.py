"""Imagination: dreaming (hypotheses) and the council for decisions."""

from __future__ import annotations

import logging
import random
from typing import Any, Optional

import numpy as np

from pawn_agent.core.emergence import labels, provocations
from pawn_agent.core.emergence import subjects as subj
from pawn_agent.core.emergence import vault_notes
from pawn_agent.core.emergence.llm import ask_json

logger = logging.getLogger(__name__)

_DREAM_PROMPT = """\
Two things a manager heard, in different contexts and at different times:

A ({a_when}, {a_where}): {a}
B ({b_when}, {b_where}): {b}

Is there a non-obvious, useful connection between them (a shared cause, a
risk one poses to the other, an opportunity, a pattern)? Return JSON:
{{"connected": false, "hypothesis": "one or two sentences", "test": "what evidence would confirm it"}}
Say "connected": false unless the link is specific and worth their time.
"""

_CRITIC_PROMPT = """\
Be a strict critic. A colleague proposes this hypothesis from two notes:

A: {a}
B: {b}
Hypothesis: {hypothesis}

Is it specific, plausible from A and B alone, and useful to a manager?
Return JSON: {{"keep": false, "why": "one line"}}
"""


def _pairs(
    observations: list[dict[str, Any]], n: int, *, seed: Optional[int] = None
) -> list[tuple]:
    """Pairs of observations that are far apart in time and moderately similar."""
    rng = random.Random(seed)
    usable = [o for o in observations if o.get("embedding") and o.get("created_at")]
    if len(usable) < 2:
        return []
    units = [subj.unit(o["embedding"]) for o in usable]
    usable = [o for o, v in zip(usable, units) if v is not None]
    if len(usable) < 2:
        return []
    vecs = np.stack([v for v in units if v is not None])
    out: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    tries = 0
    seen: set[tuple[str, str]] = set()
    while len(out) < n and tries < n * 40:
        tries += 1
        i, j = rng.randrange(len(usable)), rng.randrange(len(usable))
        a, b = usable[i], usable[j]
        if i == j or a.get("source_ref") == b.get("source_ref"):
            continue
        if abs((a["created_at"] - b["created_at"]).days) < 7:
            continue
        distance = float(1.0 - vecs[i] @ vecs[j])
        if not 0.3 <= distance <= 0.65:
            continue
        key = tuple(sorted((a["id"], b["id"])))
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b, distance))
    return out


async def dream(cfg: Any, store: Any, *, budget: Any = None, seed: Optional[int] = None) -> int:
    n = int(getattr(getattr(cfg, "emergence", None), "dream_pairs", 12) or 0)
    if n <= 0:
        return 0
    pool = subj.recent_observations(cfg.db_dsn, days=120)
    kept = 0
    for a, b, distance in _pairs(pool, n, seed=seed):
        if budget is not None and budget.exhausted:
            break
        idea = await ask_json(
            cfg,
            _DREAM_PROMPT.format(
                a=a["text"],
                b=b["text"],
                a_when=f"{a['created_at']:%Y-%m-%d}",
                b_when=f"{b['created_at']:%Y-%m-%d}",
                a_where=a.get("speaker_name") or a.get("source_ref") or "",
                b_where=b.get("speaker_name") or b.get("source_ref") or "",
            ),
            budget=budget,
        )
        if not isinstance(idea, dict) or idea.get("connected") is not True:
            continue
        hypothesis = str(idea.get("hypothesis") or "").strip()
        if not hypothesis:
            continue
        verdict = await ask_json(
            cfg,
            _CRITIC_PROMPT.format(a=a["text"], b=b["text"], hypothesis=hypothesis),
            budget=budget,
        )
        if not isinstance(verdict, dict) or verdict.get("keep") is not True:
            continue
        [vector] = labels.embed_many(cfg, [hypothesis])
        dupes = subj.nearest(vector, subj.list_subjects(cfg.db_dsn, kinds=("hypothesis",)))
        if dupes and dupes[0][0] < 0.15:
            continue
        h = subj.create_subject(
            cfg.db_dsn,
            kind="hypothesis",
            title=hypothesis[:200],
            origin="dream",
            status="proposed",
            state_md=hypothesis,
            fields={
                "evidence": [a["text"], b["text"]],
                "test": idea.get("test") or "",
                "critic": verdict.get("why") or "",
                "distance": round(distance, 3),
            },
            centroid=vector,
        )
        subj.link(cfg.db_dsn, a["id"], h["id"], role="evidence_for", linked_by="dream")
        subj.link(cfg.db_dsn, b["id"], h["id"], role="evidence_for", linked_by="dream")
        vault_notes.render_subject(cfg, store, subj.get_subject(cfg.db_dsn, h["id"]) or h)
        provocations.add(
            cfg,
            kind="hypothesis",
            subject_id=h["id"],
            text=f"A hunch from the night shift: {hypothesis} Worth watching?",
            features={"novelty": 1.0},
        )
        kept += 1
    return kept


PERSONAS = (
    ("Your boss", "cares about outcomes, risk to the roadmap and how this looks upward"),
    ("A skeptical senior engineer", "cares about technical debt, feasibility and hidden cost"),
    ("Finance", "cares about cost, headcount and return"),
    ("An affected report", "cares about workload, clarity and fairness"),
    ("You in six months", "knows how it played out and what you wish you had checked"),
)

_PERSONA_PROMPT = """\
You are {persona}: you {cares}.
A manager is deciding: {decision}
Rationale given: {rationale}
Assumptions: {assumptions}
Context: {context}

Give your honest critique. Return JSON:
{{"concerns": ["..."], "questions": ["..."], "verdict": "support|caution|oppose"}}
"""

_PREMORTEM_PROMPT = """\
It is six months from now and this decision failed badly: {decision}
Assumptions it rested on: {assumptions}
Context: {context}
Write the most plausible story of how it failed. Return JSON:
{{"story": "4-6 sentences", "early_signals": ["what would have warned them"]}}
"""

_SYNTH_PROMPT = """\
Critiques of the decision "{decision}":
{critiques}
Pre-mortem: {premortem}

Return JSON: {{"summary": "3-5 lines", "checks": ["concrete things to verify before committing"]}}
"""


async def council(cfg: Any, store: Any, subject_id: str, *, budget: Any = None) -> Optional[str]:
    """Run the council on one decision; writes a note and links it."""
    d = subj.get_subject(cfg.db_dsn, subject_id)
    if d is None or d["kind"] != "decision":
        return None
    fields = dict(d.get("fields") or {})
    context = d.get("state_md") or fields.get("quote") or ""
    if d.get("parent_id"):
        parent = subj.get_subject(cfg.db_dsn, d["parent_id"])
        if parent and parent.get("state_md"):
            context = f"{parent['title']}: {parent['state_md']}"
    common = {
        "decision": fields.get("choice") or d["title"],
        "rationale": fields.get("rationale") or "(not stated)",
        "assumptions": "; ".join(fields.get("assumptions") or []) or "(none stated)",
        "context": context[:1500] or "(none)",
    }
    critiques = []
    for persona, cares in PERSONAS:
        data = await ask_json(
            cfg,
            _PERSONA_PROMPT.format(persona=persona, cares=cares, **common),
            think=True,
            budget=budget,
        )
        if isinstance(data, dict):
            critiques.append((persona, data))
    premortem = await ask_json(cfg, _PREMORTEM_PROMPT.format(**common), think=True, budget=budget)
    premortem = premortem if isinstance(premortem, dict) else {}
    summary = await ask_json(
        cfg,
        _SYNTH_PROMPT.format(
            decision=common["decision"],
            critiques="\n".join(
                f"- {p} ({c.get('verdict')}): " + "; ".join(c.get("concerns") or [])
                for p, c in critiques
            ),
            premortem=premortem.get("story") or "",
        ),
        budget=budget,
    )
    summary = summary if isinstance(summary, dict) else {}

    def bullets(values: Any) -> str:
        return "\n".join(f"- {v}" for v in values) if isinstance(values, list) else ""

    sections = [
        ("Summary", str(summary.get("summary") or "")),
        ("Check before committing", bullets(summary.get("checks"))),
    ]
    for persona, c in critiques:
        sections.append(
            (
                f"{persona} — {c.get('verdict') or '?'}",
                bullets(c.get("concerns"))
                + ("\n\nQuestions:\n" + bullets(c.get("questions")) if c.get("questions") else ""),
            )
        )
    sections.append(
        (
            "Pre-mortem",
            str(premortem.get("story") or "")
            + (
                "\n\nEarly signals:\n" + bullets(premortem.get("early_signals"))
                if premortem.get("early_signals")
                else ""
            ),
        )
    )
    created = d.get("created_at") or subj.now()
    key = f"{vault_notes.root(cfg)}/Decisions/{created:%Y-%m-%d}-{d['slug']}-council.md"
    vault_notes.write_plain(
        cfg,
        store,
        key,
        meta={"pawn": "council", "decision": d["id"]},
        title=f"Council: {d['title']}",
        sections=sections,
    )
    fields.update(council_note=key[:-3], council=False, council_at=subj.now().isoformat())
    subj.update_subject(cfg.db_dsn, d["id"], fields=fields)
    subj.add_event(cfg.db_dsn, d["id"], "council", {"note": key})
    vault_notes.render_subject(cfg, store, subj.get_subject(cfg.db_dsn, d["id"]) or d)
    return key


async def council_due(cfg: Any, store: Any, *, budget: Any = None) -> list[str]:
    """Decisions tagged ``council: true`` in their note (read by sync)."""
    from pawn_core.vault import parse_frontmatter  # noqa: PLC0415

    written = []
    for d in subj.list_subjects(cfg.db_dsn, kinds=("decision",), statuses=("open",)):
        fields = d.get("fields") or {}
        wanted = bool(fields.get("council"))
        key = d.get("note_key")
        if not wanted and key:
            try:
                meta, _ = parse_frontmatter(store.read(key)) if store.exists(key) else ({}, "")
                # The note is re-rendered with ``council: false`` after a run,
                # so ``true`` there is always a fresh request from the user.
                wanted = meta.get("council") is True
            except Exception:
                wanted = False
        if not wanted:
            continue
        out = await council(cfg, store, d["id"], budget=budget)
        if out:
            written.append(out)
    return written

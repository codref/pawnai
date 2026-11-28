"""Night jobs for threads and questions: sync, clustering, synthesis."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import timedelta
from typing import Any, Optional

import numpy as np

from pawn_agent.core.emergence import labels
from pawn_agent.core.emergence import subjects as subj
from pawn_agent.core.emergence import vault_notes
from pawn_agent.core.emergence.llm import ask_json

logger = logging.getLogger(__name__)

# Statuses a user may set in a note's frontmatter.
USER_STATUSES = {"active", "dismissed", "done", "parked", "resolved", "open"}
CLUSTER_TYPES = ("concern", "fact", "question", "block", "contradiction", "prediction")


def _em(cfg: Any, name: str, default: Any) -> Any:
    return getattr(getattr(cfg, "emergence", None), name, default)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── sync ─────────────────────────────────────────────────────────────────────


def sync_goals(cfg: Any, store: Any) -> dict[str, int]:
    """Goals.md threads → ``thread`` subjects with origin ``goal``."""
    from pawn_core.goals import load_goals_text  # noqa: PLC0415

    goals_path = getattr(getattr(cfg, "coworker", None), "goals_path", "Goals.md")
    try:
        text = store.read(goals_path) if store.exists(goals_path) else ""
    except Exception:
        text = ""
    goals = load_goals_text(text)
    counts = {"created": 0, "updated": 0}
    for thread in goals.threads:
        status = {"active": "active", "parked": "parked"}.get(thread.status, "done")
        seed = f"{thread.name}. {thread.why} {thread.movement}".strip()
        digest = _hash(seed)
        existing = subj.find_subject(cfg.db_dsn, kind="thread", slug=thread.slug)
        if existing is None:
            [vector] = labels.embed_many(cfg, [seed])
            subj.create_subject(
                cfg.db_dsn,
                kind="thread",
                title=thread.name,
                slug=thread.slug,
                origin="goal",
                status=status,
                fields={
                    "why": thread.why,
                    "movement": thread.movement,
                    "interrupt": thread.interrupt,
                    "_seed": digest,
                },
                centroid=vector,
            )
            counts["created"] += 1
            continue
        fields = dict(existing.get("fields") or {})
        changes: dict[str, Any] = {}
        if fields.get("_seed") != digest:
            fields.update(
                why=thread.why, movement=thread.movement, interrupt=thread.interrupt, _seed=digest
            )
            changes["fields"] = fields
            if existing.get("centroid") is None:
                changes["centroid"] = labels.embed_many(cfg, [seed])[0]
        if existing["origin"] != "goal":
            changes["origin"] = "goal"
        if changes:
            subj.update_subject(cfg.db_dsn, existing["id"], **changes)
            counts["updated"] += 1
        if existing["status"] != status:
            subj.set_status(cfg.db_dsn, existing["id"], status, reason="Goals.md")
    return counts


def sync_notes(cfg: Any, store: Any) -> dict[str, int]:
    """Read user edits back: frontmatter ``status`` and new question notes.

    A note's ``status`` that differs from what Pawn last wrote there is the
    user's command (accept a proposed thread with ``active``, drop it with
    ``dismissed``). A status changed elsewhere (verdict, API, CLI) since the
    last render is written back to the note instead.
    A note under ``{root}/Questions/`` with ``pawn: question`` and no ``id``
    becomes a user question.
    """
    from pawn_core.vault import dump_frontmatter, parse_frontmatter  # noqa: PLC0415

    counts = {"status": 0, "questions": 0, "expired": 0}
    for s in subj.list_subjects(cfg.db_dsn):
        key = s.get("note_key")
        if not key:
            continue
        try:
            if not store.exists(key):
                continue
            meta, _ = parse_frontmatter(store.read(key))
        except Exception:
            continue
        wanted = str(meta.get("status") or "").strip().lower()
        written = str((s.get("fields") or {}).get("_note_status") or s["status"])
        if wanted != written and wanted in USER_STATUSES:
            subj.set_status(cfg.db_dsn, s["id"], wanted, reason="note edited")
            counts["status"] += 1
            s = subj.get_subject(cfg.db_dsn, s["id"]) or s
        elif written != s["status"] or wanted != s["status"]:
            counts.setdefault("rerendered", 0)
            counts["rerendered"] += 1
        else:
            continue
        if s["kind"] not in ("commitment", "prediction"):
            vault_notes.render_subject(cfg, store, s)
    prefix = f"{vault_notes.root(cfg)}/Questions/"
    try:
        keys = store.list(prefix)
    except Exception:
        keys = []
    known = {s.get("note_key") for s in subj.list_subjects(cfg.db_dsn, kinds=("question",))}
    for key in keys:
        if key in known:
            continue
        try:
            raw = store.read(key)
            meta, body = parse_frontmatter(raw)
        except Exception:
            continue
        if str(meta.get("pawn") or "") != "question" or meta.get("id"):
            continue
        title = key.rsplit("/", 1)[-1][:-3]
        first = next((ln.lstrip("# ").strip() for ln in body.splitlines() if ln.strip()), "")
        title = first or title
        [vector] = labels.embed_many(cfg, [title])
        created = subj.create_subject(
            cfg.db_dsn,
            kind="question",
            title=title,
            origin="user",
            status="active",
            note_key=key,
            centroid=vector,
        )
        meta.update(id=created["id"], status="active", origin="user")
        store.write(key, dump_frontmatter(meta, body))
        counts["questions"] += 1
    expiry = subj.now() - timedelta(days=int(_em(cfg, "proposed_expiry_days", 21)))
    for s in subj.list_subjects(cfg.db_dsn, statuses=("proposed",)):
        if s.get("created_at") and s["created_at"] < expiry:
            subj.set_status(cfg.db_dsn, s["id"], "expired", reason="not accepted")
            counts["expired"] += 1
    return counts


# ── clustering ───────────────────────────────────────────────────────────────


def cluster(
    observations: list[dict[str, Any]], *, max_distance: float
) -> list[list[dict[str, Any]]]:
    """Greedy centroid clustering on unit embeddings (order-stable)."""
    clusters: list[dict[str, Any]] = []
    for obs in observations:
        vec = subj.unit(obs.get("embedding"))
        if vec is None:
            continue
        best, best_d = None, max_distance
        for c in clusters:
            d = float(1.0 - c["centroid"] @ vec)
            if d < best_d:
                best, best_d = c, d
        if best is None:
            clusters.append({"centroid": vec, "sum": vec.copy(), "members": [obs]})
        else:
            best["members"].append(obs)
            best["sum"] = best["sum"] + vec
            best["centroid"] = best["sum"] / np.linalg.norm(best["sum"])
    return [c["members"] for c in clusters]


_NAME_PROMPT = """\
These observations from different meetings and notes seem to form one
storyline this manager is not tracking yet.

{observations}

Return JSON:
{{"title": "short storyline name (3-8 words)",
  "state": "3-6 lines: what is going on, who is involved, what is at stake",
  "question": "the open question at its heart, or empty",
  "coherent": true}}
Set "coherent": false if they do not actually belong together.
"""


async def emergent_threads(cfg: Any, store: Any, *, budget: Any = None) -> list[dict[str, Any]]:
    """Propose threads from clusters of unrouted observations."""
    from pawn_agent.core.emergence import provocations  # noqa: PLC0415

    unrouted = subj.recent_observations(
        cfg.db_dsn,
        days=int(_em(cfg, "cluster_window_days", 30)),
        unrouted=True,
        types=CLUSTER_TYPES,
    )
    groups = cluster(unrouted, max_distance=float(_em(cfg, "cluster_distance", 0.35)))
    min_obs = int(_em(cfg, "cluster_min_observations", 4))
    min_sources = int(_em(cfg, "cluster_min_sources", 2))
    proposed: list[dict[str, Any]] = []
    for members in groups:
        sources = {m.get("source_ref") for m in members}
        if len(members) < min_obs or len(sources) < min_sources:
            continue
        listing = "\n".join(
            f"- [{m['type']}] {m['text']}"
            + (f" ({m['speaker_name']})" if m.get("speaker_name") else "")
            for m in members[:20]
        )
        data = await ask_json(cfg, _NAME_PROMPT.format(observations=listing), budget=budget)
        if not isinstance(data, dict) or not data.get("title") or data.get("coherent") is False:
            continue
        vectors = [v for v in (subj.unit(m["embedding"]) for m in members) if v is not None]
        centroid = np.mean(np.stack(vectors), axis=0)
        thread = subj.create_subject(
            cfg.db_dsn,
            kind="thread",
            title=str(data["title"]).strip()[:120],
            origin="emergent",
            status="proposed",
            state_md=str(data.get("state") or "").strip(),
            state_at=subj.now(),
            centroid=[float(x) for x in centroid],
        )
        for m in members:
            subj.link(cfg.db_dsn, m["id"], thread["id"], linked_by="cluster")
        question = str(data.get("question") or "").strip()
        if question:
            [qvec] = labels.embed_many(cfg, [question])
            subj.create_subject(
                cfg.db_dsn,
                kind="question",
                title=question,
                origin="emergent",
                status="proposed",
                parent_id=thread["id"],
                centroid=qvec,
            )
        people = sorted({m["speaker_name"] for m in members if m.get("speaker_name")})
        provocations.add(
            cfg,
            kind="emergent_thread",
            subject_id=thread["id"],
            text=(
                f"A storyline is forming: **{thread['title']}** ({len(members)} observations "
                f"across {len(sources)} sources"
                + (f", {', '.join(people[:4])}" if people else "")
                + "). Track it? Set `status: active` in its note."
            ),
            features={"tension": 0.3, "recurrence": len(sources), "novelty": 1.0},
        )
        vault_notes.render_subject(cfg, store, subj.get_subject(cfg.db_dsn, thread["id"]) or thread)
        proposed.append(thread)
    return proposed


# ── synthesis ────────────────────────────────────────────────────────────────


_SYNTH_PROMPT = """\
You keep a running summary of one {kind} for a manager.

Title: {title}
{extra}
Current summary:
{state}

New observations since the last summary:
{observations}

Rewrite the summary in 4-10 short markdown lines: where things stand, who
is involved, what changed, what is at stake. Only claim what the summary or
observations support.
Return JSON: {{"state": "...", "answered": false, "answer": ""}}
{answer_rule}
"""


def momentum(cfg: Any, subject_id: str) -> float:
    """Evidence this week relative to the trailing three-week weekly rate."""
    week = subj.link_counts(cfg.db_dsn, subject_id, days=7)
    month = subj.link_counts(cfg.db_dsn, subject_id, days=28)
    baseline = max(1.0, (month - week) / 3.0)
    return round(week / baseline, 2)


async def synthesize(cfg: Any, store: Any, *, budget: Any = None) -> int:
    """Rewrite the state of every thread/question with new evidence."""
    from pawn_agent.core.emergence import provocations  # noqa: PLC0415

    done = 0
    links = vault_notes.SourceLinks(cfg)
    for s in subj.list_subjects(
        cfg.db_dsn, kinds=("thread", "question"), statuses=("active", "proposed", "parked")
    ):
        if budget is not None and budget.exhausted:
            break
        new = subj.subject_observations(cfg.db_dsn, s["id"], since=s.get("state_at"), limit=25)
        if not new:
            continue
        extra = ""
        if (s.get("fields") or {}).get("why"):
            extra = f"Why it matters: {s['fields']['why']}\n"
        answer_rule = (
            'For this question, set "answered": true and give "answer" only when '
            "the observations clearly settle it."
            if s["kind"] == "question"
            else 'Leave "answered" false.'
        )
        data = await ask_json(
            cfg,
            _SYNTH_PROMPT.format(
                kind=s["kind"],
                title=s["title"],
                extra=extra,
                state=s.get("state_md") or "(none yet)",
                observations="\n".join(
                    f"- [{o['type']}] {o['text']}"
                    + (f" ({o['speaker_name']})" if o.get("speaker_name") else "")
                    for o in new
                ),
                answer_rule=answer_rule,
            ),
            budget=budget,
        )
        if not isinstance(data, dict) or not str(data.get("state") or "").strip():
            continue
        before = s.get("state_md") or ""
        after = str(data["state"]).strip()
        m = momentum(cfg, s["id"])
        subj.update_subject(cfg.db_dsn, s["id"], state_md=after, state_at=subj.now(), momentum=m)
        subj.add_event(
            cfg.db_dsn,
            s["id"],
            "state_rewritten",
            {"before": before, "after": after, "momentum": m},
            [o["id"] for o in new],
        )
        if s["kind"] == "question" and data.get("answered") is True and data.get("answer"):
            subj.set_status(
                cfg.db_dsn,
                s["id"],
                "resolved",
                reason=str(data["answer"])[:500],
                causes=[o["id"] for o in new],
            )
            provocations.add(
                cfg,
                kind="question_answered",
                subject_id=s["id"],
                text=f"Your question **{s['title']}** looks answered: {data['answer']}",
                features={"novelty": 0.8, "goal_thread": 0.0},
            )
        elif s["kind"] == "thread" and s["status"] == "active" and m >= 2.0 and len(new) >= 3:
            provocations.add(
                cfg,
                kind="momentum",
                subject_id=s["id"],
                text=f"**{s['title']}** is heating up ({len(new)} new observations, "
                f"{m}× its usual pace). Anything you need to do?",
                features={"momentum": m, "goal_thread": 1.0 if s["origin"] == "goal" else 0.0},
            )
        vault_notes.render_subject(
            cfg, store, subj.get_subject(cfg.db_dsn, s["id"]) or s, links=links
        )
        done += 1
        await asyncio.sleep(0)
    return done


def render_all(cfg: Any, store: Any, kinds: Optional[tuple[str, ...]] = None) -> int:
    """Re-render every subject note (used after migrations or prompt changes)."""
    links = vault_notes.SourceLinks(cfg)
    count = 0
    for s in subj.list_subjects(cfg.db_dsn, kinds=kinds):
        if s["kind"] in ("commitment", "prediction"):
            continue
        if s["status"] in ("dismissed", "expired") and not s.get("note_key"):
            continue
        vault_notes.render_subject(cfg, store, s, links=links)
        count += 1
    return count

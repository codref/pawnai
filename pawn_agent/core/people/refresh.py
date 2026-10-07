"""Post-session people refresh: Appearances always (when enabled), Facts under autonomy.

Triggered by queue command ``speakers_refresh`` or from ``session_completed``.
Only gallery-resolved speakers are updated — never invent people or enroll voice.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

from pawn_agent.core.coworker import db as itemdb
from pawn_agent.core.coworker.autonomy import decide
from pawn_agent.core.coworker.notes import render_item_note
from pawn_agent.core.coworker.policy import fingerprint
from pawn_agent.core.people.extract import extract_people_facts
from pawn_agent.core.people.notes import (
    ensure_person_note,
    people_wiki_target,
    update_person_note,
)
from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.db import create_agent_run, update_agent_run
from pawn_core.vault_config import vault_store_from_config
from pawn_diarize.core.speaker_gallery import SpeakerGallery

logger = logging.getLogger(__name__)


def _people_cfg(cfg: AgentConfig) -> Any:
    return getattr(cfg.coworker, "people", None)


def _already_refreshed(cfg: AgentConfig, session_id: str, *, hours: int = 24) -> bool:
    """True when a completed speakers_refresh already ran for this session."""
    from datetime import datetime, timedelta, timezone  # noqa: PLC0415

    from sqlalchemy import select  # noqa: PLC0415
    from sqlalchemy.orm import Session  # noqa: PLC0415

    from pawn_agent.utils.db import AgentRun  # noqa: PLC0415
    from pawn_core.database import get_engine  # noqa: PLC0415

    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    with Session(get_engine(cfg.db_dsn)) as db:
        row = db.scalars(
            select(AgentRun.id)
            .where(
                AgentRun.session_id == session_id,
                AgentRun.command == "speakers_refresh",
                AgentRun.status == "completed",
                AgentRun.created_at >= since,
            )
            .limit(1)
        ).first()
    return row is not None


def _talk_stats(cfg: AgentConfig, session_id: str) -> dict[str, dict[str, Any]]:
    """Map speaker_id → {display_name, talk_s, turns} for gallery-mapped labels."""
    from pawn_diarize.core.database import get_engine, init_db  # noqa: PLC0415
    from pawn_diarize.core.vault_transcript import (  # noqa: PLC0415
        _display_name,
        load_session_segments,
    )

    engine = get_engine(cfg.db_dsn)
    init_db(engine)
    segments, name_lookup = load_session_segments(session_id, engine)
    gallery = SpeakerGallery(cfg.db_dsn, config=cfg.speakers)
    smap = gallery.load_session_map(session_id)

    # local_label → speaker_id for labels that have a gallery person.
    label_to_sid: dict[str, str] = {}
    for local, row in smap.items():
        if row.speaker_id:
            label_to_sid[local] = row.speaker_id

    stats: dict[str, dict[str, Any]] = {}
    for seg in segments:
        if not (seg.get("text") or "").strip():
            continue
        label = seg.get("label") or ""
        sid = label_to_sid.get(label)
        if not sid:
            # Also accept when display name matches a gallery person (manual relabel
            # without session_speaker_map speaker_id). Prefer map when present.
            name = _display_name(seg, name_lookup)
            if name.startswith("SPEAKER_"):
                continue
            sp = gallery.find_speaker_by_name(name)
            if sp is None:
                continue
            sid = sp.id
        sp = gallery.get_speaker(sid)
        display = (sp.display_name if sp else None) or _display_name(seg, name_lookup)
        bucket = stats.setdefault(
            sid,
            {
                "display_name": display,
                "talk_s": 0.0,
                "turns": 0,
                "aliases": list(sp.aliases or []) if sp else [],
            },
        )
        bucket["talk_s"] += max(0.0, float(seg["end_time"]) - float(seg["start_time"]))
        bucket["turns"] += 1
    return stats


def _source_wiki(cfg: AgentConfig, session_id: str) -> str:
    try:
        from pawn_core.vault_db import get_vault_note  # noqa: PLC0415

        row = get_vault_note(cfg.db_dsn, session_id)
    except Exception:
        return ""
    if row is None or not getattr(row, "key", None):
        return ""
    return f"[[{row.key}]]"


def _analysis_wiki(cfg: AgentConfig, session_id: str) -> str:
    """Best-effort analysis note link when one exists on disk mapping."""
    try:
        from pawn_core.vault import resolve_path_template  # noqa: PLC0415
        from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

        vault = cfg.vault
        key = resolve_path_template(
            vault.analysis_path_template,
            agent_root=vault.agent_root,
            session_id=session_id,
            title=session_id,
        )
        store = vault_store_from_config(cfg)
        store.read(key)
        return f"[[{key}]]"
    except Exception:
        return ""


def _format_clock(seconds: float) -> str:
    total = int(max(0.0, seconds))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def apply_people_updates(
    cfg: AgentConfig,
    updates: list[dict[str, Any]],
    *,
    store: Any = None,
) -> list[str]:
    """Apply structured updates to People notes + optional gallery aliases/notes.

    Each update may include: speaker_id, display_name, facts, aliases, tags,
    summary, appearance_line, mirror_gallery_notes (bool).
    """
    vault = store if store is not None else vault_store_from_config(cfg)
    gallery = SpeakerGallery(cfg.db_dsn, config=cfg.speakers)
    me_names = {n.strip().lower() for n in (cfg.coworker.me or []) if n.strip()}
    written: list[str] = []
    people = _people_cfg(cfg)
    auto_tags = bool(getattr(people, "auto_tags", True)) if people else True

    for upd in updates:
        sid = str(upd.get("speaker_id") or "").strip()
        if not sid:
            continue
        sp = gallery.get_speaker(sid) or gallery.find_speaker_by_name(sid)
        if sp is None:
            logger.info("people update skipped — unknown speaker %r", sid)
            continue
        display = str(upd.get("display_name") or sp.display_name).strip()
        is_me = display.lower() in me_names or sp.display_name.lower() in me_names
        ensure_person_note(
            cfg,
            speaker_id=sp.id,
            display_name=display,
            aliases=sp.aliases or [],
            me=is_me,
            store=vault,
        )
        tags = list(upd.get("tags") or []) if auto_tags else []
        key = update_person_note(
            cfg,
            speaker_id=sp.id,
            display_name=display,
            summary=upd.get("summary") or None,
            aliases=upd.get("aliases") or None,
            tags=tags or None,
            add_facts=upd.get("facts") or None,
            add_appearances=[upd["appearance_line"]] if upd.get("appearance_line") else None,
            me=is_me,
            store=vault,
        )
        written.append(key)

        # Keep gallery aliases / short card in sync when requested.
        new_aliases = list(sp.aliases or [])
        for a in upd.get("aliases") or []:
            a = str(a).strip()
            if a and a.lower() not in {x.lower() for x in new_aliases}:
                new_aliases.append(a)
        card = (upd.get("summary") or "").strip()
        if new_aliases != list(sp.aliases or []) or (card and upd.get("mirror_gallery_notes")):
            gallery.update_speaker(
                sp.id,
                aliases=new_aliases if new_aliases != list(sp.aliases or []) else None,
                notes=card if card and upd.get("mirror_gallery_notes") else None,
            )
    return written


async def refresh_people_for_session(
    cfg: AgentConfig,
    session_id: str,
    *,
    store: Any = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run one people refresh pass for a diarization session.

    Steps:
    1. Resolve gallery speakers present in the session.
    2. Ensure stubs + append Appearance lines (low risk).
    3. LLM extract durable facts.
    4. Apply or propose under autonomy ``people_refresh``.
    """
    people = _people_cfg(cfg)
    if people is None:
        return {"skipped": "no people config"}
    if not force and not getattr(cfg.coworker, "enabled", False):
        logger.info("speakers_refresh for %s skipped; coworker disabled", session_id)
        return {"skipped": "coworker disabled"}
    if not force and not getattr(people, "refresh_after_session", True):
        return {"skipped": "refresh_after_session false"}

    # De-dupe: skip when we already completed a speakers_refresh for this session
    # in the last day (covers re-queued chain messages). ``force`` bypasses.
    fp = fingerprint(f"people_refresh:{session_id}", "people")
    if not force and _already_refreshed(cfg, session_id):
        logger.info("speakers_refresh already done for %s — skip", session_id)
        return {"skipped": "duplicate"}

    run_id = create_agent_run(
        cfg.db_dsn,
        source="coworker",
        command="speakers_refresh",
        prompt=f"people_refresh:{session_id}",
        session_id=session_id,
        model=getattr(cfg, "chat_model_id", None) or "coworker",
    )
    update_agent_run(cfg.db_dsn, run_id, "running")

    try:
        vault = store if store is not None else vault_store_from_config(cfg)
        stats = _talk_stats(cfg, session_id)
        if not stats:
            update_agent_run(cfg.db_dsn, run_id, "completed", response="no gallery speakers")
            return {"session_id": session_id, "speakers": 0, "applied": 0, "proposed": 0}

        source_wiki = _source_wiki(cfg, session_id)
        analysis_wiki = _analysis_wiki(cfg, session_id)
        day = date.today().isoformat()
        create_stubs = bool(getattr(people, "create_stubs", True))

        # Phase A — stubs + appearances (always when refresh runs).
        appearance_updates: list[dict[str, Any]] = []
        roster: dict[str, str] = {}
        for sid, info in stats.items():
            roster[sid] = info["display_name"]
            if create_stubs:
                ensure_person_note(
                    cfg,
                    speaker_id=sid,
                    display_name=info["display_name"],
                    aliases=info.get("aliases") or [],
                    me=info["display_name"].lower()
                    in {n.strip().lower() for n in (cfg.coworker.me or [])},
                    store=vault,
                )
            line = f"{source_wiki or session_id} — {_format_clock(info['talk_s'])} talk · {day}"
            appearance_updates.append(
                {
                    "speaker_id": sid,
                    "display_name": info["display_name"],
                    "appearance_line": line,
                }
            )
            if analysis_wiki:
                appearance_updates.append(
                    {
                        "speaker_id": sid,
                        "display_name": info["display_name"],
                        "appearance_line": f"{analysis_wiki} · {day}",
                    }
                )
        apply_people_updates(cfg, appearance_updates, store=vault)

        # Phase B — durable facts via LLM.
        from pawn_agent.utils.transcript import fetch_transcript  # noqa: PLC0415

        transcript = fetch_transcript(cfg, session_id)
        if transcript.startswith("Error") or transcript.startswith("No transcript"):
            update_agent_run(cfg.db_dsn, run_id, "completed", response="no transcript for facts")
            return {
                "session_id": session_id,
                "speakers": len(stats),
                "appearances": len(appearance_updates),
                "applied": 0,
                "proposed": 0,
            }

        extracted = await extract_people_facts(cfg, transcript, roster)
        # Attach source links to facts.
        fact_updates: list[dict[str, Any]] = []
        for entry in extracted:
            facts = []
            for f in entry.get("facts") or []:
                text = f.strip()
                if source_wiki and source_wiki not in text:
                    text = f"{day}: {text} ({source_wiki})"
                else:
                    text = f"{day}: {text}"
                facts.append(text)
            fact_updates.append(
                {
                    "speaker_id": entry["speaker_id"],
                    "display_name": roster.get(entry["speaker_id"], entry["speaker_id"]),
                    "facts": facts,
                    "aliases": entry.get("aliases") or [],
                    "tags": entry.get("tags") or [],
                    "summary": entry.get("summary") or "",
                    "mirror_gallery_notes": bool(entry.get("summary")),
                }
            )

        if not fact_updates:
            update_agent_run(
                cfg.db_dsn,
                run_id,
                "completed",
                response=f"appearances={len(appearance_updates)} facts=0",
            )
            return {
                "session_id": session_id,
                "speakers": len(stats),
                "appearances": len(appearance_updates),
                "applied": 0,
                "proposed": 0,
            }

        # Appearances already wrote outside Pawn/; fact writes use the same
        # dedicated path. Policy treats people_refresh as allowlisted action_kind
        # (not generic note_write), so writes_outside_pawn stays False here.
        decision = decide(cfg, "people_refresh", writes_outside_pawn=False)
        if decision.decision == "deny":
            itemdb.record_decision(
                cfg.db_dsn,
                event_kind="people_refresh",
                policy_decision="deny",
                proposed_action=f"facts for {session_id}",
                outcome=decision.reason,
            )
            update_agent_run(cfg.db_dsn, run_id, "completed", response=f"denied:{decision.reason}")
            return {
                "session_id": session_id,
                "speakers": len(stats),
                "appearances": len(appearance_updates),
                "applied": 0,
                "proposed": 0,
                "denied": decision.reason,
            }

        if decision.decision == "needs_approval":
            await _propose_people_update(cfg, session_id, fact_updates, source_wiki, vault, fp)
            update_agent_run(cfg.db_dsn, run_id, "completed", response="proposed")
            return {
                "session_id": session_id,
                "speakers": len(stats),
                "appearances": len(appearance_updates),
                "applied": 0,
                "proposed": len(fact_updates),
            }

        keys = apply_people_updates(cfg, fact_updates, store=vault)
        update_agent_run(
            cfg.db_dsn,
            run_id,
            "completed",
            response=f"applied={len(keys)} appearances={len(appearance_updates)}",
        )
        return {
            "session_id": session_id,
            "speakers": len(stats),
            "appearances": len(appearance_updates),
            "applied": len(keys),
            "proposed": 0,
            "keys": keys,
        }
    except Exception as exc:
        update_agent_run(cfg.db_dsn, run_id, "failed", error=str(exc)[:2000])
        raise


async def _propose_people_update(
    cfg: AgentConfig,
    session_id: str,
    updates: list[dict[str, Any]],
    source_wiki: str,
    store: Any,
    fp: str,
) -> None:
    """File a coworker item ``kind=people_update`` for Matrix/HTTP approve."""
    lines = []
    for upd in updates:
        sid = upd["speaker_id"]
        wiki = f"[[{people_wiki_target(cfg, sid)}]]"
        lines.append(f"**{upd.get('display_name') or sid}** ({wiki})")
        for f in upd.get("facts") or []:
            lines.append(f"- fact: {f}")
        for a in upd.get("aliases") or []:
            lines.append(f"- alias: {a}")
        for t in upd.get("tags") or []:
            lines.append(f"- tag: {t}")
        if upd.get("summary"):
            lines.append(f"- summary: {upd['summary']}")
    text = f"People updates from session {session_id}:\n" + "\n".join(lines)
    row = itemdb.insert_item(
        cfg.db_dsn,
        source_kind="session",
        source_ref=session_id,
        kind="people_update",
        text=text[:4000],
        thread="People",
        fingerprint=fp,
        interrupt=True,
        reason="people_refresh needs approval",
        payload={"action_kind": "people_refresh", "updates": updates},
        status="new",
    )
    from pawn_agent.core.coworker.actions import item_note_key  # noqa: PLC0415

    note_key = item_note_key(cfg, short_id=row["short_id"], text=text[:4000])
    body = render_item_note(
        item_id=row["id"],
        short_id=row["short_id"],
        status="notified",
        kind="people_update",
        text=text[:4000],
        thread="People",
        source_link=source_wiki,
        reason="Approve to write these facts into People/ notes.",
        interrupt=True,
    )
    try:
        store.write(note_key, body)
        itemdb.update_item(cfg.db_dsn, row["id"], status="notified", note_key=note_key)
    except Exception as exc:
        logger.warning("people proposal note failed: %s", exc)
        itemdb.update_item(cfg.db_dsn, row["id"], status="notified")

    try:
        from pawn_server.core.notify import notify  # noqa: PLC0415

        await notify(
            cfg,
            kind="coworker_item",
            text=(
                f"Pawn: people update from {session_id}\n"
                f"Reply: approve {row['short_id']} | reject {row['short_id']}"
            ),
            link=note_key,
            item_id=row["id"],
        )
    except Exception as exc:
        logger.debug("people proposal notify skipped: %s", exc)


async def propose_people_from_chat(
    cfg: AgentConfig,
    text: str,
    *,
    source_ref: str,
    store: Any = None,
) -> Optional[dict[str, Any]]:
    """Phase 4: after chat, propose durable person facts (never auto-write).

    Only runs when ``coworker.people.refresh_after_chat`` is True and coworker
    is enabled. Resolves roster from the full Speakers gallery (active).
    """
    people = _people_cfg(cfg)
    if not getattr(cfg.coworker, "enabled", False):
        return None
    if people is None or not getattr(people, "refresh_after_chat", False):
        return None
    if not (text or "").strip():
        return None

    gallery = SpeakerGallery(cfg.db_dsn, config=cfg.speakers)
    roster = {sp.id: sp.display_name for sp in gallery.list_speakers()}
    if not roster:
        return None

    # Cheap gate: at least one display name / id must appear in the text.
    lowered = text.lower()
    if not any(name.lower() in lowered or sid.lower() in lowered for sid, name in roster.items()):
        return None

    try:
        extracted = await extract_people_facts(cfg, text, roster)
    except Exception as exc:
        logger.warning("chat people extract failed: %s", exc)
        return None
    if not extracted:
        return None

    day = date.today().isoformat()
    updates = []
    for entry in extracted:
        facts = [f"{day}: {f}" for f in entry.get("facts") or []]
        updates.append(
            {
                "speaker_id": entry["speaker_id"],
                "display_name": roster.get(entry["speaker_id"], entry["speaker_id"]),
                "facts": facts,
                "aliases": entry.get("aliases") or [],
                "tags": entry.get("tags") or [],
                "summary": entry.get("summary") or "",
                "mirror_gallery_notes": False,
            }
        )
    vault = store if store is not None else vault_store_from_config(cfg)
    fp = fingerprint(f"people_chat:{source_ref}:{updates[0]['speaker_id']}", "people")
    await _propose_people_update(cfg, source_ref, updates, "", vault, fp)
    return {"proposed": len(updates), "source_ref": source_ref}


def gallery_people_wiki_lookup(cfg: AgentConfig, session_id: str) -> dict[str, str]:
    """Map display_name → wiki target for Speakers table linking.

    Used by vault transcript push. Only names with a gallery speaker_id.
    """
    gallery = SpeakerGallery(cfg.db_dsn, config=cfg.speakers)
    smap = gallery.load_session_map(session_id)
    out: dict[str, str] = {}
    for row in smap.values():
        if not row.speaker_id:
            continue
        sp = gallery.get_speaker(row.speaker_id)
        name = (row.display_name or (sp.display_name if sp else None) or "").strip()
        if not name:
            continue
        out[name] = people_wiki_target(cfg, row.speaker_id)
    # Also map by gallery display_name for relabelled sessions.
    for sp in gallery.list_speakers():
        out.setdefault(sp.display_name, people_wiki_target(cfg, sp.id))
    return out

"""Brainmap board API (``/v1/portal/brainmaps/*``) — see docs/PORTAL.md "Brainmap".

Thin HTTP layer over :mod:`pawn_server.core.brainmap`. Blocking DB work runs
in a thread so a slow query never stalls SSE or long-poll handlers.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from pawn_server.core import boards
from pawn_server.core import brainmap as bm


class TieBody(BaseModel):
    from_: str = Field(alias="from")
    to: str
    role: str = "mentions"

    model_config = {"populate_by_name": True}


class SuggestionBody(BaseModel):
    verdict: str  # confirm | snip
    role: str = ""


class LassoBody(BaseModel):
    node_ids: list[str]
    title: Optional[str] = None
    into: Optional[str] = None


class LayoutBody(BaseModel):
    positions: dict[str, Any] = Field(default_factory=dict)
    viewport: Optional[dict[str, Any]] = None


def _split(value: Optional[str]) -> Optional[list[str]]:
    parts = [p.strip() for p in (value or "").split(",") if p.strip()]
    return parts or None


async def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except bm.BrainmapError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


def _store(cfg: Any) -> Any:
    try:
        from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

        return vault_store_from_config(cfg)
    except Exception:  # no vault configured: subjects still exist in the DB
        return None


async def _require_board(cfg: Any, board: str) -> None:
    if not await asyncio.to_thread(boards.exists, cfg.db_dsn, board):
        raise HTTPException(status_code=404, detail=f"no such board {board!r}")


def build_router(get_cfg: Callable[..., Any], require_token: Callable[..., Any]) -> APIRouter:
    router = APIRouter(
        prefix="/v1/portal/brainmaps", tags=["brainmap"], dependencies=[Depends(require_token)]
    )

    @router.get("/{board}")
    async def board_graph(
        board: str,
        statuses: Optional[str] = None,
        kinds: Optional[str] = None,
        focus: str = "",
        limit: int = 150,
        cfg: Any = Depends(get_cfg),
    ) -> dict:
        return await _call(
            bm.build_board,
            cfg,
            board,
            statuses=_split(statuses),
            kinds=_split(kinds),
            focus=focus,
            limit=limit,
        )

    @router.get("/{board}/spill/{subject_id}")
    async def board_spill(board: str, subject_id: str, cfg: Any = Depends(get_cfg)) -> dict:
        return await _call(bm.spill, cfg, subject_id)

    @router.post("/{board}/strings")
    async def board_tie(board: str, body: TieBody, cfg: Any = Depends(get_cfg)) -> dict:
        if body.from_.startswith("note:") or body.to.startswith("note:"):
            return await _call(boards.tie_note, cfg.db_dsn, board, body.from_, body.to)
        return await _call(bm.tie, cfg, body.from_, body.to, body.role)

    @router.delete("/{board}/strings/{edge_id}")
    async def board_cut(board: str, edge_id: str, cfg: Any = Depends(get_cfg)) -> dict:
        if edge_id.startswith("notesrc:"):
            return await _call(boards.cut_note_source, cfg.db_dsn, board, edge_id)
        return await _call(bm.cut, cfg, edge_id)

    @router.post("/{board}/suggestions/{suggestion_id}")
    async def board_suggestion(
        board: str, suggestion_id: str, body: SuggestionBody, cfg: Any = Depends(get_cfg)
    ) -> dict:
        if suggestion_id.startswith("prop:"):
            res = await _call(
                boards.decide_proposal,
                cfg,
                board,
                suggestion_id[5:],
                body.verdict,
                store=_store(cfg),
            )
            return {"edge": None, "receipt": res["receipt"], "reload": True}
        return await _call(bm.suggestion_verdict, cfg, suggestion_id, body.verdict, body.role)

    @router.post("/{board}/lasso")
    async def board_lasso(board: str, body: LassoBody, cfg: Any = Depends(get_cfg)) -> dict:
        return await _call(
            bm.lasso,
            cfg,
            body.node_ids,
            title=body.title or "",
            into=body.into or "",
            store=_store(cfg),
        )

    @router.get("/{board}/layout")
    async def board_layout_get(board: str, cfg: Any = Depends(get_cfg)) -> dict:
        await _require_board(cfg, board)
        return await _call(bm.get_layout, cfg.db_dsn, board)

    @router.put("/{board}/layout")
    async def board_layout_put(board: str, body: LayoutBody, cfg: Any = Depends(get_cfg)) -> dict:
        await _require_board(cfg, board)
        return await _call(bm.put_layout, cfg.db_dsn, board, body.positions, body.viewport)

    return router

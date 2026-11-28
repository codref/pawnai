"""User boards API (``/v1/portal/boards/*``) — see docs/PORTAL.md "Boards".

CRUD, membership (pins, remove), the board chat and proposals. The graph
itself and the gestures stay on ``/v1/portal/brainmaps/{board}*``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from pawn_server.core import boards
from pawn_server.core.brainmap_api import _call, _store


class BoardCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    kind: str = "manual"
    spec: dict[str, Any] = Field(default_factory=dict)
    nodes: list[str] = Field(default_factory=list)
    folder_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    chat_id: Optional[str] = None
    question: str = ""
    subject_id: Optional[str] = None


class BoardPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = None
    folder_id: Optional[str] = None
    unfile: bool = False
    tags: Optional[list[str]] = None
    pinned: Optional[bool] = None
    archived: Optional[bool] = None
    spec: Optional[dict[str, Any]] = None


class NodesBody(BaseModel):
    node_ids: list[str]


class VerdictBody(BaseModel):
    verdict: str  # accept | reject


def build_router(get_cfg: Callable[..., Any], require_token: Callable[..., Any]) -> APIRouter:
    router = APIRouter(
        prefix="/v1/portal/boards", tags=["boards"], dependencies=[Depends(require_token)]
    )

    @router.get("")
    async def boards_list(archived: bool = False, cfg: Any = Depends(get_cfg)) -> dict:
        return {"boards": await _call(boards.list_boards, cfg.db_dsn, archived=archived)}

    @router.post("", status_code=201)
    async def boards_create(body: BoardCreate, cfg: Any = Depends(get_cfg)) -> dict:
        return await _call(
            boards.create_board,
            cfg,
            title=body.title,
            kind=body.kind,
            spec=body.spec,
            nodes=body.nodes,
            folder_id=body.folder_id,
            tags=body.tags,
            chat_id=body.chat_id,
            question=body.question,
            subject_id=body.subject_id,
            store=_store(cfg) if body.kind == "question" else None,
        )

    @router.get("/{board}")
    async def boards_get(board: str, cfg: Any = Depends(get_cfg)) -> dict:
        return await _call(boards.get_board, cfg.db_dsn, board)

    @router.patch("/{board}")
    async def boards_patch(board: str, body: BoardPatch, cfg: Any = Depends(get_cfg)) -> dict:
        return await _call(
            boards.patch_board, cfg.db_dsn, board, **body.model_dump(exclude_none=True)
        )

    @router.delete("/{board}")
    async def boards_delete(board: str, cfg: Any = Depends(get_cfg)) -> dict:
        return await _call(boards.delete_board, cfg.db_dsn, board)

    @router.post("/{board}/pins")
    async def boards_pin(board: str, body: NodesBody, cfg: Any = Depends(get_cfg)) -> dict:
        return await _call(boards.add_pins, cfg.db_dsn, board, body.node_ids)

    @router.post("/{board}/remove")
    async def boards_remove(board: str, body: NodesBody, cfg: Any = Depends(get_cfg)) -> dict:
        return await _call(boards.remove_nodes, cfg.db_dsn, board, body.node_ids)

    @router.post("/{board}/seen")
    async def boards_seen(board: str, cfg: Any = Depends(get_cfg)) -> dict:
        return await _call(boards.mark_seen, cfg.db_dsn, board)

    @router.post("/{board}/chat")
    async def boards_chat(board: str, cfg: Any = Depends(get_cfg)) -> dict:
        return await _call(boards.ensure_chat, cfg.db_dsn, board)

    @router.get("/{board}/proposals")
    async def boards_proposals(
        board: str, status: str = "proposed", cfg: Any = Depends(get_cfg)
    ) -> dict:
        return {
            "proposals": await _call(
                boards.list_proposals, cfg.db_dsn, board, status=status or None
            )
        }

    @router.post("/{board}/proposals/{proposal_id}")
    async def boards_decide(
        board: str, proposal_id: str, body: VerdictBody, cfg: Any = Depends(get_cfg)
    ) -> dict:
        return await _call(
            boards.decide_proposal, cfg, board, proposal_id, body.verdict, store=_store(cfg)
        )

    return router

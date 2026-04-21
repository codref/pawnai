"""LangGraph AsyncPostgresStore singleton with sentence-transformer embeddings.

This module owns the single shared AsyncPostgresStore instance used by the
entire pawn-agent runtime.  All memory and knowledge is stored here; the old
rag_sources / text_chunks pipeline has been removed.

Namespaces
----------
NS_MEMORIES  = ("memories",)
    Persistent agent memories saved via the ``memorize`` tool.

NS_SESSIONS  = ("knowledge", "sessions")
    Session analysis chunks stored via the ``vectorize`` tool.

NS_SIYUAN    = ("knowledge", "siyuan")
    SiYuan page blocks stored via the ``vectorize`` tool.

Usage
-----
    from pawn_agent.core.store import NS_MEMORIES, get_store

    store = await get_store(cfg)
    await store.aput(NS_MEMORIES, key, {"text": "...", "tags": []})
    results = await store.asearch(NS_MEMORIES, query="...", limit=5)
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ── Namespace constants ────────────────────────────────────────────────────────
NS_MEMORIES: tuple[str, ...] = ("memories",)
NS_SESSIONS: tuple[str, ...] = ("knowledge", "sessions")
NS_SIYUAN: tuple[str, ...] = ("knowledge", "siyuan")

# ── Module-level singleton ─────────────────────────────────────────────────────
_store_instance: Any = None


# ── Embeddings wrapper ─────────────────────────────────────────────────────────


class STEmbeddings:
    """LangChain-compatible Embeddings wrapper around sentence-transformers.

    Implements ``embed_documents``, ``embed_query``, ``aembed_documents``, and
    ``aembed_query`` so it can be passed to ``AsyncPostgresStore`` as the
    ``index.embed`` value.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        truncate_dim: Optional[int] = None,
        local_files_only: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._truncate_dim = truncate_dim
        self._local_files_only = local_files_only
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            try:
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                    truncate_dim=self._truncate_dim,
                    local_files_only=self._local_files_only,
                )
            except Exception:
                # Fall back to network download on first cold start.
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                    truncate_dim=self._truncate_dim,
                )
            logger.info(
                "Loaded embedding model %r (device=%s, dim=%s)",
                self._model_name,
                self._device,
                self._truncate_dim,
            )
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_model().encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._get_model().encode(text, show_progress_bar=False).tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _strip_driver(dsn: str) -> str:
    """Convert ``postgresql+psycopg://`` DSN to plain ``postgresql://``."""
    return dsn.replace("postgresql+psycopg://", "postgresql://", 1)


# ── Singleton factory ──────────────────────────────────────────────────────────


async def get_store(cfg: Any) -> Any:
    """Return the singleton :class:`AsyncPostgresStore`, creating it if needed.

    The first call initialises the store and calls ``store.setup()`` to create
    the ``store`` and ``store_vectors`` tables if they do not exist.  All
    subsequent calls return the same instance without any I/O.

    Args:
        cfg: Any config object that exposes ``db_dsn``, ``embed_model``,
            ``embed_device``, ``embed_dim``, and ``embed_local_files_only``.
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    from langgraph.store.postgres import AsyncPostgresStore  # noqa: PLC0415

    embed = STEmbeddings(
        model_name=cfg.embed_model,
        device=cfg.embed_device,
        truncate_dim=cfg.embed_dim if cfg.embed_dim else None,
        local_files_only=cfg.embed_local_files_only,
    )

    dims = cfg.embed_dim or 1024
    conn_str = _strip_driver(cfg.db_dsn)

    store = AsyncPostgresStore.from_conn_string(
        conn_str,
        index={
            "dims": dims,
            "embed": embed,
            "fields": ["text"],
        },
    )
    await store.setup()
    _store_instance = store
    logger.info("AsyncPostgresStore ready (dims=%d)", dims)
    return _store_instance

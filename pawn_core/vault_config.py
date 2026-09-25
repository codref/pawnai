"""Build :class:`VaultStore` from application config."""

from __future__ import annotations

from typing import Any

from pawn_core.vault import VaultStore


def vault_store_from_config(cfg: Any) -> VaultStore:
    """Construct a :class:`VaultStore` from ``cfg.vault`` (or legacy attrs)."""
    vault = getattr(cfg, "vault", None)
    if vault is None:
        raise ValueError("vault configuration is missing")
    bucket = getattr(vault, "bucket", "") or ""
    if not bucket:
        raise ValueError("vault.bucket must be set")
    return VaultStore(
        bucket=bucket,
        prefix=getattr(vault, "prefix", "") or "",
        endpoint_url=getattr(vault, "endpoint_url", None),
        access_key=getattr(vault, "access_key", None),
        secret_key=getattr(vault, "secret_key", None),
        region=getattr(vault, "region", None),
        path_style=bool(getattr(vault, "path_style", True)),
        verify_ssl=bool(getattr(vault, "verify_ssl", True)),
        agent_root=getattr(vault, "agent_root", "Pawn") or "Pawn",
    )

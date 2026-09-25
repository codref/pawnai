"""Build :class:`VaultStore` from application config."""

from __future__ import annotations

from typing import Any

from pawn_core.vault import VaultStore


def _vault_s3(vault: Any) -> Any:
    """Return the dedicated vault S3 credentials section."""
    s3 = getattr(vault, "s3", None)
    if s3 is not None:
        return s3
    # Legacy flat vault.* credentials (pre vault.s3 nesting).
    return vault


def vault_store_from_config(cfg: Any) -> VaultStore:
    """Construct a :class:`VaultStore` from ``cfg.vault.s3`` credentials."""
    vault = getattr(cfg, "vault", None)
    if vault is None:
        raise ValueError("vault configuration is missing")
    s3 = _vault_s3(vault)
    bucket = getattr(s3, "bucket", "") or ""
    if not bucket:
        raise ValueError("vault.s3.bucket must be set")
    return VaultStore(
        bucket=bucket,
        prefix=getattr(s3, "prefix", "") or "",
        endpoint_url=getattr(s3, "endpoint_url", None),
        access_key=getattr(s3, "access_key", None),
        secret_key=getattr(s3, "secret_key", None),
        region=getattr(s3, "region", None),
        path_style=bool(getattr(s3, "path_style", True)),
        verify_ssl=bool(getattr(s3, "verify_ssl", True)),
        agent_root=getattr(vault, "agent_root", "Pawn") or "Pawn",
    )

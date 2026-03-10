"""Tool executor: S3 download and temp-file cleanup.

Functions registered as:
  - ``s3.download``
  - ``s3.cleanup``
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, Dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pawn_agent.config import AgentConfig


async def download(params: Dict[str, Any], cfg: "AgentConfig") -> Dict[str, Any]:
    """Download an audio file from S3 (or pass through a local path).

    Args:
        params: Must contain ``audio_path`` (``s3://`` URI or local path).
        cfg:    Agent configuration (provides S3 credentials).

    Returns:
        ``{"local_path": str, "temp_dir": str | None}``
        ``temp_dir`` is ``None`` when the input was already a local path.
    """
    audio_path: str = params["audio_path"]

    if not audio_path.startswith("s3://"):
        # Already local — nothing to download
        return {"local_path": audio_path, "temp_dir": None}

    def _do_download() -> tuple[str, str]:
        from pawnai.core.s3 import S3Client  # lazy import

        s3_cfg = cfg.get_s3_config()
        client = S3Client(
            endpoint_url=s3_cfg.get("endpoint_url", ""),
            bucket_name=s3_cfg.get("bucket_name", ""),
            access_key=s3_cfg.get("access_key", s3_cfg.get("aws_access_key_id", "")),
            secret_key=s3_cfg.get("secret_key", s3_cfg.get("aws_secret_access_key", "")),
            region=s3_cfg.get("region", "us-east-1"),
            verify_ssl=bool(s3_cfg.get("verify_ssl", True)),
        )
        tmp = tempfile.mkdtemp(prefix="pawn_agent_")
        local_path = client.download(audio_path, tmp)
        return local_path, tmp

    loop = asyncio.get_event_loop()
    local_path, temp_dir = await loop.run_in_executor(None, _do_download)
    logger.info("Downloaded %s → %s", audio_path, local_path)
    return {"local_path": local_path, "temp_dir": temp_dir}


async def cleanup(params: Dict[str, Any], cfg: "AgentConfig") -> Dict[str, Any]:
    """Remove a temporary directory created by :func:`download`.

    Args:
        params: May contain ``temp_dir`` (path to remove). If ``None`` or
                absent, this is a no-op.

    Returns:
        ``{}``
    """
    temp_dir: Any = params.get("temp_dir")
    if temp_dir:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug("Cleaned up temp dir: %s", temp_dir)
        except Exception as exc:
            logger.warning("Failed to clean up temp dir %r: %s", temp_dir, exc)
    return {}

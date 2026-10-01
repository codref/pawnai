"""Images on a chat turn: note embeds, size caps, and vision-model gating.

Sallm captions each new image and can recall that caption later. This module
decides which bytes to hand it. Context images already captioned in the
session are left out unless the user asks for a fresh read. Question images
(a picture dropped in the chat) are always included, and sallm reuses the
stored caption for the same bytes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import unquote

from pawn_core.vault import normalize_vault_key

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
# One model call shows one question image and one context image. Extra images
# on the turn are still captioned. A later message fills free slots with the
# next embeds this session has not captioned yet.
MAX_TURN_IMAGES = 4
MAX_CANDIDATE_IMAGES = 12
MAX_IMAGE_BYTES = 4 * 1024 * 1024

_WIKI_EMBED = re.compile(r"!\[\[([^\]|#|]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
_MD_EMBED = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_MD_TITLE = re.compile(r"""\s+["'].*["']\s*$""")


class ImageRejected(ValueError):
    """The turn's images cannot be sent."""


@dataclass(frozen=True)
class TurnImage:
    """One image staged for ``Agent.ask``."""

    filename: str
    media_type: str
    data: bytes
    role: str  # question | context


def suffix_for(filename: str, media_type: str = "") -> str:
    """Image suffix from the filename, or from a mime type."""
    suffix = Path(filename or "").suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return suffix
    mime = (media_type or "").split(";", 1)[0].strip().lower()
    for ext, known in MIME_BY_SUFFIX.items():
        if mime == known:
            return ext
    return ""


def mime_for(filename: str, media_type: str = "") -> str:
    suffix = suffix_for(filename, media_type)
    if suffix:
        return MIME_BY_SUFFIX[suffix]
    return (media_type or "").split(";", 1)[0].strip().lower()


def is_image_name(filename: str, media_type: str = "") -> bool:
    return bool(suffix_for(filename, media_type))


def decode_chat_image(
    *,
    filename: str,
    media_type: str,
    data_base64: str,
    role: str,
) -> TurnImage:
    """Decode one API image. Raises ``ImageRejected`` when it cannot be used."""
    kind = (role or "").strip().lower()
    if kind not in {"question", "context"}:
        raise ImageRejected("image role must be question or context")
    name = Path((filename or "").replace("\\", "/")).name
    if not is_image_name(name, media_type):
        raise ImageRejected(f"not an image: {name or filename or 'file'}")
    try:
        data = base64.b64decode((data_base64 or "").strip(), validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ImageRejected(f"could not read image {name or filename}") from exc
    if not data:
        raise ImageRejected(f"empty image: {name or filename}")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageRejected(f"{name or filename} is larger than 4MB")
    return TurnImage(
        filename=name or f"image{suffix_for(name, media_type)}",
        media_type=mime_for(name, media_type),
        data=data,
        role=kind,
    )


def select_turn_images(
    images: Sequence[TurnImage],
    *,
    known: set[str],
    force: bool = False,
) -> list[TurnImage]:
    """Question images first, then context embeds not yet captioned, up to 4.

    ``known`` is the set of SHA-256 digests this session already captioned.
    ``force`` keeps context images even when they are known (a re-read).
    """
    questions = [image for image in images if image.role == "question"]
    contexts = [image for image in images if image.role != "question"]
    chosen: list[TurnImage] = []
    for image in questions:
        if len(image.data) > MAX_IMAGE_BYTES:
            raise ImageRejected(f"{image.filename} is larger than 4MB")
        if not image.data:
            continue
        if len(chosen) >= MAX_TURN_IMAGES:
            break
        chosen.append(image)
    for image in contexts:
        if len(chosen) >= MAX_TURN_IMAGES:
            break
        if not image.data or len(image.data) > MAX_IMAGE_BYTES:
            logger.info("skipping context image %s (empty or over 4MB)", image.filename)
            continue
        digest = hashlib.sha256(image.data).hexdigest()
        if not force and digest in known:
            continue
        chosen.append(image)
    return chosen


def note_image_targets(markdown: str) -> list[str]:
    """Image embed targets in document order. Remote URLs are skipped."""
    text = markdown or ""
    found: list[str] = []
    seen: set[str] = set()
    located: list[tuple[int, str]] = []
    for match in _WIKI_EMBED.finditer(text):
        located.append((match.start(), match.group(1)))
    for match in _MD_EMBED.finditer(text):
        located.append((match.start(), match.group(1)))
    located.sort(key=lambda item: item[0])

    def add(raw: str) -> None:
        target = unquote((raw or "").strip())
        if target.startswith("<") and target.endswith(">") and len(target) > 2:
            target = target[1:-1].strip()
        target = _MD_TITLE.sub("", target).strip()
        target = target.removeprefix("./")
        if not target or target.startswith(("#", "http://", "https://", "data:", "mailto:")):
            return
        if not is_image_name(target):
            return
        key = target.replace("\\", "/")
        if key in seen:
            return
        seen.add(key)
        found.append(key)

    for _start, raw in located:
        add(raw)
    return found


def embed_keys(note_path: str, target: str) -> list[str]:
    """Vault keys to try for one embed, note-relative paths first when bare."""
    raw = normalize_vault_key(unquote(target or "").removeprefix("./"))
    if not raw or raw.startswith(("http://", "https://", "data:")):
        return []
    if not is_image_name(raw):
        return []
    note_key = normalize_vault_key(note_path)
    note_dir = note_key.rsplit("/", 1)[0] if "/" in note_key else ""
    keys: list[str] = []

    def add(key: str) -> None:
        cleaned = normalize_vault_key(key)
        if cleaned and cleaned not in keys:
            keys.append(cleaned)

    if "/" in raw:
        add(raw)
        if note_dir:
            add(f"{note_dir}/{raw}")
    else:
        if note_dir:
            add(f"{note_dir}/{raw}")
        add(raw)
    return keys


def load_note_images(store: Any, notes: Sequence[tuple[str, Optional[str]]]) -> list[TurnImage]:
    """Read embedded images from the vault, in document order, up to the candidate cap."""
    if store is None:
        return []
    images: list[TurnImage] = []
    seen: set[str] = set()
    for path, content in notes:
        for target in note_image_targets(content or ""):
            if len(images) >= MAX_CANDIDATE_IMAGES:
                return images
            for key in embed_keys(path, target):
                if key in seen:
                    break
                try:
                    data = store.read_bytes(key)
                except Exception:
                    continue
                if not isinstance(data, (bytes, bytearray)) or not data:
                    continue
                if len(data) > MAX_IMAGE_BYTES:
                    logger.info("skipping vault image %s (over 4MB)", key)
                    seen.add(key)
                    break
                seen.add(key)
                images.append(
                    TurnImage(
                        filename=Path(key).name,
                        media_type=mime_for(key),
                        data=bytes(data),
                        role="context",
                    )
                )
                break
    return images


def stage_images(images: Sequence[TurnImage]) -> tuple[list[Any], Optional[Path]]:
    """Write spaceless temp files and return sallm ``ImageMention`` values."""
    if not images:
        return [], None
    from sallm.attachments import ImageMention  # noqa: PLC0415

    directory = Path(tempfile.mkdtemp(prefix="pawn-vision-"))
    mentions = []
    try:
        for image in images:
            suffix = suffix_for(image.filename, image.media_type) or ".png"
            digest = hashlib.sha256(image.data).hexdigest()
            path = directory / f"{digest}{suffix}"
            path.write_bytes(image.data)
            role = image.role if image.role in {"question", "context"} else "context"
            name = Path(image.filename).name or path.name
            mentions.append(ImageMention(role=role, path=path, filename=name))
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return mentions, directory


def cleanup_staged(directory: Optional[Path]) -> None:
    if directory is not None:
        shutil.rmtree(directory, ignore_errors=True)


def vision_refusal(source: str) -> str:
    """Short reply when the chosen model is not flagged vision-ready."""
    if source == "matrix":
        return (
            "This model can't see images. Use /model to pick a vision-ready model "
            "(/vision lists them)."
        )
    return "This model can't see images. Pick a vision-ready model in the model menu."


def vision_status(cfg: Any) -> str:
    """Plain-text status for ``/vision``."""
    from pawn_agent.utils.model_catalog import (  # noqa: PLC0415
        catalog_entries,
        default_selection,
        get_background_model,
    )

    entries = catalog_entries(cfg)
    default_id = default_selection(cfg).catalog_id
    background = get_background_model(cfg) or default_id
    ready = any(entry.id == background and entry.vision for entry in entries)
    state = "can see images" if ready else "cannot see images"
    lines = [f"Background model {background} {state}."]
    vision_ids = [entry.id for entry in entries if entry.vision]
    if vision_ids:
        lines.append("Vision models: " + ", ".join(vision_ids))
    else:
        lines.append("No vision models are configured. Set vision: true on a model.")
    lines.append("On Matrix, /vision refresh reads the next image in this room again.")
    return "\n".join(lines)


def try_vision_command(cfg: Any, text: str) -> Optional[str]:
    """Reply for ``/vision``. ``/vision refresh`` is armed by the Matrix bot."""
    raw = (text or "").strip()
    if raw == "/vision":
        return vision_status(cfg)
    if raw == "/vision refresh":
        return "Use Re-read images in the composer. On Matrix this reads the next image again."
    return None

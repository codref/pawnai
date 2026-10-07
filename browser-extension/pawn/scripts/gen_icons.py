#!/usr/bin/env python3
"""Write minimal solid-color PNG icons for the extension."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def png(size: int, rgb: tuple[int, int, int] = (0x4A, 0x9E, 0x8E)) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "icons"
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("icon-48.png").write_bytes(png(48))
    root.joinpath("icon-128.png").write_bytes(png(128))
    print(f"Wrote {root / 'icon-48.png'} and {root / 'icon-128.png'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate Pawn Capture toolbar icons (teal tile + white pawn + snip corners)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

# Brand teal used across the extension UI.
TEAL = (0x4A, 0x9E, 0x8E, 255)
WHITE = (255, 255, 255, 255)


def _xy(size: int, x: float, y: float) -> tuple[float, float]:
    """Map 0–100 design coords into the icon inset."""
    pad = size * 0.14
    inner = size - 2 * pad
    return pad + x / 100 * inner, pad + y / 100 * inner


def draw_pawn(draw: ImageDraw.ImageDraw, size: int) -> None:
    """Simple chess-pawn silhouette (head, collar, body, base)."""
    # Head
    hx, hy = _xy(size, 50, 22)
    hr = size * 0.11
    draw.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), fill=WHITE)

    # Collar / neck ring
    x0, y0 = _xy(size, 34, 32)
    x1, y1 = _xy(size, 66, 40)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=max(1, size // 24), fill=WHITE)

    # Body (trapezoid)
    body = [
        _xy(size, 40, 40),
        _xy(size, 60, 40),
        _xy(size, 70, 72),
        _xy(size, 30, 72),
    ]
    draw.polygon(body, fill=WHITE)

    # Base
    bx0, by0 = _xy(size, 24, 72)
    bx1, by1 = _xy(size, 76, 86)
    draw.rounded_rectangle((bx0, by0, bx1, by1), radius=max(1, size // 20), fill=WHITE)


def draw_snip_corners(draw: ImageDraw.ImageDraw, size: int) -> None:
    """Tiny L-brackets suggesting region/selection capture."""
    w = max(2, size // 14)
    arm = max(4, size // 5)
    inset = max(3, size // 10)
    # top-left
    draw.rectangle((inset, inset, inset + arm, inset + w), fill=WHITE)
    draw.rectangle((inset, inset, inset + w, inset + arm), fill=WHITE)
    # bottom-right
    draw.rectangle((size - inset - arm, size - inset - w, size - inset, size - inset), fill=WHITE)
    draw.rectangle((size - inset - w, size - inset - arm, size - inset, size - inset), fill=WHITE)


def png(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = max(1, round(size * 0.04))
    radius = max(4, round(size * 0.22))
    draw.rounded_rectangle(
        (margin, margin, size - 1 - margin, size - 1 - margin),
        radius=radius,
        fill=TEAL,
    )
    draw_pawn(draw, size)
    if size >= 32:
        draw_snip_corners(draw, size)
    return img


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "icons"
    root.mkdir(parents=True, exist_ok=True)
    for size in (16, 48, 128):
        out = root / f"icon-{size}.png"
        png(size).save(out, format="PNG", optimize=True)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

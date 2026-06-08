"""Generate the owned iOS app icon asset catalog.

The icon intentionally avoids player likenesses, league marks, team logos, text,
letters, and numbers. It is a deterministic sports-analytics mark: field-line
geometry plus a projection curve.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "Resources" / "Assets.xcassets"
ICONSET = ASSET_ROOT / "AppIcon.appiconset"


def _hex(rgb: str) -> tuple[int, int, int]:
    rgb = rgb.lstrip("#")
    return tuple(int(rgb[i : i + 2], 16) for i in (0, 2, 4))


def _lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def _gradient(size: int) -> Image.Image:
    top = _hex("#111827")
    bottom = _hex("#05070d")
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        t = y / max(1, size - 1)
        for x in range(size):
            vignette = 1 - min(0.42, math.hypot(x - size / 2, y - size / 2) / (size * 1.25))
            c = tuple(_lerp(top[i], bottom[i], t) for i in range(3))
            px[x, y] = tuple(max(0, min(255, round(ch * vignette))) for ch in c)
    return img


def _draw_icon(size: int = 1024) -> Image.Image:
    scale = 4
    canvas = _gradient(size * scale)
    draw = ImageDraw.Draw(canvas)
    s = size * scale

    emerald = _hex("#22c55e")
    emerald_dim = _hex("#14532d")
    gold = _hex("#facc15")
    white = _hex("#d1fae5")

    # Subtle field grid.
    for i in range(8):
        x = round((0.18 + i * 0.092) * s)
        alpha = 72 if i in (2, 5) else 40
        draw.line([(x, 0.22 * s), (x, 0.82 * s)], fill=(*emerald_dim, alpha), width=3 * scale)
    for i in range(5):
        y = round((0.29 + i * 0.095) * s)
        draw.line([(0.14 * s, y), (0.86 * s, y)], fill=(*emerald_dim, 42), width=2 * scale)

    # Central shield/field plate.
    plate = [
        (0.5 * s, 0.15 * s),
        (0.82 * s, 0.31 * s),
        (0.76 * s, 0.73 * s),
        (0.5 * s, 0.89 * s),
        (0.24 * s, 0.73 * s),
        (0.18 * s, 0.31 * s),
    ]
    shadow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.polygon([(x, y + 22 * scale) for x, y in plate], fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18 * scale))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow)
    draw = ImageDraw.Draw(canvas)
    draw.polygon(plate, fill=(9, 21, 18, 235), outline=(*emerald, 210))
    draw.line(plate + [plate[0]], fill=(*emerald, 230), width=8 * scale, joint="curve")

    # Field yard marks, abstracted to avoid numbers/text.
    for i in range(5):
        y = (0.32 + i * 0.09) * s
        draw.line([(0.33 * s, y), (0.67 * s, y)], fill=(*white, 90), width=4 * scale)
        draw.line(
            [(0.42 * s, y + 0.036 * s), (0.58 * s, y + 0.036 * s)],
            fill=(*white, 50),
            width=3 * scale,
        )

    # Projection curve and nodes.
    points = [
        (0.27 * s, 0.68 * s),
        (0.39 * s, 0.57 * s),
        (0.49 * s, 0.60 * s),
        (0.61 * s, 0.43 * s),
        (0.74 * s, 0.35 * s),
    ]
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.line(points, fill=(*emerald, 170), width=22 * scale, joint="curve")
    glow = glow.filter(ImageFilter.GaussianBlur(10 * scale))
    canvas = Image.alpha_composite(canvas, glow)
    draw = ImageDraw.Draw(canvas)
    draw.line(points, fill=(*emerald, 255), width=10 * scale, joint="curve")
    draw.line(points[-2:], fill=(*gold, 255), width=10 * scale, joint="curve")
    for idx, (x, y) in enumerate(points):
        r = (16 if idx == len(points) - 1 else 12) * scale
        fill = gold if idx == len(points) - 1 else emerald
        draw.ellipse(
            (x - r, y - r, x + r, y + r), fill=(*fill, 255), outline=(*white, 210), width=3 * scale
        )

    # Small football-like analytical marker, generic and mark-free.
    ball_box = (0.37 * s, 0.205 * s, 0.63 * s, 0.31 * s)
    draw.ellipse(ball_box, fill=(92, 54, 20, 230), outline=(*gold, 220), width=5 * scale)
    draw.arc(ball_box, 188, 352, fill=(*white, 180), width=3 * scale)

    return canvas.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    ICONSET.mkdir(parents=True, exist_ok=True)

    (ASSET_ROOT / "Contents.json").write_text(
        json.dumps({"info": {"author": "xcode", "version": 1}}, indent=2) + "\n",
        encoding="utf-8",
    )

    slots = [
        ("20x20", "2x", 40),
        ("20x20", "3x", 60),
        ("29x29", "2x", 58),
        ("29x29", "3x", 87),
        ("40x40", "2x", 80),
        ("40x40", "3x", 120),
        ("60x60", "2x", 120),
        ("60x60", "3x", 180),
        ("1024x1024", "1x", 1024),
    ]

    source = _draw_icon(1024)
    images = []
    for size_name, scale, pixels in slots:
        filename = (
            "Icon-1024.png" if pixels == 1024 else f"Icon-{size_name.replace('x', '')}@{scale}.png"
        )
        source.resize((pixels, pixels), Image.Resampling.LANCZOS).save(ICONSET / filename)
        images.append(
            {
                "filename": filename,
                "idiom": "ios-marketing" if pixels == 1024 else "iphone",
                "scale": scale,
                "size": size_name,
            }
        )

    (ICONSET / "Contents.json").write_text(
        json.dumps({"images": images, "info": {"author": "xcode", "version": 1}}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

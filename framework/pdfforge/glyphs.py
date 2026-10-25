"""Glyphs drawn as images (catalog F16): the page shows the symbol, the PDF holds no character."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = Path(__file__).resolve().parent / "fonts" / "STIXTwoText-Regular.ttf"


def glyph_png(ch: str, px: int = 64) -> str:
    cache = Path(tempfile.gettempdir()) / "pdfforge_glyphs"
    cache.mkdir(exist_ok=True)
    out = cache / f"g{ord(ch):05x}_{px}.png"
    if not out.exists():
        font = ImageFont.truetype(str(FONT), px)
        img = Image.new("L", (px, px), 255)
        d = ImageDraw.Draw(img)
        bbox = d.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((px - w) / 2 - bbox[0], (px - h) / 2 - bbox[1]), ch, fill=0, font=font)
        img.save(out)
    return str(out)

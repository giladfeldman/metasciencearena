"""Post-render PDF surgery: make an alias font's text layer say something other than its
glyphs show (catalog F), and other page-level rewrites (MediaBox offset, L02).

`rewrite_tounicode(pdf, alias, mapping)`:
  mapping = {source_char: target_str | None}; None deletes that entry (the glyph then
  has no Unicode at all); target "�" is an explicit replacement character.
  mapping = None removes the whole ToUnicode map from the alias font (Linotype-style).
Only fonts whose BaseFont ends with "+<alias PostScript name>" are touched, so the same
character drawn in the body face elsewhere stays honest.
"""
from __future__ import annotations

import io
import re

import pikepdf

from ..render import alias_ps_name


def _fonts(pdf: pikepdf.Pdf):
    seen = set()
    for page in pdf.pages:
        res = page.get("/Resources")
        if res is None or "/Font" not in res:
            continue
        for _, f in res.Font.items():
            og = f.objgen
            if og in seen:
                continue
            seen.add(og)
            yield f


def rewrite_tounicode(pdf_bytes: bytes, alias: str, mapping: dict | None) -> tuple[bytes, int]:
    ps = alias_ps_name(alias)
    pdf = pikepdf.open(io.BytesIO(pdf_bytes))
    changed = 0
    for f in _fonts(pdf):
        if not str(f.get("/BaseFont", "")).endswith("+" + ps):
            continue
        if mapping is None:
            if "/ToUnicode" in f:
                del f["/ToUnicode"]
                changed += 1
            continue
        if "/ToUnicode" not in f:
            continue
        cm = f.ToUnicode.read_bytes().decode("latin-1")

        def fix_block(mm):
            kept = []
            nonlocal changed
            for line in mm.group(2).splitlines():
                m = re.fullmatch(r"\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*", line)
                if not m:
                    continue
                src = bytes.fromhex(m.group(2)).decode("utf-16-be")
                if src in mapping:
                    changed += 1
                    tgt = mapping[src]
                    if tgt is None:
                        continue  # entry deleted: the glyph has no Unicode
                    line = f"<{m.group(1)}> <{tgt.encode('utf-16-be').hex().upper()}>"
                kept.append(line.strip())
            return f"{len(kept)} beginbfchar\n" + "\n".join(kept) + "\nendbfchar"

        out = re.sub(r"(\d+)\s+beginbfchar(.*?)endbfchar", fix_block, cm, flags=re.S)
        f.ToUnicode = pdf.make_stream(out.encode("latin-1"))
    buf = io.BytesIO()
    pdf.save(buf, deterministic_id=True)
    return buf.getvalue(), changed


def offset_mediabox(pdf_bytes: bytes, dx: float, dy: float) -> bytes:
    """Non-zero MediaBox origin (catalog L02): same page, shifted coordinate frame."""
    pdf = pikepdf.open(io.BytesIO(pdf_bytes))
    for page in pdf.pages:
        x0, y0, x1, y1 = [float(v) for v in page.MediaBox]
        page.MediaBox = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
        if "/CropBox" in page:
            del page["/CropBox"]
    buf = io.BytesIO()
    pdf.save(buf, deterministic_id=True)
    return buf.getvalue()

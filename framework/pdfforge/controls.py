"""Catalog X03: the same paper with every page an image -- no text layer at all.

A tool that returns nothing here must score zero; a tool that returns text it could
not have read (cached, hallucinated) is caught. Gold is unchanged: the page shows the
same paper, so the answer key is what an OCR-capable tool should recover.
"""
from __future__ import annotations

import io

import pypdfium2 as pdfium
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def image_only(pdf_bytes: bytes, scale: float = 2.0) -> bytes:
    src = pdfium.PdfDocument(pdf_bytes)
    out = io.BytesIO()
    c = None
    for page in src:
        w, h = page.get_size()
        if c is None:
            c = canvas.Canvas(out, pagesize=(w, h), invariant=1)
        else:
            c.setPageSize((w, h))
        img = page.render(scale=scale).to_pil().convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        c.drawImage(ImageReader(buf), 0, 0, width=w, height=h)
        c.showPage()
    c.save()
    return out.getvalue()

"""Build one PDF-universe task: paper -> defects -> render -> post -> gold -> manifest proof.

The manifest proof answers "did this defect actually do anything?" with real
extractors, not by assumption (lesson CL:513: a synthetic probe passed while the real
document failed). The same paper is rendered once WITHOUT the defects (the clean twin,
catalog X01); both PDFs go through pdftotext (poppler) and pypdf; a defect manifests if
at least one extractor's text differs between the two. A recipe that manifests in
neither extractor is a no-op and the corpus builder drops it.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import io
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import defects as D
from .content import build_paper
from .gold import all_gold
from .model import Paper
from .render import render


@dataclass
class Built:
    paper: Paper
    pdf: bytes
    placement: dict
    gold: dict
    proof: dict


def finalize(paper: Paper) -> None:
    """Renumber tables in reading order after defects inserted or moved some."""
    n = 0
    for b in paper.blocks:
        if b.kind == "table" and b.table is not None:
            n += 1
            if b.table.label:
                b.table.label = f"Table {n}"
            if b.layout.get("side_by_side") is not None:
                n += 1
                b.layout["side_by_side"].label = f"Table {n}"


def extract_pdftotext(pdf: bytes) -> str:
    exe = shutil.which("pdftotext")
    if not exe:
        raise RuntimeError("pdftotext not on PATH; the manifest proof needs it")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.pdf"
        p.write_bytes(pdf)
        r = subprocess.run([exe, "-enc", "UTF-8", str(p), "-"], capture_output=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"pdftotext failed: {r.stderr[:200]!r}")
        return r.stdout.decode("utf-8", "replace")


def extract_pypdf(pdf: bytes) -> str:
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


EXTRACTORS = {"pdftotext": extract_pdftotext, "pypdf": extract_pypdf}


def _diff_snippet(a: str, b: str, n: int = 3) -> list[str]:
    out = []
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0):
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            out.append(line[:160])
        if len(out) >= 2 * n:
            break
    return out


def manifest_proof(pdf: bytes, twin_pdf: bytes) -> dict:
    """Per extractor: does the defective PDF extract differently from its twin?

    The twin is the clean paper for "diff"/"notation" recipes, and for "visible" recipes
    (lying glyph maps) the SAME defective paper rendered honestly, before the font-map
    rewrite -- so the test isolates the lie: the page is identical, only the text layer
    differs, and an extractor that recovers the truth anyway shows no change.
    """
    res = {}
    for name, fn in EXTRACTORS.items():
        a, b = fn(twin_pdf), fn(pdf)
        res[name] = {"changed": a != b, "diff": _diff_snippet(a, b)}
    res["manifests"] = any(v["changed"] for v in res.values() if isinstance(v, dict))
    return res


def build_task(paper_id: str, seed: int, prose: dict, split: str, defect_ids: list[str],
               layout: dict | None = None, *, proof: bool = True, **paper_kw) -> Built:
    base = build_paper(paper_id, seed, prose, split, **paper_kw)
    base.layout.update(layout or {})
    clean = copy.deepcopy(base)
    rng = random.Random(f"{paper_id}|{seed}|defects")
    applied = D.apply_defects(base, defect_ids, rng)
    finalize(base)
    pdf, placement = render(base)
    honest_pdf = pdf
    pdf = D.post_process(pdf, base)
    gold = all_gold(base, placement)
    pr: dict = {"applied": applied, "requested": defect_ids}
    if proof and applied:
        probes = {D.REGISTRY[i].probe for i in applied}
        if probes == {"property"}:
            pr.update({"twin": None, "manifests": True,
                       "note": "a structural property of the paper (present once applied), not a defect to diff"})
        elif probes == {"visible"}:
            pr["twin"] = "honest render of the same paper (before the font-map rewrite)"
            pr.update(manifest_proof(pdf, honest_pdf))
        else:
            finalize(clean)
            cpdf, _ = render(clean)
            pr["twin"] = "clean paper (defects not applied)"
            pr.update(manifest_proof(pdf, cpdf))
    pr["pdf_sha256"] = hashlib.sha256(pdf).hexdigest()
    return Built(base, pdf, placement, gold, pr)

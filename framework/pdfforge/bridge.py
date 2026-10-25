"""Map PDF-universe tasks onto each arena's envelope + gold (the arena side of the universe).

Every PDF arena's new task-set version calls `envelopes(arena_id, version)` and serves
gold through `gold(arena_id, task_id)`. The PDF bytes and all gold come from the same
built task, so the six arenas see identical documents.

Task ids are the universe keys ("pu1-..."): revealed keys name their recipe, private
keys are opaque hashes. Held-out (private) tasks follow the arena convention: every
answer field starts with "gold" so framework.holdout strips it from tracked artifacts.
"""
from __future__ import annotations

import base64

from . import universe as U

ARENAS = ("pdf-text-fidelity-v1", "pdf-table-extraction-v1", "pdf-section-structure-v1",
          "pdf-reference-parsing-v1", "pdf-citation-matching-v1", "stats-from-pdf-v1")
_GOLD: dict[tuple[str, str], dict] = {}


def _load(arena: str, include_private: bool = True):
    """All universe specs. A missing private seed RAISES (framework.holdout rule): serving
    only the public half would be a smaller benchmark reported as a success. The
    declared opt-out is SCIENCEARENA_SYNTHETIC_ONLY=1, which then yields revealed only."""
    if not include_private:
        return U.task_specs("v1", include_private=False)
    from framework.holdout import require_corpus
    if U.private_seed(required=False) is None and             require_corpus(U.PRIVATE_SEED_FILE, arena_id=arena, kind="PDF-universe private seed") is None:
        return U.task_specs("v1", include_private=False)
    return U.task_specs("v1", include_private=True)


def _defect_load(spec) -> int:
    return {"twin": 0, "isolation": 1, "house": 2, "control": 3}[spec.kind]


def _families(bt) -> list[str]:
    return sorted({d["id"][0] for d in bt.paper.defects})


def _difficulty(arena: str, spec, bt) -> dict:
    g = bt.gold
    cols = dict(spec.layout).get("columns", 1)
    load = _defect_load(spec)
    if arena == "pdf-text-fidelity-v1":
        return {"source_kind": 1, "column_count": cols, "corruption_intensity": load}
    if arena == "pdf-table-extraction-v1":
        n = len(g["tables"]["gold_tables"])
        hdr = max([t["header_rows"] for t in g["tables"]["gold_tables"]] + [1])
        spans = any(c["colspan"] > 1 or c["rowspan"] > 1 for t in g["tables"]["gold_tables"] for c in t["cells"])
        return {"source_kind": 1, "n_tables": min(max(n, 1), 8), "header_complexity": 3 if spans else min(hdr, 2)}
    if arena == "pdf-section-structure-v1":
        labels = {s["label"] for s in g["sections"]["gold_sections"]}
        return {"source_kind": 1, "section_count": min(len(g["sections"]["gold_sections"]), 8),
                "label_diversity": 3 if len(labels) >= 6 else 2 if len(labels) >= 4 else 1}
    if arena in ("pdf-reference-parsing-v1", "pdf-citation-matching-v1"):
        n = len(g["references"]["gold_references"])
        style = {"numeric": 1, "apa": 2, "harvard": 2, "aom": 2, "superscript": 3}[bt.paper.cite_style]
        return {"source_kind": 1, "citation_style": style, "column_count": min(cols, 3),
                "reference_count": 1 if n < 20 else 2 if n <= 50 else 3,
                "hard_format": 1 if any(f in _families(bt) for f in ("B", "C")) else 0}
    if arena == "stats-from-pdf-v1":
        n = len(g["stats"]["gold_stats"])
        return {"source_kind": 1, "defect_load": load, "stat_count": 1 if n < 10 else 2 if n < 20 else 3}
    raise KeyError(arena)


def _arena_gold(arena: str, spec, bt) -> dict:
    g = bt.gold
    meta = {"doc_id": spec.task_key, "source_kind": "synthetic", "pu_kind": spec.kind,
            "pu_defects": sorted({d["id"] for d in bt.paper.defects}), "pu_house": spec.house}
    if arena == "pdf-text-fidelity-v1":
        t = g["text"]
        return {**meta, "corruption_intensity": _defect_load(spec), "column_count": t["column_count"],
                "gold_preserve_greek": t["gold_preserve_greek"], "gold_ascii_greek": t["gold_ascii_greek"]}
    if arena == "pdf-table-extraction-v1":
        # no "scoring": "v2" -- the v2 content-overlap table matching is unreleased (d-24891c)
        return {**meta, "n_tables": g["tables"]["n_tables"], "gold_tables": g["tables"]["gold_tables"]}
    if arena == "pdf-section-structure-v1":
        s = g["sections"]
        return {**meta, "section_count": s["section_count"], "gold_sections": s["gold_sections"],
                "gold_full_text": s["gold_full_text"]}
    if arena == "pdf-reference-parsing-v1":
        r = g["references"]
        return {**meta, "scoring": "v2", "reference_count": r["reference_count"], "gold_references": r["gold_references"]}
    if arena == "pdf-citation-matching-v1":
        c = g["citations"]
        return {**meta, "scoring": "v2", "gold_references": c["gold_references"], "gold_markers": c["gold_markers"],
                "gold_consistency": c["gold_consistency"]}
    if arena == "stats-from-pdf-v1":
        return {**meta, "stat_count": g["stats"]["stat_count"], "gold_stats": g["stats"]["gold_stats"]}
    raise KeyError(arena)


def envelopes(arena: str, version: str, *, include_private: bool = True):
    """Yield (envelope) for every universe task, caching each task's gold for gold()."""
    for spec in _load(arena, include_private):
        bt = U.built(spec)
        env = {"task_id": spec.task_key, "arena_id": arena, "task_set_version": version,
               "difficulty": _difficulty(arena, spec, bt), "visibility": spec.visibility,
               "input": {"document_bytes_b64": base64.b64encode(bt.pdf).decode("ascii"),
                         "filename": f"{spec.task_key}.pdf", "n_pages": int(bt.placement["pages"]),
                         "media_type": "application/pdf"}}
        _GOLD[(arena, spec.task_key)] = _arena_gold(arena, spec, bt)
        yield env


def gold(arena: str, task_id: str) -> dict | None:
    return _GOLD.get((arena, task_id))


def is_universe_task(task_id: str) -> bool:
    return task_id.startswith("pu1-")

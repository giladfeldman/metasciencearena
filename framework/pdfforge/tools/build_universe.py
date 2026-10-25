"""Build every PDF-universe task (parallel), verify each, write the committed manifest.

    .venv/Scripts/python.exe -m framework.pdfforge.tools.build_universe --jobs 6

Per task it checks: the PDF opens and has pages; every gold has content (non-degenerate:
text must_include, >=1 section, >=1 reference, >=1 statistic); isolation/house tasks
carry a manifest proof that the defects showed up. A failing task stops the build
(nothing is silently dropped). The manifest (framework/pdfforge/universe_manifest.json)
lists revealed tasks in full (key, kind, defects, pdf sha256, counts) and private tasks
only as aggregate counts, so it can be tracked in git.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from framework.pdfforge import universe as U

MANIFEST = Path(U.__file__).resolve().parent / "universe_manifest.json"


def _check(spec_d: dict) -> dict:
    spec = U.Spec(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in spec_d.items()})
    t0 = time.time()
    bt = U.built(spec)
    g = bt.gold
    problems = []
    if not bt.pdf.startswith(b"%PDF"):
        problems.append("not a PDF")
    if not g["text"]["gold_preserve_greek"]["must_include"].strip():
        problems.append("empty text gold")
    if not g["sections"]["gold_sections"]:
        problems.append("no sections")
    if not g["references"]["gold_references"]:
        problems.append("no references")
    if not g["stats"]["gold_stats"]:
        problems.append("no statistics")
    if spec.kind in ("isolation", "house") and not bt.proof.get("manifests"):
        problems.append(f"defects did not manifest: {bt.proof.get('applied')}")
    if spec.kind in ("isolation", "house") and set(spec.defects) - set(bt.proof.get("applied", [])) and spec.kind == "isolation":
        problems.append(f"isolation defect not applied: {spec.defects}")
    return {"task_key": spec.task_key, "kind": spec.kind, "split": spec.split, "visibility": spec.visibility,
            "defects_requested": list(spec.defects), "defects_applied": bt.proof.get("applied", []),
            "house": spec.house, "pdf_sha256": bt.proof["pdf_sha256"], "pages": bt.placement["pages"],
            "n_stats": len(g["stats"]["gold_stats"]), "n_tables": len(g["tables"]["gold_tables"]),
            "n_refs": len(g["references"]["gold_references"]), "n_markers": len(g["citations"]["gold_markers"]),
            "n_sections": len(g["sections"]["gold_sections"]),
            "n_planted": sum(1 for s in g["stats"]["gold_stats"] if s.get("planted")),
            "seconds": round(time.time() - t0, 1), "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    specs = U.task_specs("v1")[: a.limit]
    print(f"{len(specs)} specs: {collections.Counter((s.kind, s.split) for s in specs)}", flush=True)
    rows, bad = [], []
    with cf.ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(_check, asdict(s)): s for s in specs}
        for k, f in enumerate(cf.as_completed(futs), 1):
            s = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 - every failure is reported and stops the build
                r = {"task_key": s.task_key, "split": s.split, "problems": [f"{type(e).__name__}: {e}"]}
            rows.append(r)
            if r["problems"]:
                bad.append(r)
                print(f"BAD  {r['task_key']} ({s.split}) {r['problems']}", flush=True)
            if k % 25 == 0:
                print(f"  {k}/{len(specs)}", flush=True)
    rows.sort(key=lambda r: r["task_key"])
    revealed = [r for r in rows if r.get("split") == "revealed"]
    private = [r for r in rows if r.get("split") == "private"]
    agg = lambda rs, k: sum(r.get(k, 0) for r in rs)  # noqa: E731
    manifest = {
        "universe": "pdf-universe", "version": "v1", "source_hash": U._source_hash(),
        "n_tasks": len(rows), "n_revealed": len(revealed), "n_private": len(private),
        "private_summary": {"n": len(private), "stats": agg(private, "n_stats"), "tables": agg(private, "n_tables"),
                            "refs": agg(private, "n_refs"), "planted": agg(private, "n_planted"),
                            "kinds": dict(collections.Counter(r.get("kind") for r in private))},
        "revealed": revealed,
    }
    if bad:
        print(f"FAILED: {len(bad)} task(s) with problems; manifest NOT written")
        return 1
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"ok: {len(rows)} tasks ({len(revealed)} revealed, {len(private)} private); "
          f"stats {agg(rows, 'n_stats')}, tables {agg(rows, 'n_tables')}, refs {agg(rows, 'n_refs')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Manifest proof for every defect recipe, saved as evidence.

    .venv/Scripts/python.exe -m framework.pdfforge.tools.probe_recipes [--only F03,L04]

For each recipe: build papers (up to --tries seeds) until the recipe applies, render it
and its clean twin, and record whether pdftotext / pypdf output changed plus a short
diff. Writes framework/pdfforge/recipe_proof.json. The corpus builder admits only
recipes whose proof says `manifests: true` (or that are declared known-impossible, which
are reported in their own stratum).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from framework.pdfforge import defects as D
from framework.pdfforge.build import build_task

OUT = Path(__file__).resolve().parents[1] / "recipe_proof.json"
BANK = Path(__file__).resolve().parents[1] / "prose_bank.json"


def probe(did: str, bank: dict, tries: int, base_seed: int) -> dict:
    topics = sorted(bank)
    last = None
    for k in range(tries):
        seed = base_seed + 7919 * k
        bt = build_task(f"probe-{did}-{k}", seed, bank[topics[(seed // 7) % len(topics)]], "revealed",
                        [did], {"columns": 2 if k % 2 else 1})
        last = bt.proof
        if bt.proof["applied"]:
            r = D.REGISTRY[did]
            return {"defect_id": did, "family": r.family, "title": r.title, "recoverable": r.recoverable,
                    "seed": seed, "tries": k + 1, "applied": True, "manifests": bt.proof.get("manifests"),
                    "pdftotext": bt.proof.get("pdftotext"), "pypdf": bt.proof.get("pypdf"),
                    "pdf_sha256": bt.proof["pdf_sha256"]}
    return {"defect_id": did, "applied": False, "tries": tries, "manifests": None, "last": last}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--tries", type=int, default=8)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    bank = json.loads(BANK.read_text(encoding="utf-8"))["entries"]
    ids = [x for x in a.only.split(",") if x] or sorted(D.REGISTRY)
    report = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for i, did in enumerate(ids):
        try:
            res = probe(did, bank, a.tries, 1000 + 31 * i)
        except Exception as e:  # noqa: BLE001 - recorded as evidence, then shown
            res = {"defect_id": did, "error": f"{type(e).__name__}: {e}"}
        report[did] = res
        flag = res.get("error") or ("NOT APPLIED" if not res.get("applied") else f"manifests={res['manifests']}")
        print(f"{did:5s} {flag}", flush=True)
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    bad = [k for k, v in report.items() if v.get("error") or not v.get("applied") or v.get("manifests") is False]
    print(f"done: {len(report)} probed, {len(bad)} not admissible: {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

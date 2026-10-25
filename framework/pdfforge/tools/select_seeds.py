"""Choose, once, a seed offset per isolation task so its recipe applies and manifests.

    .venv/Scripts/python.exe -m framework.pdfforge.tools.select_seeds [--only F02,S13] [--max 25]

Writes framework/pdfforge/universe_seed_offsets.json ({"iso|<split>|<recipe>": offset}).
Offsets are not secret: without the private seed they reproduce nothing.
A recipe that fails for every offset is reported (and the build will fail on it).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from framework.pdfforge import universe as U
from framework.pdfforge.build import build_task


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--max", type=int, default=25)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    only = {x for x in a.only.split(",") if x}
    offsets = json.loads(U.OFFSETS.read_text(encoding="utf-8")) if U.OFFSETS.exists() else {}
    bank = json.loads(U.BANK.read_text(encoding="utf-8"))["entries"]
    pseed = U.private_seed()
    failed = []
    for spec in U.task_specs("v1"):
        if spec.kind != "isolation":
            continue
        did = spec.defects[0]
        if only and did not in only:
            continue
        key = f"iso|{spec.split}|{did}"
        base = U.PUBLIC_BASE_SEED if spec.split == "revealed" else pseed
        for off in range(0, a.max):
            seed = U._seed(base, "iso", did) if off == 0 else U._seed(base, "iso", did, off)
            kw = {}
            if spec.style_pair:
                kw["style_pair"] = tuple(spec.style_pair)
            if spec.n_studies:
                kw["n_studies"] = spec.n_studies
            bt = build_task(spec.task_key, seed, bank[spec.topic], spec.split, [did], dict(spec.layout), **kw)
            if did in bt.proof["applied"] and bt.proof.get("manifests"):
                offsets[key] = off
                break
        else:
            failed.append(key)
            print(f"FAIL {key}: no offset in 0..{a.max - 1} applies and manifests", flush=True)
            continue
        if offsets[key]:
            print(f"{key}: offset {offsets[key]}", flush=True)
        U.OFFSETS.write_text(json.dumps(dict(sorted(offsets.items())), indent=1), encoding="utf-8")
    print(f"done; {len(failed)} failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build task set v4 of citation-claim-fidelity-v1: v3 + tier rd-mmc-pilot (citeclaim Stage E).

GOLD-BUILDING code. Inputs and output are real article text, in custody only:
    $ARTICLE_REPOSITORY/ground_truth/citeclaim-2026-10/stage_e/jobs/*.json      (MetaMisCitations)
    $ARTICLE_REPOSITORY/ground_truth/citeclaim-2026-10/stage_e/gold/gold_mmc_v1.jsonl
    $ARTICLE_REPOSITORY/ground_truth/citeclaim-2026-10/build_v4/tasks_v4.jsonl  (output)

v4 = every v3 task byte-for-byte (sha-checked against MANIFEST) plus one task per call
the citeclaim library makes on the MetaMisCitations pilot: one cited source and up to 6 of
its citing sentences, in the library's grouping and order. Labels are FRONTIER labels, not
human labels (docs/design/2026-10-citeclaim-stage-e-mmc-pilot.md, amendments 1 and 1a):
`frontier_consensus` (three seats agreed) or `adjudicated` (a blind Opus xhigh decider).
An `undecidable` item is dropped from its task (counted); a task left empty is dropped.

    python arenas/citation-claim-fidelity-v1/tools/build_mmc_tier.py
Prints counts only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ccf_stage_e_panel", HERE / "stage_e_panel.py")
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)

STAGE_E = panel.STAGE_E
CUSTODY = STAGE_E.parent
TIER = "rd-mmc-pilot"
LABELS = ("consistent", "contradicts", "silent")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def mmc_tasks(report: Counter) -> list[dict]:
    gold = {g["citation_id"]: g for g in
            (json.loads(l) for l in (STAGE_E / "gold" / "gold_mmc_v1.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip())}
    jobs = {jf.stem: json.loads(jf.read_text(encoding="utf-8")) for jf in (STAGE_E / "jobs").glob("*.json")}
    cit = {c["id"]: c for j in jobs.values() for c in j["citations"]}
    absd = {a["doi"]: a for j in jobs.values() for a in j["abstracts"]}
    out = []
    for g in panel.groups():
        items = []
        for cid in g["citation_ids"]:
            gl = gold[cid]
            if gl["label"] not in LABELS:
                report[f"excluded_{gl['label']}"] += 1
                continue
            report[f"items_{gl['label_source']}_{gl['label']}"] += 1
            items.append({"item_id": _sid(TIER, cid), "sentence": cit[cid]["sentence"], "gold": gl["label"],
                          "type": f"MMC-{gl['label']}", "traps": [],
                          "meta": {"source": gl["label_source"], "discipline": gl["discipline"],
                                   "multi_marker_sentence": (gl["sentence_sources"] or 0) > 1}})
        if not items:
            report["tasks_dropped_empty"] += 1
            continue
        a = absd[g["doi"]]
        out.append({"task_key": _sid(TIER, g["job"], g["rendered_sha256"]), "tier": TIER, "abstract_doi": g["doi"],
                    "title": a.get("title") or "(unknown)", "year": a.get("year") or "n.d.",
                    "abstract": a["abstract"], "items": items})
    return out


def main() -> int:
    manifest_path = CUSTODY / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    v3 = CUSTODY / "build_v3" / "tasks_v3.jsonl"
    v3_sha = _sha(v3)
    if v3_sha != manifest["build_v3/tasks_v3.jsonl"]["sha256"]:
        raise SystemExit("tasks_v3.jsonl sha256 != MANIFEST; refusing to build v4 on altered v3")
    report: Counter = Counter()
    v3_lines = [l for l in v3.read_text(encoding="utf-8").splitlines() if l.strip()]
    real = mmc_tasks(report)
    keys = [json.loads(l)["task_key"] for l in v3_lines] + [t["task_key"] for t in real]
    if len(keys) != len(set(keys)):
        raise SystemExit("v4 task keys collide")
    out_dir = CUSTODY / "build_v4"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "tasks_v4.jsonl"
    out.write_text("\n".join(v3_lines + [json.dumps(t, ensure_ascii=False) for t in real]) + "\n", encoding="utf-8")
    sha = _sha(out)
    summary = {"tasks_v3": len(v3_lines), "tasks_mmc": len(real), "tasks_v4": len(v3_lines) + len(real),
               "items_mmc": sum(len(t["items"]) for t in real),
               "group_sizes": dict(Counter(len(t["items"]) for t in real)),
               "counts": dict(sorted(report.items())), "tasks_v3_sha256": v3_sha, "tasks_v4_sha256": sha,
               "gold_sha256": _sha(STAGE_E / "gold" / "gold_mmc_v1.jsonl")}
    (out_dir / "assemble_report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    n_items = sum(len(json.loads(l)["items"]) for l in v3_lines) + summary["items_mmc"]
    manifest["build_v4/tasks_v4.jsonl"] = {"sha256": sha, "n_tasks": summary["tasks_v4"], "n_items": n_items}
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

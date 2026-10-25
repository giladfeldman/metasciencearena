"""Build task set v3 of citation-claim-fidelity-v1: v2 + the real-data tiers (citeclaim Stage D).

GOLD-BUILDING code. Every input and output is real article text or human labels, so all of
it lives in article custody, never in this repository:
    $ARTICLE_REPOSITORY/ground_truth/citeclaim-2026-10/stage_d/<dataset>/raw/   (as downloaded)
    $ARTICLE_REPOSITORY/ground_truth/citeclaim-2026-10/build_v3/tasks_v3.jsonl  (output)

v3 = every v2 task byte-for-byte (sha-checked against MANIFEST, tier "synthetic") plus one
task per usable labelled pair of two public human-labelled datasets. No model writes or
filters any v3 item: the gold is the datasets' own human labels, mapped by the fixed table
below (docs/design/2026-10-citeclaim-stage-d-real-data.md; consulted before any score).

  tier rd-zhang2024   Zhang & Abernethy 2024 (arXiv 2411.06101), data licence ODC-By 1.0.
                      Labels were judged against the reference's FULL TEXT, so at abstract
                      level:  Fully substantiated -> "not_contradicts" (consistent or silent
                      correct; contradicts is a false suggestion),  Unsubstantiated ->
                      "not_consistent" (contradicts or silent correct).  Partially
                      substantiated (14) and pairs without an abstract (8) are EXCLUDED.
  tier rd-scifact-dev SciFact dev claims (CC BY 4.0) x their cited abstracts (ODC-By 1.0),
                      labels judged against the ABSTRACT:  SUPPORT -> consistent,
                      CONTRADICT -> contradicts, no evidence in a cited abstract -> silent.
                      Calibration tier: the claims are rewritten, not real citing sentences.

Every real-data task holds ONE sentence (group size 1): the datasets have no realistic
groups of different sentences citing one source.

    python arenas/citation-claim-fidelity-v1/tools/build_real_tier.py
Prints counts only, never text.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

CUSTODY = Path(os.environ["ARTICLE_REPOSITORY"]) / "ground_truth" / "citeclaim-2026-10"
STAGE_D = CUSTODY / "stage_d"
ZA_XLSX = STAGE_D / "zhang_abernethy_2024" / "raw" / "ReferenceErrorDetection_data.xlsx"
SF_DIR = STAGE_D / "scifact" / "raw" / "data"
#: sha256 of the files as downloaded on 2026-10-18 (GitHub tianmai-zhang/ReferenceErrorDetection
#: at 0202bff; SciFact release/latest/data.tar.gz). A changed input refuses to build.
INPUT_SHA = {
    ZA_XLSX: "624f88801fe87e3a76bcf3cd18aea5454a5effe4dfb452f936ad0e8bf3b2ce91",
    SF_DIR / "claims_dev.jsonl": "86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217",
    SF_DIR / "corpus.jsonl": "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62",
}
ZA_GOLD = {"Fully substantiate": ("not_contradicts", "ZA-FULL"),
           "Unsubstantiate": ("not_consistent", "ZA-UNSUB")}
SF_GOLD = {"SUPPORT": ("consistent", "SF-SUPPORT"), "CONTRADICT": ("contradicts", "SF-CONTRADICT"),
           None: ("silent", "SF-NEI")}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _task(tier: str, key: str, doi: str, title: str, year, abstract: str, sentence: str,
          gold: str, typ: str, meta: dict) -> dict:
    return {"task_key": _sid(tier, key), "tier": tier, "abstract_doi": doi, "title": title, "year": year,
            "abstract": abstract, "items": [{"item_id": _sid(tier, key, "item"), "sentence": sentence,
                                             "gold": gold, "type": typ, "traps": [], "meta": meta}]}


def zhang_tasks(report: Counter) -> list[dict]:
    import openpyxl
    ws = openpyxl.load_workbook(ZA_XLSX, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header, rows = rows[0], rows[1:]
    col = {h: i for i, h in enumerate(header)}
    # Sonnet, consult 2026-10-18: an unexpected label or source string (a case or spacing
    # variant) was silently binned as "partial" / "citation-study". Refuse instead.
    labels = {r[col["Label"]] for r in rows}
    if labels != {"Fully substantiate", "Partially substantiate", "Unsubstantiate"}:
        raise SystemExit(f"unexpected Zhang label set ({len(labels)} values)")
    study_sources = {"Lee & Lee, 1999", "Fenton et al., 2000", "Lukic et al., 2004", "Handoll, 2015",
                     "Smith & Cumberledge, 2020", "Smith & Cumberledge, 2021"}
    unknown = {r[col["Source"]] for r in rows} - study_sources - {"PubPeer", "Correction"}
    if unknown:
        raise SystemExit(f"unexpected Zhang source value(s): {len(unknown)}")
    out = []
    for n, r in enumerate(rows):
        label, abstract = r[col["Label"]], (r[col["Reference Article Abstract"]] or "").strip()
        report[f"za_rows_{label}"] += 1
        if label not in ZA_GOLD:
            report["za_excluded_partial"] += 1
            continue
        if not abstract:
            report[f"za_excluded_no_abstract_{label}"] += 1
            continue
        gold, typ = ZA_GOLD[label]
        source = r[col["Source"]]
        meta = {"source": "PubPeer" if source == "PubPeer" else "Correction" if source == "Correction"
                else "citation-study", "source_detail": source, "explanation": r[col["Explanation"]],
                "domain": r[col["Domain"]], "row": n + 2}
        out.append(_task("rd-zhang2024", f"row{n + 2}", r[col["Reference Article DOI"]],
                         r[col["Reference Article Title"]] or "(unknown)", "n.d.", abstract,
                         r[col["Statement with Citation"]].strip(), gold, typ, meta))
    return out


def scifact_tasks(report: Counter) -> list[dict]:
    corpus = {}
    for line in (SF_DIR / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            corpus[str(d["doc_id"])] = d
    out = []
    for line in (SF_DIR / "claims_dev.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        docs = list(dict.fromkeys(c["cited_doc_ids"]))  # claim 1245 lists one doc twice
        report["sf_duplicate_cited_doc"] += len(c["cited_doc_ids"]) - len(docs)
        for doc in docs:
            ev = c["evidence"].get(str(doc))
            labels = {e["label"] for e in ev} if ev else {None}
            if len(labels) != 1:
                report["sf_excluded_mixed_label"] += 1
                continue
            gold, typ = SF_GOLD[labels.pop()]
            d = corpus.get(str(doc))
            if d is None:
                report["sf_excluded_doc_missing"] += 1
                continue
            abstract = " ".join(s.strip() for s in d["abstract"])
            out.append(_task("rd-scifact-dev", f"claim{c['id']}-doc{doc}", f"s2orc:{doc}", d["title"], "n.d.",
                             abstract, c["claim"].strip(), gold, typ,
                             {"claim_id": c["id"], "doc_id": doc, "n_cited_docs": len(c["cited_doc_ids"])}))
    return out


def main() -> int:
    for path, sha in INPUT_SHA.items():
        if _sha(path) != sha:
            raise SystemExit(f"{path.name}: sha256 differs from the 2026-10-18 download; refusing to build")
    manifest_path = CUSTODY / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    v2 = CUSTODY / "build_v2" / "tasks_v2.jsonl"
    v2_sha = _sha(v2)
    if v2_sha != manifest["build_v2/tasks_v2.jsonl"]["sha256"]:
        raise SystemExit("tasks_v2.jsonl sha256 != MANIFEST; refusing to build v3 on altered v2")
    report: Counter = Counter()
    v2_lines = [l for l in v2.read_text(encoding="utf-8").splitlines() if l.strip()]
    real = zhang_tasks(report) + scifact_tasks(report)
    keys = [json.loads(l)["task_key"] for l in v2_lines] + [t["task_key"] for t in real]
    if len(keys) != len(set(keys)):
        raise SystemExit("v3 task keys collide")
    out_dir = CUSTODY / "build_v3"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "tasks_v3.jsonl"
    # v2 lines are copied verbatim (no "tier" key = synthetic), so every v2 task is byte-identical.
    out.write_text("\n".join(v2_lines + [json.dumps(t, ensure_ascii=False) for t in real]) + "\n",
                   encoding="utf-8")
    sha = _sha(out)
    by = Counter((t["tier"], t["items"][0]["type"]) for t in real)
    za_src = Counter((t["items"][0]["type"], t["items"][0]["meta"]["source"]) for t in real
                     if t["tier"] == "rd-zhang2024")
    summary = {"tasks_v2": len(v2_lines), "tasks_real": len(real), "tasks_v3": len(v2_lines) + len(real),
               "by_tier_type": {f"{a}/{b}": n for (a, b), n in sorted(by.items())},
               "zhang_type_by_source": {f"{a}/{b}": n for (a, b), n in sorted(za_src.items())},
               "exclusions": dict(sorted(report.items())), "tasks_v2_sha256": v2_sha, "tasks_v3_sha256": sha,
               "inputs_sha256": {p.name: s for p, s in INPUT_SHA.items()}}
    (out_dir / "assemble_report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    n_items = sum(len(json.loads(l)["items"]) for l in v2_lines) + len(real)
    manifest["build_v3/tasks_v3.jsonl"] = {"sha256": sha, "n_tasks": len(v2_lines) + len(real), "n_items": n_items}
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

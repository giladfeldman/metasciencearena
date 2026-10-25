"""Pooled comparison of the stats-from-pdf-v1 tool players, read from their stored run files.

    python -m framework.pdfforge.tools.summarize_stats_runs [--version v1] [--json out.json]

Public split (revealed): counts pooled over tasks (micro): recall = matched / findable,
precision = matched / non-neutral player results, decision errors caught / total, false
alarms over matched non-flagged-by-gold results (n_matched - inconsistencies_total; a
matched gold result with no computed verdict sits in that denominator too, so the rate is
a slight under-estimate). Errored tasks (the pipeline failed; primary 0 on the board)
are counted, not pooled: every pooled count, the decision-error total included, is over
the SCORED tasks only (consult 2026-10-18 F7). Held-out split (private): the mean primary only, since its
per-task breakdowns stay private.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _rows(f: Path) -> list[dict]:
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(version: str = "v1") -> dict:
    d = REPO / "arenas/stats-from-pdf-v1/runs" / version
    out: dict = {}
    for f in sorted(d.glob("*__revealed__*.jsonl")):
        pid = f.name.split("__")[0]
        s = {k: 0 for k in ("tasks", "scored", "errored", "findable", "matched", "player", "neutral", "de_caught",
                            "de_total", "false_alarms", "flagged_total")}
        for r in _rows(f):
            s["tasks"] += 1
            b = (r.get("score") or {}).get("breakdown")
            if not b or "error" in b:
                s["errored"] += 1  # a pipeline failure: primary 0 on the board, no counts to pool
                continue
            s["scored"] += 1
            s["findable"] += b["n_gold_findable"]
            s["matched"] += b["n_matched"]
            s["player"] += b["n_player_results"]
            s["neutral"] += b["n_neutral_matched"] + b["n_duplicates"]
            s["de_caught"] += b["decision_errors_caught"]
            s["de_total"] += b["decision_errors_total"]
            s["false_alarms"] += b["false_alarms"]
            s["flagged_total"] += b["inconsistencies_total"]
        s["recall"] = s["matched"] / s["findable"] if s["findable"] else None
        nn = s["player"] - s["neutral"]
        s["precision"] = s["matched"] / nn if nn else None
        neg = s["matched"] - s["flagged_total"]
        s["false_alarm_rate"] = s["false_alarms"] / neg if neg else None
        out.setdefault(pid, {})["public"] = s
    for f in sorted(d.glob("*__private__*.jsonl")):
        pid = f.name.split("__")[0]
        prim = [r["score"]["primary"] for r in _rows(f) if (r.get("score") or {}).get("primary") is not None]
        out.setdefault(pid, {})["heldout"] = {"tasks_scored": len(prim),
                                              "mean_primary": sum(prim) / len(prim) if prim else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = summarize(a.version)
    fmt = lambda x: "-" if x is None else f"{x:.3f}"
    print(f"{'player':28} {'scored':>6} {'recall':>7} {'prec':>7} {'DE caught':>11} {'FA rate':>8} {'errored':>7} {'held-out':>9}")
    for pid, v in res.items():
        p, h = v.get("public", {}), v.get("heldout", {})
        print(f"{pid:28} {p.get('scored', 0):>6} {fmt(p.get('recall')):>7} {fmt(p.get('precision')):>7} "
              f"{str(p.get('de_caught'))+'/'+str(p.get('de_total')):>11} {fmt(p.get('false_alarm_rate')):>8} "
              f"{p.get('errored', 0):>7} {fmt(h.get('mean_primary')):>9}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

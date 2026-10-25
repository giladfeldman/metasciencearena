"""Join the per-repeat analyses of one instrument into the pre-registered verdict.

The pass bars say "every bar must hold on each of the 2 repeat runs of the same instrument"
(Sonnet, consult 2026-10-18: nothing enforced it; one passing repeat could be reported as a
pass). This reads the `--out` JSON of tools/analyze_run.py for each repeat and passes only
when: there are at least 2; every one is complete; all name the same player, task set and
tier; they come from different run files; they scored the same task ids; and every one's
bars passed. Prints counts only.

    python arenas/citation-claim-fidelity-v1/tools/combine_repeats.py <analysis r1.json> <analysis r2.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def combine(docs: list[dict]) -> dict:
    sums = [d["summary"] for d in docs]
    problems = []
    if len(sums) < 2:
        problems.append(f"{len(sums)} repeat(s); the bars need 2")
    for key in ("player_id", "task_set", "tier"):
        vals = {str(s["meta"].get(key)) for s in sums}
        if len(vals) != 1:
            problems.append(f"repeats differ in {key}: {sorted(vals)}")
    shas = [s["meta"].get("run_file_sha256") for s in sums]
    if len(set(shas)) != len(shas):
        problems.append("two analyses come from the same run file")
    ids = [sorted({it["task_id"] for it in d["items"]}) for d in docs]
    if any(i != ids[0] for i in ids):
        problems.append("repeats scored different task ids")
    for i, s in enumerate(sums, 1):
        if not s.get("complete"):
            problems.append(f"repeat {i} is incomplete")
    per = [{"run_file_sha256": (s["meta"].get("run_file_sha256") or "")[:12], "complete": s.get("complete"),
            "bars_passed": s["bars"].get("passed"), "gating": s["bars"].get("gating", True),
            "failing": [c["bar"] for c in s["bars"].get("checks", []) if not c["pass"]]} for s in sums]
    passed = not problems and all(p["bars_passed"] is True for p in per)
    return {"player_id": sums[0]["meta"].get("player_id") if sums else None,
            "task_set": sums[0]["meta"].get("task_set") if sums else None,
            "tier": sums[0]["meta"].get("tier") if sums else None,
            "gating": per[0]["gating"] if per else None,
            "repeats": per, "problems": problems, "passed": passed}


def main() -> int:
    docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:]]
    print(json.dumps(combine(docs), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

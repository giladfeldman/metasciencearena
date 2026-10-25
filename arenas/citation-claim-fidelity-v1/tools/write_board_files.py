"""Write the two committed files the leaderboard build needs for a task set of this arena.

    python arenas/citation-claim-fidelity-v1/tools/write_board_files.py --version v4 [--check]

1. ``task_sets/<v>/_ground_truth.json``: the REVEALED split only (the 5 fictional examples),
   envelope + gold, exactly as v1/v2's were committed. The build reads it for the public task
   pages; held-out gold never enters it.
2. ``task_sets/<v>/tiers.jsonl`` (v3+): one ``{"task_id", "tier", "split"}`` line per task,
   revealed and private. Run records carry no tier, and the board must score each tier on its own (owner
   2026-10-19), so the build needs this map. It publishes only tier names and opaque task ids
   (private ids are salted with the private seed), never task text, DOIs or gold.

Private tasks are read from article custody through ``generator.generate`` (hash-checked
against custody MANIFEST.json), so the map is the generator's own ``tier`` key. ``--check``
writes nothing and exits 1 if a committed file differs from what would be written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARENA))
sys.path.insert(0, str(ARENA.parents[1]))

import generator  # noqa: E402

TIERED_VERSIONS = ("v3", "v4")


def private_seed(version: str) -> int:
    return int((ARENA / "task_sets" / version / ".private_seed").read_text(encoding="utf-8").strip())


def revealed_ground_truth(version: str) -> dict:
    out = {}
    for env in generator.generate(version, 0, split="revealed"):
        out[env["task_id"]] = {"envelope": env, "ground_truth": generator.ground_truth(env["task_id"])}
    return out


def tier_lines(version: str) -> list[dict]:
    rows = []
    for split, seed in (("revealed", 0), ("private", private_seed(version))):
        for env in generator.generate(version, seed, split=split):
            tier = generator.ground_truth(env["task_id"])["tier"]
            if tier not in generator.TIERS:
                raise SystemExit(f"{env['task_id']}: unknown tier {tier!r}")
            rows.append({"task_id": env["task_id"], "tier": tier, "split": split})
    ids = [r["task_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{version}: duplicate task ids in the tier map")
    return sorted(rows, key=lambda r: (generator.TIERS.index(r["tier"]), r["task_id"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", required=True, choices=generator.KNOWN_TASK_SET_VERSIONS)
    ap.add_argument("--check", action="store_true", help="compare with the committed files; write nothing")
    a = ap.parse_args(argv)

    base = ARENA / "task_sets" / a.version
    files = {base / "_ground_truth.json": json.dumps(revealed_ground_truth(a.version), indent=2) + "\n"}
    if a.version in TIERED_VERSIONS:
        rows = tier_lines(a.version)
        files[base / "tiers.jsonl"] = "".join(json.dumps(r) + "\n" for r in rows)
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["tier"]] = counts.get(r["tier"], 0) + 1
        print(f"{a.version}: {len(rows)} tasks; " + ", ".join(f"{t} {n}" for t, n in counts.items()))

    def same(path: Path, text: str) -> bool:
        # Content, not bytes: a Windows checkout turns LF into CRLF, and v1/v2's file was
        # committed without a trailing newline.
        if not path.exists():
            return False
        current = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            return json.loads(current) == json.loads(text)
        return current.splitlines() == text.splitlines()

    stale = []
    for path, text in files.items():
        if a.check:
            if not same(path, text):
                stale.append(path.name)
        elif not same(path, text):
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {path.relative_to(ARENA.parents[1])}")
    if stale:
        print(f"{a.version}: stale or missing: {', '.join(stale)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

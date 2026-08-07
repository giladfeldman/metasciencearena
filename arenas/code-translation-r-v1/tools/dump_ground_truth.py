"""Materialize task_sets/v1/_ground_truth.json for code-translation-r-v1.

The leaderboard's Node build reads this tracked file rather than importing the
Python generator, so it has to be regenerated whenever the source scripts or
gold change.

Every entry is routed through `framework.holdout.redact_ground_truth_entry`,
which strips gold from held-out tasks. That is the single redaction boundary
(framework/holdout.py) — never hand-roll stripping here.

    python arenas/code-translation-r-v1/tools/dump_ground_truth.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
SA_ROOT = ARENA_DIR.parents[1]
OUT_PATH = ARENA_DIR / "task_sets" / "v1" / "_ground_truth.json"

sys.path.insert(0, str(SA_ROOT))
sys.path.insert(0, str(ARENA_DIR))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


generator = _load("_xlat_generator", ARENA_DIR / "generator.py")
from framework.holdout import redact_ground_truth_entry  # noqa: E402


def main() -> None:
    entries: dict[str, dict] = {}
    for split, seed in (("revealed", 0), ("private", 1)):
        for env in generator.generate("v1", seed, split):
            gt = generator.ground_truth(env["task_id"])
            # `gold_statistics` is the answer key; holdout.py strips any field
            # whose name starts with "gold" for held-out tasks.
            entries[env["task_id"]] = redact_ground_truth_entry(
                {"envelope": env, "ground_truth": dict(gt)}
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    n_public = sum(1 for e in entries.values()
                   if e["envelope"].get("visibility") == "public")
    kb = OUT_PATH.stat().st_size / 1024
    print(f"wrote {len(entries)} entries ({n_public} public) to {OUT_PATH} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()

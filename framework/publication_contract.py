"""The publication contract: a record reaches the site only with its evidence (plan D, 2026-11-01).

From schema v13 a record must carry what produced it: its environment hash, and for a language-model
player the hash of its call settings (whose file must exist) plus, when those settings say the call
went through a command-line tool, the per-call block (`model_call`). A record older than v13 is
accepted only as LEGACY: written before the inventory was frozen, and no more legacy records per
arena than the inventory counted. That closes the door the 2026-11-01 diagnosis found open: a
record written by any path that skips the v13 runner would otherwise publish with no evidence
and nothing would fail.

The site build enforces the same rule (leaderboard-app/scripts/lib/publication-contract.mjs; parity:
framework/tests/test_publication_contract.py). QA gate C12 runs `check()` over every live run file.

    python -m framework.publication_contract check
    python -m framework.publication_contract freeze --reason "<why the inventory is re-frozen>"

Re-freezing is for the one release that lands the v13 runner (it admits the records already made on
older code). After that a re-freeze is a visible decision in a commit, never a way to clear the gate.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Callable, Iterable

REPO = Path(__file__).resolve().parents[1]
ARENAS = REPO / "arenas"
INVENTORY = REPO / "run_env" / "legacy_inventory.json"
CALL_CONFIGS = REPO / "run_env" / "call_configs"
CONTRACT_SINCE = 13


def iter_live_records(arenas: Path = ARENAS) -> Iterable[tuple[str, Path, dict]]:
    """(arena_id, run file, record) for every durable run file, as QA's C-checks read them.

    Skips `_archive/` and `_pilot_archive/` (superseded evidence), `*.retry-r<N>.jsonl` (transient
    retry temps) and `*__rescored.jsonl` (local demo rescores the site build never reads).
    """
    for arena_dir in sorted(p for p in arenas.iterdir() if p.is_dir()):
        for f in sorted(arena_dir.glob("runs/*/*.jsonl")):
            if "_archive" in f.parts or "_pilot_archive" in f.parts:
                continue
            if ".retry-r" in f.name or f.name.endswith("__rescored.jsonl"):
                continue
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip():
                    try:
                        yield arena_dir.name, f, json.loads(line)
                    except ValueError:
                        continue


def load_inventory(path: Path = INVENTORY) -> dict:
    """The frozen legacy inventory; a missing file is an empty one (every legacy record refused)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"frozen_utc": None, "arenas": {}}


def _schema(rec: dict) -> int:
    """The record's shape version; an integral float (13.0) is that version, as JSON readers see it."""
    v = rec.get("schema_version")
    if isinstance(v, bool):
        return 0
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return 0


def has_call_block(mc) -> bool:
    """A model_call that records at least one call: a block with a launch, or a non-empty list of them.

    `{}`, `{"blocks": []}` and a block without `launch` are not evidence of anything (review
    2026-11-01: an empty aggregate read on the site as "requested and confirmed").
    """
    if not isinstance(mc, dict):
        return False
    if "blocks" in mc:
        blocks = mc["blocks"]
        return isinstance(blocks, list) and bool(blocks) and all(has_call_block(b) for b in blocks)
    return isinstance(mc.get("launch"), dict) and bool(mc["launch"])


def _call_config_from_store(sha: str) -> dict | None:
    try:
        return json.loads((CALL_CONFIGS / f"{sha}.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def record_problem(rec: dict, *, frozen_utc: str | None,
                   call_config_of: Callable[[str], dict | None] = _call_config_from_store) -> str | None:
    """Why this one record may not be published, or None. The per-arena legacy count is separate."""
    if _schema(rec) >= CONTRACT_SINCE:
        if not rec.get("env_sha"):
            return "v13 record without env_sha"
        if rec.get("player_type") != "ai-model":
            return None
        sha = rec.get("call_config_sha256")
        if not sha:
            return "v13 language-model record without call_config_sha256"
        cfg = call_config_of(sha)
        if cfg is None:
            return f"call settings {sha[:12]} cited but not in run_env/call_configs/"
        if str(cfg.get("api_type", "")).startswith("cli:") and not has_call_block(rec.get("model_call")):
            return "v13 command-line-tool record without its model_call block"
        return None
    ts = rec.get("timestamp_utc")
    if not frozen_utc:
        return "legacy (pre-v13) record and no frozen legacy inventory"
    if not isinstance(ts, str) or not ts or ts >= frozen_utc:
        return f"legacy (pre-v13) record written after the inventory was frozen ({frozen_utc})"
    return None


def check(arenas: Path = ARENAS, inventory: dict | None = None) -> list[str]:
    """Every contract failure over the live corpus, one line per (arena, problem)."""
    inv = load_inventory() if inventory is None else inventory
    frozen = inv.get("frozen_utc")
    allowed = inv.get("arenas", {})
    legacy: collections.Counter = collections.Counter()
    problems: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    seen_any = False
    for arena_id, f, rec in iter_live_records(arenas):
        seen_any = True
        if _schema(rec) < CONTRACT_SINCE:
            legacy[arena_id] += 1
        why = record_problem(rec, frozen_utc=frozen)
        if why:
            problems[(arena_id, why)].append(f"{f.name}:{rec.get('task_id')}")
    fails = [f"{a}: {why} -- {len(where)} record(s), e.g. {', '.join(where[:3])}"
             for (a, why), where in sorted(problems.items())]
    for arena_id, n in sorted(legacy.items()):
        cap = int(allowed.get(arena_id, 0))
        if n > cap:
            fails.append(f"{arena_id}: {n} legacy (pre-v13) records, the frozen inventory allows {cap}")
    if not seen_any:
        fails.append(f"no run records found under {arenas} -- an empty corpus is not a pass")
    return fails


def freeze(reason: str, arenas: Path = ARENAS, path: Path = INVENTORY) -> dict:
    """Count today's legacy records per arena and stamp the time; every later legacy record is refused."""
    counts: collections.Counter = collections.Counter()
    latest = ""
    for arena_id, _f, rec in iter_live_records(arenas):
        if _schema(rec) < CONTRACT_SINCE:
            counts[arena_id] += 1
            latest = max(latest, str(rec.get("timestamp_utc") or ""))
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prev = load_inventory(path)
    inv = {
        "description": "Pre-v13 run records admitted to the site (framework/publication_contract.py). "
                       "A legacy record must predate frozen_utc, and an arena may hold no more legacy "
                       "records than counted here.",
        "frozen_utc": now,
        "reason": reason,
        "latest_legacy_timestamp_utc": latest or None,
        "total": sum(counts.values()),
        "arenas": dict(sorted(counts.items())),
        "previous_frozen_utc": prev.get("frozen_utc"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inv, indent=2) + "\n", encoding="utf-8")
    return inv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    fr = sub.add_parser("freeze")
    fr.add_argument("--reason", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "freeze":
        inv = freeze(args.reason)
        print(f"froze {inv['total']} legacy records over {len(inv['arenas'])} arenas at {inv['frozen_utc']}")
        return 0
    fails = check()
    for f in fails:
        print(f"  - {f}")
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

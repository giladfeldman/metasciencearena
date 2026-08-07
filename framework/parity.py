"""Revealed/private parity checker.

Generates BOTH benchmark splits for an arena and verifies they cover the same
difficulty x injected-mistake matrix, so the public Open Benchmark and the
private (official-leaderboard) suite stay equally hard. This is the technical
guarantee behind the user requirement that the revealed and private suites
"parallel in complexity and difficulty, both covering all array of opportunities
and injected mistakes." See contract/README.md "Revealed/private dual benchmark".

Registry-free: gold is read from the generator's in-process ground-truth cache
(the same `_GROUND_TRUTH_CACHE` convention that tools/build_gold.py snapshots),
so a parity check needs neither the article-finder registry nor any secret beyond
the private seed.

CLI: `python tools/check_parity.py <arena-id> [task_set_version]`.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# No module-level REPO_ROOT / ARENAS_ROOT here: both were dead (nothing in this
# module or any caller read them) and both would have resolved into site-packages
# under a real install. Callers pass an arena directory; `framework.paths`
# resolves roots for the CLI.

# Gold items label their injected mistake under one of these keys (None/"" = a
# clean control with no injected mistake). Add new arenas' field names here.
_KIND_KEYS = ("deception_kind", "deviation_kind", "mistake_kind", "injected_mistake")

# Deterministic offset used to derive a stand-in private seed for LOCAL parity
# checks when the real gitignored .private_seed is absent. Production parity runs
# (and the official suite) must use the real secret; the report flags the fallback.
_DEV_PRIVATE_OFFSET = 1_000_003


class ParityError(Exception):
    pass


@dataclass
class ParityReport:
    arena_id: str
    task_set_version: str
    ok: bool
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    revealed_cells: dict = field(default_factory=dict)
    private_cells: dict = field(default_factory=dict)
    revealed_categories: dict = field(default_factory=dict)
    private_categories: dict = field(default_factory=dict)

    def summary(self) -> str:
        head = "PARITY OK" if self.ok else "PARITY FAIL"
        lines = [f"[{head}] {self.arena_id} {self.task_set_version}"]
        lines.append(
            f"  cells: revealed={len(self.revealed_cells)} private={len(self.private_cells)} | "
            f"mistake categories: revealed={sorted(self.revealed_categories)} "
            f"private={sorted(self.private_categories)}"
        )
        for note in self.notes:
            lines.append(f"  note: {note}")
        for prob in self.problems:
            lines.append(f"  PROBLEM: {prob}")
        return "\n".join(lines)


def _load_manifest(arena_dir: Path) -> dict:
    with (arena_dir / "arena.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _import_generator(arena_dir: Path):
    path = arena_dir / "generator.py"
    spec = importlib.util.spec_from_file_location(f"_parity_generator_{arena_dir.name}", path)
    if spec is None or spec.loader is None:
        raise ParityError(f"could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_seed(manifest: dict, arena_dir: Path, task_set_version: str, split: str) -> tuple[int, str | None]:
    """Return (seed, note). `note` is non-None when a dev fallback was used."""
    splits = manifest.get("benchmark_splits") or {}
    if split == "revealed":
        return int(splits["revealed"]["seed"]), None
    if split != "private":
        raise ParityError(f"unknown split {split!r}")
    priv = splits["private"]
    source = priv.get("seed_source", "secret_file")
    if source == "env":
        raw = os.environ.get("SCIENCEARENA_PRIVATE_SEED")
        if raw is not None and raw.strip():
            return int(raw), None
    else:  # secret_file
        seed_file = arena_dir / "task_sets" / task_set_version / ".private_seed"
        if seed_file.exists():
            return int(seed_file.read_text(encoding="utf-8").strip()), None
    # Dev fallback: distinct-but-deterministic so local parity is meaningful.
    fallback = int(splits["revealed"]["seed"]) + _DEV_PRIVATE_OFFSET
    return fallback, (
        "private seed not found (no .private_seed / SCIENCEARENA_PRIVATE_SEED); "
        f"used dev-fallback seed {fallback}. Official runs MUST supply the real secret."
    )


def _mistake_labels(gold) -> list[str]:
    """Extract injected-mistake labels from one task's gold (arena-agnostic).

    Two conventions, checked in order:
      1. Preferred & uniform: gold carries a top-level `mistake_kinds` list (the
         labels injected into this task; empty / ["clean"] => a clean control).
         Works for any gold shape, including field-map arenas. A scalar
         `mistake_kind` is also accepted.
      2. Fallback: scan gold items (gold["items"], the first list value, or gold
         itself) for a per-item `*_kind` field (deception_kind / deviation_kind /
         mistake_kind / injected_mistake). One label per item.
    """
    if isinstance(gold, dict) and isinstance(gold.get("mistake_kinds"), list):
        ks = gold["mistake_kinds"]
        if not ks:
            return ["clean"]
        return ["clean" if k in (None, "", False) else str(k) for k in ks]
    if isinstance(gold, dict) and "mistake_kind" in gold and not isinstance(gold.get("items"), list):
        k = gold["mistake_kind"]
        return ["clean" if k in (None, "", False) else str(k)]

    if isinstance(gold, dict):
        items = gold.get("items")
        if items is None:
            # Some arenas key records differently; treat any list value as items.
            list_vals = [v for v in gold.values() if isinstance(v, list)]
            items = list_vals[0] if list_vals else [gold]
    elif isinstance(gold, list):
        items = gold
    else:
        items = []
    labels: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        kind = None
        for key in _KIND_KEYS:
            if key in it:
                kind = it[key]
                break
        labels.append("clean" if kind in (None, "", False) else str(kind))
    return labels


def _collect_coverage(gen_mod, task_set_version: str, seed: int, split: str,
                      match_axes: list[str]) -> tuple[Counter, Counter]:
    envs = list(gen_mod.generate(task_set_version, seed, split=split))
    cache = dict(getattr(gen_mod, "_GROUND_TRUTH_CACHE", {}))
    cells: Counter = Counter()
    categories: Counter = Counter()
    for env in envs:
        diff = env.get("difficulty", {})
        cells[tuple(diff.get(a) for a in match_axes)] += 1
        gold = cache.get(env["task_id"], {})
        for label in _mistake_labels(gold):
            categories[label] += 1
    return cells, categories


def _within_tolerance(a: int, b: int, tol: float) -> bool:
    if a == b:
        return True
    return abs(a - b) / max(a, b, 1) <= tol


def check_parity(arena_dir: Path, task_set_version: str = "v1") -> ParityReport:
    manifest = _load_manifest(arena_dir)
    arena_id = manifest.get("arena_id", arena_dir.name)
    splits = manifest.get("benchmark_splits")
    if not splits:
        return ParityReport(
            arena_id=arena_id, task_set_version=task_set_version, ok=True,
            notes=["single-suite arena (no benchmark_splits) — parity check skipped"],
        )

    parity = splits.get("parity", {})

    # An INDEPENDENT holdout (real third-party artifacts under real_holdout_dir)
    # cannot mirror the revealed difficulty grid: real material arrives with
    # whatever constructs its authors happened to use, and hand-picking it to fit
    # our tier x language template would stop it being a holdout at all. Enforcing
    # cell parity there would either fail forever or get quietly relaxed until it
    # proved nothing — so the arena declares the shape and the check steps aside,
    # loudly, in the report note.
    if parity.get("independent_holdout"):
        return ParityReport(
            arena_id=arena_id, task_set_version=task_set_version, ok=True,
            notes=["independent-holdout arena (private split is a real, separate "
                   "corpus — not the same generator under a secret seed) — "
                   "cell/category parity intentionally not enforced; revealed-split "
                   "symmetry is gated by `framework audit`"],
        )

    match_axes = parity.get("match_axes", [])
    match_categories = parity.get("match_categories", True)
    tol = float(parity.get("count_tolerance", 0.2))

    gen = _import_generator(arena_dir)
    rev_seed, rev_note = resolve_seed(manifest, arena_dir, task_set_version, "revealed")
    priv_seed, priv_note = resolve_seed(manifest, arena_dir, task_set_version, "private")

    rev_cells, rev_cats = _collect_coverage(gen, task_set_version, rev_seed, "revealed", match_axes)
    priv_cells, priv_cats = _collect_coverage(gen, task_set_version, priv_seed, "private", match_axes)

    problems: list[str] = []
    notes = [n for n in (rev_note, priv_note) if n]

    if rev_seed == priv_seed:
        problems.append(
            f"revealed and private resolved to the SAME seed ({rev_seed}); the private suite "
            "would be identical to the public one."
        )

    # Difficulty-cell coverage.
    only_rev = set(rev_cells) - set(priv_cells)
    only_priv = set(priv_cells) - set(rev_cells)
    if only_rev:
        problems.append(f"difficulty cells only in revealed: {sorted(map(str, only_rev))}")
    if only_priv:
        problems.append(f"difficulty cells only in private: {sorted(map(str, only_priv))}")
    for cell in set(rev_cells) & set(priv_cells):
        if not _within_tolerance(rev_cells[cell], priv_cells[cell], tol):
            problems.append(
                f"difficulty cell {cell}: revealed={rev_cells[cell]} private={priv_cells[cell]} "
                f"exceeds count_tolerance={tol}"
            )

    # Injected-mistake category coverage.
    if match_categories:
        only_rev_c = set(rev_cats) - set(priv_cats)
        only_priv_c = set(priv_cats) - set(rev_cats)
        if only_rev_c:
            problems.append(f"injected-mistake categories only in revealed: {sorted(only_rev_c)}")
        if only_priv_c:
            problems.append(f"injected-mistake categories only in private: {sorted(only_priv_c)}")
        for cat in set(rev_cats) & set(priv_cats):
            if not _within_tolerance(rev_cats[cat], priv_cats[cat], tol):
                problems.append(
                    f"injected-mistake category '{cat}': revealed={rev_cats[cat]} "
                    f"private={priv_cats[cat]} exceeds count_tolerance={tol}"
                )

    return ParityReport(
        arena_id=arena_id, task_set_version=task_set_version, ok=not problems,
        problems=problems, notes=notes,
        revealed_cells={str(k): v for k, v in rev_cells.items()},
        private_cells={str(k): v for k, v in priv_cells.items()},
        revealed_categories=dict(rev_cats), private_categories=dict(priv_cats),
    )

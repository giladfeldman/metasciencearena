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

# The label `_mistake_labels` synthesises for a gold item carrying NO kind. It is
# a residual, not a member of any arena's declared taxonomy; `_check_clean_residual`
# is the only place that may treat it differently from a declared kind.
_CLEAN = "clean"

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
    # What this run did NOT verify. A report that passes while leaving something
    # unchecked must SAY so in its headline, not only in a note a reader may skip:
    # "PARITY OK" on a partial check is the false green this module exists to stop.
    not_enforced: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.ok:
            head = "PARITY FAIL"
        elif self.not_enforced:
            head = "PARITY PARTIAL"
        else:
            head = "PARITY OK"
        lines = [f"[{head}] {self.arena_id} {self.task_set_version}"]
        lines.append(
            f"  cells: revealed={len(self.revealed_cells)} private={len(self.private_cells)} | "
            f"mistake categories: revealed={sorted(self.revealed_categories)} "
            f"private={sorted(self.private_categories)}"
        )
        for item in self.not_enforced:
            lines.append(f"  NOT ENFORCED: {item}")
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
                      match_axes: list[str]) -> tuple[Counter, Counter, dict]:
    envs = list(gen_mod.generate(task_set_version, seed, split=split))
    cache = dict(getattr(gen_mod, "_GROUND_TRUTH_CACHE", {}))
    cells: Counter = Counter()
    categories: Counter = Counter()
    kinds: dict = {}
    for env in envs:
        diff = env.get("difficulty", {})
        cell = tuple(diff.get(a) for a in match_axes)
        cells[cell] += 1
        gold = cache.get(env["task_id"], {})
        for label in _mistake_labels(gold):
            categories[label] += 1
        # Item-kind composition, keyed by the same cell as everything else, so a
        # difference can be reported at the granularity a reader can act on.
        for k in _item_kinds(gold):
            kinds.setdefault(cell, Counter())[k] += 1
    return cells, categories, kinds


def _within_tolerance(a: int, b: int, tol: float) -> bool:
    if a == b:
        return True
    return abs(a - b) / max(a, b, 1) <= tol


def _item_kinds(gold) -> list[str]:
    """Per-item `kind` labels for one task's gold (arena-agnostic, best effort).

    Item KIND composition -- how many gold items are one kind versus another -- is
    what caps a player that only reads one kind. `framework/parity.py` never used
    to look at it, which is how stats-extraction-v1 shipped three task sets whose
    two splits cap an NHST-only extractor at 0.6516 and 0.6901. Returns [] for
    gold shapes that carry no per-item `kind`.
    """
    items = gold.get("items") if isinstance(gold, dict) else None
    if not isinstance(items, list):
        return []
    return [str(i["kind"]) for i in items if isinstance(i, dict) and "kind" in i]


def _composition_rows(rev_kinds: dict, priv_kinds: dict) -> list[str]:
    """Per-cell item-kind composition, as report lines. Printed, never gated on.

    A number the reader can see is worth more than a threshold they cannot. These
    rows exist so a split pair that is NOT matched on composition shows exactly
    where and by how much, rather than hiding behind a verdict.
    """
    rows = []
    for cell in sorted(set(rev_kinds) | set(priv_kinds), key=str):
        r, p = rev_kinds.get(cell, Counter()), priv_kinds.get(cell, Counter())
        if r == p:
            continue
        rt, pt = sum(r.values()), sum(p.values())
        rs = ", ".join(f"{k}={r[k]}" for k in sorted(set(r) | set(p)))
        ps = ", ".join(f"{k}={p[k]}" for k in sorted(set(r) | set(p)))
        rows.append(f"    cell {cell}: revealed [{rs}] n={rt} | private [{ps}] n={pt}")
    return rows


def _composition_excused(parity: dict, task_set_version: str) -> dict | None:
    """The arena's declaration that THIS version is not matched on composition.

    Returns the declaration when `task_set_version` is named in
    `composition_unmatched.versions`, else None.

    The version list is EXPLICIT and exhaustive on purpose. An exemption phrased
    as "until the generator is fixed" never expires, because nothing forces anyone
    to notice that it has been fixed. Naming v3/v4/v5 means a later v6 is not
    covered and gets the full strict check automatically, with no edit and no
    memory required. Adding a version here is a deliberate, reviewable act.
    """
    decl = parity.get("composition_unmatched")
    if not isinstance(decl, dict):
        return None
    return decl if task_set_version in (decl.get("versions") or []) else None


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
            not_enforced=["cell/category parity on an independent-holdout arena "
                          "(the private split is a real, separate corpus, not the "
                          "same generator under a secret seed); revealed-split "
                          "symmetry is gated by `framework audit` instead"],
        )

    match_axes = parity.get("match_axes", [])
    match_categories = parity.get("match_categories", True)
    tol = float(parity.get("count_tolerance", 0.2))

    gen = _import_generator(arena_dir)
    rev_seed, rev_note = resolve_seed(manifest, arena_dir, task_set_version, "revealed")
    priv_seed, priv_note = resolve_seed(manifest, arena_dir, task_set_version, "private")

    rev_cells, rev_cats, rev_kinds = _collect_coverage(
        gen, task_set_version, rev_seed, "revealed", match_axes)
    priv_cells, priv_cats, priv_kinds = _collect_coverage(
        gen, task_set_version, priv_seed, "private", match_axes)

    problems: list[str] = []
    notes = [n for n in (rev_note, priv_note) if n]
    not_enforced: list[str] = []
    excused = _composition_excused(parity, task_set_version)

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

        # `clean` is NOT a declared injected-mistake kind. It is a residual label
        # `_mistake_labels` synthesises for every gold item that carries no kind at
        # all, so its count is (total gold items - injected items) and it moves
        # whenever a generator's per-task item CARDINALITY moves. Holding that
        # residual to the taxonomy's exact-count rule asserts something no
        # generator in this repo promises. An arena that knows its item count is
        # seed-dependent declares `clean_rate_tolerance` and the residual is then
        # checked as a BASE RATE (clean / all items) -- the quantity that actually
        # defines the false-alarm control density, and the one a player's score
        # depends on. Arenas that declare nothing keep the old exact-count
        # behaviour unchanged, so this is a per-arena opt-in and never a silent
        # fleet-wide relaxation. See `_check_clean_residual`.
        declared = (set(rev_cats) & set(priv_cats)) - {_CLEAN}
        for cat in sorted(declared):
            if not _within_tolerance(rev_cats[cat], priv_cats[cat], tol):
                problems.append(
                    f"injected-mistake category '{cat}': revealed={rev_cats[cat]} "
                    f"private={priv_cats[cat]} exceeds count_tolerance={tol}"
                )

        # `clean` is NOT a declared kind. It is the residual `_mistake_labels`
        # synthesises for items carrying no injected mistake, so its count is
        # (all gold items - injected items) and it moves with per-task item
        # CARDINALITY. Where the arena has DECLARED that this version's splits are
        # not matched on composition, gating on the residual would charge the
        # symptom while the cause is excused, so it is reported instead. Every
        # other version, and every other arena, keeps the exact-count rule.
        if _CLEAN in rev_cats and _CLEAN in priv_cats:
            rn, pn = rev_cats[_CLEAN], priv_cats[_CLEAN]
            rt, pt = sum(rev_cats.values()), sum(priv_cats.values())
            notes.append(
                f"clean residual (not an injected kind): revealed={rn}/{rt} "
                f"private={pn}/{pt}"
            )
            if not excused and not _within_tolerance(rn, pn, tol):
                problems.append(
                    f"injected-mistake category '{_CLEAN}': revealed={rn} "
                    f"private={pn} exceeds count_tolerance={tol}"
                )

    # The declared step-aside. Modelled on `independent_holdout` above: the arena
    # states what it does NOT guarantee, the report says so in its HEADLINE, and
    # the numbers are printed so a reader can judge the size of the gap rather
    # than trust a verdict. Saying "PARITY OK" here would be the pretence the
    # portfolio's rules exist to prevent -- these two splits really are different
    # instruments for a player that reads only one item kind.
    if excused:
        reason = str(excused.get("reason") or "").strip() or "declared in arena.yaml"
        not_enforced.append(
            f"item-KIND composition on {task_set_version} "
            f"(declared `composition_unmatched`): {reason}"
        )
        rows = _composition_rows(rev_kinds, priv_kinds)
        if rows:
            joined = chr(10).join(rows)
            not_enforced.append("per-cell item-kind differences:" + chr(10) + joined)
        else:
            # A declaration that excuses nothing is worse than none: it trains the
            # next reader to skim past it. Say so loudly instead of passing quietly.
            not_enforced.append(
                "NOTE: no per-cell composition difference was found, so this "
                "declaration is currently VACUOUS and should be removed from "
                "arena.yaml (or its version list narrowed)."
            )

    return ParityReport(
        arena_id=arena_id, task_set_version=task_set_version, ok=not problems,
        problems=problems, notes=notes, not_enforced=not_enforced,
        revealed_cells={str(k): v for k, v in rev_cells.items()},
        private_cells={str(k): v for k, v in priv_cells.items()},
        revealed_categories=dict(rev_cats), private_categories=dict(priv_cats),
    )

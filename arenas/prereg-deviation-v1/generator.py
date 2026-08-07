"""Generator for prereg-deviation-v1 (regcheck-style arena).

Builds (preregistration, paper) pairs with KNOWN deviations injected per
dimension, mirroring regcheck's comparison dimensions. All-procedural and
deterministic from (task_set_version, seed).

Dual-benchmark (revealed/private): both splits run the IDENTICAL tier matrix and
assign deviation MODES/KINDS deterministically (index-driven, seed-independent),
so every split covers the full array of injected deviations equally — this is
what makes framework/parity.py pass at count_tolerance 0. Only the concrete
option TEXT is seed-driven, so revealed and private content still differ.

Gold is regenerated from the seed and served from the in-process cache (no
article-finder registry needed): the secret is the private seed, not a stored
answer key. The revealed seed is committed in arena.yaml#benchmark_splits.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import yaml

ARENA_DIR = Path(__file__).resolve().parent
CATALOGS_DIR = ARENA_DIR / "catalogs"

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

# Paper-sentence mode -> whether it constitutes a deviation from the prereg.
_DEVIATING_MODES = {"subtle", "deviated"}
_T6_MODE_CYCLE = ["consistent", "paraphrase", "subtle", "deviated"]


def _load_dimensions() -> list[dict]:
    with (CATALOGS_DIR / "dimensions.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _render_dim(dim: dict, rng: random.Random, mode: str) -> tuple[str, str, dict]:
    planned = rng.choice(dim["planned_options"])
    actual = rng.choice(dim["actual_options"])
    prereg_s = dim["prereg"].format(planned=planned, actual=actual)
    paper_s = dim[mode].format(planned=planned, actual=actual)
    deviates = mode in _DEVIATING_MODES
    gold = {
        "dimension": dim["id"],
        "deviation": deviates,
        "deviation_kind": dim["deviation_kind"] if deviates else None,
    }
    return prereg_s, paper_s, gold


def _assemble(task_id, tier, dims_modes, rng, split, visibility) -> tuple[dict, dict]:
    prereg_parts, paper_parts, gold = [], [], []
    for dim, mode in dims_modes:
        ps, qs, g = _render_dim(dim, rng, mode)
        prereg_parts.append(ps)
        paper_parts.append(qs)
        gold.append(g)
    n_dev = sum(1 for g in gold if g["deviation"])
    envelope = {
        "task_id": task_id,
        "arena_id": "prereg-deviation-v1",
        "task_set_version": "v1",
        "split": split,
        "visibility": visibility,
        "difficulty": {"tier": tier, "n_deviations": n_dev},
        "input": {
            "preregistration": "\n".join(prereg_parts),
            "paper": "\n".join(paper_parts),
            "dimensions": [d["id"] for d, _ in dims_modes],
        },
    }
    return envelope, {"dimensions": gold}


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    visibility = "public" if split == "revealed" else "held_out"
    dims = _load_dimensions()
    n = len(dims)

    def emit(tier, idx, dims_modes):
        tid = f"pd-t{tier}-{idx}-s{seed}"
        rng = random.Random(_seed_int(task_set_version, seed, tier, idx))
        env, gt = _assemble(tid, tier, dims_modes, rng, split, visibility)
        _GROUND_TRUTH_CACHE[tid] = gt
        return env

    # T1: every dimension consistent (verbatim). No deviations.
    for k in range(4):
        yield emit(1, k, [(d, "consistent") for d in dims])

    # T2: every dimension paraphrased but still consistent -> false-alarm trap.
    for k in range(4):
        yield emit(2, k, [(d, "paraphrase") for d in dims])

    # T3: exactly one dimension clearly deviates (cycling through every dimension
    # so every deviation_kind is covered), the rest verbatim-consistent.
    for i in range(n):
        modes = [(d, "deviated" if j == i else "consistent") for j, d in enumerate(dims)]
        yield emit(3, i, modes)

    # T4: exactly one dimension subtly deviates amid paraphrased (reworded) others.
    for i in range(n):
        modes = [(d, "subtle" if j == i else "paraphrase") for j, d in enumerate(dims)]
        yield emit(4, i, modes)

    # T5: several co-occurring deviations (deterministic cycling offset), rest paraphrased.
    for k in range(4):
        deviating = {(k * 3 + o) % n for o in range(3)}
        modes = [(d, "deviated" if j in deviating else "paraphrase") for j, d in enumerate(dims)]
        yield emit(5, k, modes)

    # T6: full composition — a deterministic mix of all four modes across dimensions.
    for k in range(2):
        modes = [(d, _T6_MODE_CYCLE[(j + k) % len(_T6_MODE_CYCLE)]) for j, d in enumerate(dims)]
        yield emit(6, k, modes)


def ground_truth(task_id: str) -> dict:
    """Return gold for a task. Regenerated from seed via the in-process cache.

    The runner always calls generate() before ground_truth(); this arena needs
    no external gold registry (unlike stats-extraction-v1) because the secret is
    the private seed, not a stored answer key.
    """
    if task_id not in _GROUND_TRUTH_CACHE:
        raise KeyError(
            f"No cached gold for {task_id!r}; call generate() for the matching "
            "split/seed before ground_truth()."
        )
    return _GROUND_TRUTH_CACHE[task_id]

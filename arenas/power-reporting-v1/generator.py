"""Generator for power-reporting-v1 (metacheck-style FIELD-MAP arena).

Builds methods/power excerpts with KNOWN power-analysis content, mirroring
metacheck's `power` module. The player must (a) detect whether a power analysis
is present, (b) classify its kind (apriori / sensitivity / posthoc), and (c)
extract the structured fields (test, sample, alpha, power, effect_size,
software). All-procedural and deterministic from (task_set_version, seed).

Dual-benchmark (revealed/private): both splits run the IDENTICAL tier matrix and
assign the injected-mistake KIND DETERMINISTICALLY (index-driven cycling through
ALL kinds, NOT rng.choice), so every split covers the full array of injected
mistakes equally — this is what makes framework/parity.py pass. Only the concrete
field VALUES and template WORDING are seed-driven, so revealed and private
content still differ.

Injected mistake_kinds (cycled deterministically):
  - posthoc_as_apriori : a post-hoc power analysis worded as if a-priori (trap).
  - missing_fields     : some structured fields are absent from the excerpt.
  - no_power_analysis  : the excerpt reports NO power analysis at all.
Clean variants (mistake_kinds == []  ->  parity bucket "clean"):
  - a complete a-priori power analysis,
  - a complete sensitivity power analysis,
  - a complete, correctly-labelled post-hoc power analysis.

Gold is regenerated from the seed and served from the in-process cache (no
external registry): the secret is the private seed, not a stored answer key. The
revealed seed is committed in arena.yaml#benchmark_splits.
"""
from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

import yaml

ARENA_DIR = Path(__file__).resolve().parent
CATALOGS_DIR = ARENA_DIR / "catalogs"

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

# The full, ordered set of extractable structured fields (the field-map).
FIELDS = ["test", "sample", "alpha", "power", "effect_size", "software"]

# The correctly-labelled (clean) power-analysis kinds, cycled across clean tasks.
CLEAN_KINDS = ["apriori", "sensitivity", "posthoc"]

# The injected-mistake kinds, cycled deterministically so every split covers all.
#   sensitivity_as_apriori (added 2026-06-29): a SENSITIVITY analysis (solves for the
#   smallest detectable effect at a fixed N) worded as if it were a-priori. TRUE kind
#   is sensitivity; the legitimate look-alike is the clean `sensitivity` excerpt.
MISTAKE_KINDS = [
    "posthoc_as_apriori",
    "missing_fields",
    "no_power_analysis",
    "sensitivity_as_apriori",
]

# Which fields the `missing_fields` mistake drops, keyed by a cycling index so
# different missing-field tasks omit different (non-empty, non-total) subsets.
_MISSING_SUBSETS = [
    ["effect_size", "software"],
    ["alpha", "power"],
    ["software"],
    ["effect_size"],
    ["alpha", "power", "software"],
]


def _load_catalog() -> dict:
    with (CATALOGS_DIR / "power_statements.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _pick_fields(cat: dict, rng: random.Random) -> dict:
    """Choose a concrete value for every structured field (seed-driven)."""
    opts = cat["options"]
    return {f: rng.choice(opts[f]) for f in FIELDS}


def _render(template: str, values: dict) -> str:
    """Fill a {placeholder} template, collapsing whitespace from YAML folding."""
    text = template.format(**values)
    return re.sub(r"\s+", " ", text).strip()


def _clean_excerpt(cat: dict, kind: str, values: dict, rng: random.Random) -> str:
    template = rng.choice(cat["kinds"][kind]["templates"])
    return _render(template, values)


def _build_no_power(cat: dict, values: dict, rng: random.Random) -> str:
    """Two filler sentences that mention methods but contain NO power analysis."""
    fillers = list(cat["no_power_fillers"])
    rng.shuffle(fillers)
    chosen = fillers[:2]
    return " ".join(_render(t, values) for t in chosen)


def _make_gold(has_pa: bool, kind, values: dict, present_fields: list[str],
               mistake_kinds: list[str]) -> dict:
    fields = {f: values[f] for f in present_fields} if has_pa else {}
    return {
        "has_power_analysis": has_pa,
        "kind": kind if has_pa else None,
        "fields": fields,
        "mistake_kinds": mistake_kinds,
    }


def _task(task_set_version, seed, tier, idx, kind_or_mistake, *, mistake: bool,
          split: str, visibility: str):
    """Build one (envelope, gold) pair for the given tier/idx/kind.

    `kind_or_mistake` is a clean power kind when mistake=False, otherwise an
    injected-mistake label from MISTAKE_KINDS.
    """
    cat = _load_catalog()
    rng = random.Random(_seed_int(task_set_version, seed, tier, idx, kind_or_mistake))
    values = _pick_fields(cat, rng)
    tid = f"pr-t{tier}-{idx}-s{seed}"

    if not mistake:
        # Clean, correctly-labelled, complete power analysis.
        kind = kind_or_mistake
        text = _clean_excerpt(cat, kind, values, rng)
        gold = _make_gold(True, kind, values, FIELDS, [])
    elif kind_or_mistake == "no_power_analysis":
        text = _build_no_power(cat, values, rng)
        gold = _make_gold(False, None, values, [], ["no_power_analysis"])
    elif kind_or_mistake == "posthoc_as_apriori":
        # A post-hoc analysis dressed up as a-priori. TRUE kind is posthoc.
        template = rng.choice(cat["posthoc_as_apriori_templates"])
        text = _render(template, values)
        gold = _make_gold(True, "posthoc", values, FIELDS, ["posthoc_as_apriori"])
    elif kind_or_mistake == "sensitivity_as_apriori":
        # A sensitivity analysis (smallest detectable effect at fixed N) worded as
        # a-priori. TRUE kind is sensitivity. All fields present, just mislabelled.
        template = rng.choice(cat["sensitivity_as_apriori_templates"])
        text = _render(template, values)
        gold = _make_gold(True, "sensitivity", values, FIELDS, ["sensitivity_as_apriori"])
    elif kind_or_mistake == "missing_fields":
        # A genuine (apriori/sensitivity/posthoc) analysis missing some fields.
        base_kind = CLEAN_KINDS[idx % len(CLEAN_KINDS)]
        drop = _MISSING_SUBSETS[idx % len(_MISSING_SUBSETS)]
        present = [f for f in FIELDS if f not in drop]
        # Render from the clean template, then surgically strip dropped fields by
        # blanking their concrete value tokens (so the text no longer states them).
        full = _clean_excerpt(cat, base_kind, values, rng)
        text = full
        for f in drop:
            text = text.replace(values[f], "[not reported]")
        gold = _make_gold(True, base_kind, values, present, ["missing_fields"])
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown mistake kind {kind_or_mistake!r}")

    envelope = {
        "task_id": tid,
        "arena_id": "power-reporting-v1",
        "task_set_version": "v1",
        "split": split,
        "visibility": visibility,
        "difficulty": {"tier": tier, "n_fields": len(gold["fields"])},
        "input": {"text": text},
    }
    _GROUND_TRUTH_CACHE[tid] = gold
    return envelope


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    visibility = "public" if split == "revealed" else "held_out"

    def clean(tier, idx, kind):
        return _task(task_set_version, seed, tier, idx, kind,
                     mistake=False, split=split, visibility=visibility)

    def injected(tier, idx, mk):
        return _task(task_set_version, seed, tier, idx, mk,
                     mistake=True, split=split, visibility=visibility)

    # T1 (clean/simple): complete, clearly-labelled analyses — one per clean kind.
    for i, kind in enumerate(CLEAN_KINDS):
        yield clean(1, i, kind)

    # T2 (false-alarm trap): CLEAN excerpts that look suspicious but a good
    #   player must NOT mis-handle:
    #     - a methods excerpt that mentions sample size / alpha but reports NO
    #       power analysis (must not be flagged as having one);
    #     - a correctly-labelled COMPLETE post-hoc analysis (must not be
    #       mislabelled as a-priori — the inverse of the headline trap).
    yield injected(2, 0, "no_power_analysis")   # clean wrt extraction (no PA)
    yield clean(2, 1, "posthoc")                # correctly-labelled posthoc
    yield clean(2, 2, "sensitivity")            # correctly-labelled sensitivity

    # T3 (single injected mistake): one mistake kind, cycling through ALL kinds.
    for i, mk in enumerate(MISTAKE_KINDS):
        yield injected(3, i, mk)

    # T4 (subtle): the headline trap (posthoc_as_apriori) plus subtle
    #   missing_fields — the borderline discriminations.
    yield injected(4, 0, "posthoc_as_apriori")
    yield injected(4, 1, "missing_fields")
    yield injected(4, 2, "posthoc_as_apriori")

    # T5 (multiple): each mistake kind once more, exercised at higher index so the
    #   missing-field subset and wording differ from T3/T4.
    for i, mk in enumerate(MISTAKE_KINDS):
        yield injected(5, i + 3, mk)

    # T6 (full composition): one clean control + the full array of mistakes, so a
    #   single tier touches every kind (clean + all three injected).
    yield clean(6, 0, "apriori")
    yield injected(6, 1, "posthoc_as_apriori")
    yield injected(6, 2, "missing_fields")
    yield injected(6, 3, "no_power_analysis")


def ground_truth(task_id: str) -> dict:
    """Return gold for a task. Regenerated from seed via the in-process cache.

    The runner always calls generate() before ground_truth(); this arena needs
    no external gold registry because the secret is the private seed, not a
    stored answer key. Raises KeyError if the task is unknown.
    """
    if task_id not in _GROUND_TRUTH_CACHE:
        raise KeyError(
            f"No cached gold for {task_id!r}; call generate() for the matching "
            "split/seed before ground_truth()."
        )
    return _GROUND_TRUTH_CACHE[task_id]

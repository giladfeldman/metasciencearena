"""Scorer for code-translation-r-v1 — executable equivalence.

The player's R is EXECUTED against the same fixed dataset the gold was built
from, and the statistics it prints are compared to gold by name. Textual
similarity to a reference translation is deliberately not part of the score: the
central failure mode this arena measures is code that runs cleanly, reads
plausibly, and computes the wrong quantity.

    composite = execution_rate x statistic_accuracy

`execution_rate` is 0/1 for a single task (did it run and emit parseable JSON?);
it becomes a genuine rate when averaged across a task set. Multiplying rather
than averaging the two channels means unrunnable code scores 0 outright — a
translator that emits confident nonsense cannot bank partial credit for it.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register first: @dataclass needs it
    spec.loader.exec_module(mod)
    return mod


_r_runner = _load("_xlat_r_runner", ARENA_DIR / "r_runner.py")
_generator = _load("_xlat_generator", ARENA_DIR / "generator.py")

# Relative tolerance for comparing a statistic to gold. Loose enough to absorb
# floating-point and print-precision noise, far tighter than the gap any of the
# arena's traps produce (the smallest, Type I vs Type III F, is ~2%).
REL_TOL = 1e-6
ABS_TOL = 1e-9

# Statistics that are COUNTS or DEGREES OF FREEDOM must match exactly. A relative
# tolerance is wrong for them: at n = 1,000,000 a rel_tol of 1e-6 accepts an
# off-by-one, and "the translation analysed one more case than SPSS did" is
# precisely the listwise/pairwise deletion error this arena exists to catch.
_EXACT_SUFFIXES = ("_n", "_df", "df")
_EXACT_PREFIXES = ("n_", "df_")


def is_exact_statistic(name: str) -> bool:
    """True when `name` is a count/df that must match exactly, not within tolerance."""
    n = name.lower()
    return (n in {"n", "df"}
            or n.startswith(_EXACT_PREFIXES)
            or n.endswith(_EXACT_SUFFIXES))


def _close(a, b, *, exact: bool = False) -> bool:
    """Compare one reported statistic to gold.

    Strict about types on purpose. An earlier version coerced through ``bool()``
    whenever either side was a bool, so ``_close(2, True)`` and ``_close("x", True)``
    both returned True — a player emitting a non-numeric value could be credited
    for a statistic it never computed. Booleans now only compare against booleans.

    A non-finite player value (NaN/Inf, typically an all-missing computation)
    never matches, even against a non-finite gold: silently "agreeing" on NaN
    would reward a translation that computed nothing.

    `exact=True` demands equality (counts, df) rather than tolerance.
    """
    a_is_bool, b_is_bool = isinstance(a, bool), isinstance(b, bool)
    if a_is_bool or b_is_bool:
        # Only bool-vs-bool can match; bool-vs-anything-else is a type mismatch.
        return a_is_bool and b_is_bool and a is b
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return False
    if exact:
        return fa == fb
    return math.isclose(fa, fb, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def score(player_output: dict, ground_truth: dict) -> dict:
    """Execute the player's R and grade it against executed gold."""
    required: list[str] = list(ground_truth.get("required_statistics") or [])
    gold: dict = ground_truth.get("gold_statistics") or {}
    findings: list[dict] = []

    if not gold:
        # No gold means the arena cannot judge this task. Report it honestly as
        # an error rather than scoring 0, which would look like a player failure.
        return {
            "primary": 0.0,
            "breakdown": {"error": "gold_not_built"},
            "findings": [{
                "category": "does_not_execute",
                "detail": "No executed gold for this task — run tools/build_gold.py.",
            }],
        }

    r_code = (player_output or {}).get("r_code")
    if not isinstance(r_code, str) or not r_code.strip():
        # An empty translation is a real, reportable outcome — a converter that
        # cannot handle a construct emits nothing. Say so precisely rather than
        # implying the harness failed to collect an answer.
        return _fail(required, gold, "does_not_execute",
                     "The translator produced no R code for this script "
                     "(unsupported construct, or the converter returned empty).")

    data_csv = _generator.dataset_csv(ground_truth["dataset"])
    res = _r_runner.run_r(r_code, data_csv, required=required)

    if not res.ok:
        detail = f"R execution failed: {res.error}"
        if res.stderr.strip():
            detail += f" | stderr: {res.stderr.strip()[:300]}"
        return _fail(required, gold, "does_not_execute", detail)

    matched, wrong, missing, no_gold = [], [], [], []
    for key in required:
        if key not in res.stats:
            missing.append(key)
        elif key not in gold:
            # Gold is incomplete for a declared statistic. That is an ARENA
            # defect, not a player failure — never silently charge it to the
            # player as a wrong answer (which `gold.get(key)` -> None would).
            no_gold.append(key)
        elif _close(res.stats[key], gold[key], exact=is_exact_statistic(key)):
            matched.append(key)
        else:
            wrong.append(key)

    for key in missing:
        findings.append({
            "category": "missing_statistic",
            "detail": f"The R ran but never emitted '{key}'.",
        })
    for key in no_gold:
        findings.append({
            "category": "missing_statistic",
            "detail": (f"Gold has no value for '{key}', so the arena cannot judge it — "
                       "an arena defect, not a player failure. Re-run tools/build_gold.py."),
        })
    for key in wrong:
        findings.append({
            "category": "wrong_statistic",
            "detail": (f"'{key}' = {res.stats[key]!r}, gold = {gold.get(key)!r}. "
                       "The code ran but computed a different quantity."),
        })

    # Statistics with no gold are un-judgeable, so they leave the denominator
    # entirely rather than counting against the player for an arena defect.
    n_judgeable = len(required) - len(no_gold)
    statistic_accuracy = (len(matched) / n_judgeable) if n_judgeable > 0 else 0.0
    execution_rate = 1.0
    composite = execution_rate * statistic_accuracy

    return {
        "primary": composite,
        "breakdown": {
            "execution_rate": execution_rate,
            "statistic_accuracy": statistic_accuracy,
            "composite": composite,
            "n_statistics_required": len(required),
            "n_statistics_judgeable": n_judgeable,
            "n_statistics_matched": len(matched),
            "n_statistics_wrong": len(wrong),
            "n_statistics_missing": len(missing),
            "n_statistics_no_gold": len(no_gold),
        },
        "findings": findings,
    }


def _fail(required: list[str], gold: dict, category: str, detail: str) -> dict:
    """Score for a translation that never produced comparable numbers."""
    return {
        "primary": 0.0,
        "breakdown": {
            "execution_rate": 0.0,
            "statistic_accuracy": 0.0,
            "composite": 0.0,
            "n_statistics_required": len(required),
            "n_statistics_matched": 0,
            "n_statistics_wrong": 0,
            "n_statistics_missing": len(required),
        },
        "findings": [{"category": category, "detail": detail}],
    }

"""Scorer for publication-bias-v1.

Single verdict per task: the player emits one `bias_detected` boolean plus a
`confidence` (and optionally `egger_p` / `n_missing_trimfill`). The scorer compares
that boolean to the computed gold label and blends in a light calibration term on the
confidence so a correct-but-unconfident (or wrong-but-overconfident) player is nudged.

primary = 0.85 * correct + 0.15 * calibration   (in [0,1])
  correct      = 1.0 if bias_detected == gold else 0.0
  calibration  = 1 - |confidence - correct|     (rewards confidence aligned with
                 correctness: high confidence when right, low when wrong)

A perfectly-correct, fully-confident player scores 1.0; a deterministic tool that is
always right with confidence 1.0 therefore scores 1.0 (the cross-validation oracle).

Findings categories (must match arena.yaml#error_categories):
  - bias_missed      (major)  gold biased, player said unbiased (false negative)
  - bias_false_alarm (major)  gold unbiased, player flagged bias (false positive)
"""
from __future__ import annotations


#: Names the scoring FORMULA this module implements, stamped into every record's
#: `instrument` block by framework/runner.py::_build_instrument. Two records with a
#: different formula_id are not comparable and must never be averaged into one
#: leaderboard row. Bump the suffix when the FORMULA changes; never when the code
#: merely moves.
#:
#: primary = 0.85 * correct + 0.15 * calibration, on a single bias_detected
#: boolean. calibration = 1 - |confidence - correct|.
FORMULA_ID = "publication-bias-ACC85-CAL15-v1"

CORRECT_WEIGHT = 0.85
CALIB_WEIGHT = 0.15


# --------------------------------------------------------------------------- #
# Task set v2 — the STATISTICS, not a verdict (owner decision 2026-09-17).
#
# Dispatch is on the SHAPE of the gold: v2 gold carries `gold_stats`, v1 gold
# carries `bias_detected`. A gold carrying neither is refused (primary 0.0 with a
# named error), never scored as if it were the other version -- the failure
# pdf-text-fidelity-v1 once had, where a mixed-shape gold published a wrong NUMBER.
#
# primary = WEIGHTED mean over the six METHODS (V2_METHOD_WEIGHTS) of the mean over
# that method's statistics.
# Each statistic scores in [0, 1] by its own tolerance band (V2_STATISTICS):
# full credit within `full` of gold, falling linearly to 0 at `zero`. An omitted
# statistic scores 0 -- exactly like a wrong one -- because v1 taught that when
# reporting is optional, reporting is not computing: of 115 emitted `egger_p` values
# only 45 were within 0.01 of the correct Egger p (33 of 103 once the reference tool
# is excluded; re-measured 2026-09-30 against scipy, since v1's own gold p was off on
# one task -- see CHANGELOG.md#v2).
#
# METHOD WEIGHTS -- owner decision (Gilad, 2026-10-01: "more weight on the latest
# over the mostly outdated old ones"), ordered by his ranking of 2026-09-17: RoBMA the
# most comprehensive, PET-PEESE and the selection model current, Egger among the
# worst, trim-and-fill worse, Begg included for coverage only. Modern methods carry
# 0.75 of the score, the classic ones 0.25. The exact split within each group is this
# session's reading of that ranking, not a separate owner decision.
# --------------------------------------------------------------------------- #

#: Formula id stamped on v2 records (runner._resolve_formula_id reads
#: formula_id_for first). Bump the suffix if a band, a statistic or a weight changes.
#: The name carries NO weight digits on purpose: two digits after a capital letter
#: claim a coefficient that reaches `primary` (framework/tests/test_formula_id.py),
#: and the modern methods' 0.75 is a SUM of three weights, not a coefficient. The
#: first name, "...MODERN75-v2", failed that gate at the 2026-10-01 /ship QA.
FORMULA_ID_V2 = "publication-bias-STATS6-MODERNW-v2"

#: The closed set of names this module can emit (`formula_id_for` may return only
#: these). framework/tests/test_formula_id.py reads it; without it a scorer that
#: dispatches on the gold declares no formula at all.
FORMULA_IDS = (FORMULA_ID, FORMULA_ID_V2)

# (method, statistic, kind, full, zero)
#   kind "abs"   : error = |reported - gold|
#   kind "logbf" : error = |ln(reported) - ln(gold)|  (Bayes factors span decades)
#   kind "count" : exact = 1, else 0
#
# The prompt names each method and its R function exactly, so a correct
# implementation reproduces gold to ~1e-5 (MEASURED: metafor vs an independent numpy
# Egger agree to 6 significant digits on 12/12 v1 sets; metafor::selmodel vs weightr
# agree to within 0.0004 on 10/10 v2 sets where both run; metafor vs meta trim-and-fill
# agree on side, k0 and estimate on 12/12). The bands are there for ROUNDING, not for
# approximation: effect estimates full credit within 0.002, zero at 0.02; p-values and
# Kendall's tau within 0.005, zero at 0.05; the trim-and-fill count exact.
#
# SIZED AGAINST A PLAYER THAT COMPUTES NOTHING (Sonnet consult, 2026-09-30): report
# the inverse-variance pooled mean for every adjusted estimate, p = 0.5, tau = 0 and
# k0 = 0. On the 12 revealed sets (RoBMA excluded) that player scored 0.393 under the
# first bands (0.01/0.10 everywhere, k0 off-by-one = 0.5) and 0.203 under these; a
# gold-reporting player scores 1.000 under both. The residue is real partial
# correctness -- when trim-and-fill imputes nothing, the plain random-effects mean IS
# its answer.
#
# RoBMA is MCMC, so its gold is one draw. Its bands are sized from the MEASURED
# seed-to-seed spread of the gold statistics (tools/build_gold_v2.py
# --stability-seeds). MEASURED 2026-09-30, gold (seed 20260930) vs seed 2 on all 12
# revealed sets: max |d ln BF| = 0.013, max |d mu| = 0.0029. Full credit is set at
# ~4x that noise (0.05 on ln BF, 0.01 on mu); zero at 10x the full band for ln BF and
# 5x for mu. See test_robma_full_credit_band_covers_measured_mcmc_noise, which fails if the
# full-credit band is narrower than the largest spread observed.
V2_STATISTICS = [
    ("egger",     "p",        "abs",   0.005, 0.05),
    ("begg",      "tau",      "abs",   0.005, 0.05),
    ("begg",      "p",        "abs",   0.005, 0.05),
    ("trimfill",  "k0",       "count", None,  None),
    ("trimfill",  "estimate", "abs",   0.002, 0.02),
    ("pet",       "estimate", "abs",   0.002, 0.02),
    ("peese",     "estimate", "abs",   0.002, 0.02),
    ("three_psm", "estimate", "abs",   0.002, 0.02),
    ("three_psm", "lrt_p",    "abs",   0.005, 0.05),
    ("robma",     "bf_bias",  "logbf", 0.05, 0.50),
    ("robma",     "mu",       "abs",   0.01, 0.05),
]

#: The six methods. PET and PEESE are one method (PET-PEESE).
V2_METHODS = {
    "egger": ["egger"],
    "begg": ["begg"],
    "trimfill": ["trimfill"],
    "pet_peese": ["pet", "peese"],
    "three_psm": ["three_psm"],
    "robma": ["robma"],
}


#: Weight of each method in `primary`; sums to 1 (asserted below and in the tests).
V2_METHOD_WEIGHTS = {
    "robma": 0.30,
    "pet_peese": 0.225,
    "three_psm": 0.225,
    "egger": 0.10,
    "begg": 0.075,
    "trimfill": 0.075,
}
assert abs(sum(V2_METHOD_WEIGHTS.values()) - 1.0) < 1e-12
assert set(V2_METHOD_WEIGHTS) == set(V2_METHODS)


def formula_id_for(ground_truth) -> str | None:
    """v2 records name the v2 formula; anything else falls through to FORMULA_ID."""
    if isinstance(ground_truth, dict) and "gold_stats" in ground_truth:
        return FORMULA_ID_V2
    return None


def _band(err: float, full: float, zero: float) -> float:
    if err <= full:
        return 1.0
    if err >= zero:
        return 0.0
    return 1.0 - (err - full) / (zero - full)


def _as_number(v):
    # bool is an int subclass; a `true` is not a statistic.
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _score_one(kind, full, zero, reported, gold):
    import math
    if reported is None:
        return 0.0, None
    if kind == "count":
        if reported != int(reported):
            return 0.0, None
        diff = abs(int(reported) - int(gold))
        return (1.0 if diff == 0 else 0.0), float(diff)
    if kind == "logbf":
        if reported <= 0:
            return 0.0, None
        err = abs(math.log(reported) - math.log(gold))
    else:
        err = abs(reported - gold)
    return _band(err, full, zero), err


def _score_v2(player_output: dict, ground_truth: dict) -> dict:
    gold_stats = ground_truth["gold_stats"]
    per_stat: dict[str, float] = {}
    errors: dict[str, float] = {}
    findings: list[dict] = []
    n_reported = 0
    for method, stat, kind, full, zero in V2_STATISTICS:
        key = f"{method}.{stat}"
        gold = float(gold_stats[method][stat])
        block = player_output.get(method)
        raw = block.get(stat) if isinstance(block, dict) else None
        reported = _as_number(raw)
        if reported is not None:
            n_reported += 1
        s, err = _score_one(kind, full, zero, reported, gold)
        per_stat[key] = s
        if err is not None:
            errors[key] = err
        if reported is None:
            findings.append({
                "category": "statistic_missing",
                "anchor": {"statistic": key},
                "correct_value": gold,
                "evidence": "not reported",
            })
        elif s < 1.0:
            findings.append({
                "category": "statistic_off",
                "anchor": {"statistic": key},
                "correct_value": gold,
                "evidence": reported,
                "detail": f"{key} = {reported:.6g}, gold = {gold:.6g} (credit {s:.2f}).",
            })

    per_method = {}
    for m, blocks in V2_METHODS.items():
        vals = [v for k, v in per_stat.items() if k.split(".")[0] in blocks]
        per_method[m] = sum(vals) / len(vals)
    primary = sum(V2_METHOD_WEIGHTS[m] * v for m, v in per_method.items())

    breakdown = {
        "primary": primary,
        "n_statistics": len(V2_STATISTICS),
        "n_reported": n_reported,
        **{f"method_{m}": v for m, v in per_method.items()},
        **{f"stat_{k.replace('.', '_')}": v for k, v in per_stat.items()},
        **{f"abs_err_{k.replace('.', '_')}": v for k, v in errors.items()},
    }
    return {"primary": primary, "breakdown": breakdown, "findings": findings}


def score(player_output: dict, ground_truth: dict) -> dict:
    if isinstance(ground_truth, dict) and "gold_stats" in ground_truth:
        return _score_v2(player_output, ground_truth)
    if not isinstance(ground_truth, dict) or "bias_detected" not in ground_truth:
        return {"primary": 0.0,
                "breakdown": {"error": "gold carries neither gold_stats (v2) nor bias_detected (v1)"},
                "findings": []}
    gold = bool(ground_truth.get("bias_detected"))
    predicted = bool(player_output.get("bias_detected"))

    conf = player_output.get("confidence", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(max(conf, 0.0), 1.0)

    correct = 1.0 if predicted == gold else 0.0
    calibration = 1.0 - abs(conf - correct)
    primary = CORRECT_WEIGHT * correct + CALIB_WEIGHT * calibration

    findings: list[dict] = []
    if gold and not predicted:
        findings.append({
            "category": "bias_missed",
            "anchor": {"task": "verdict"},
            "correct_value": "bias_detected=true",
            "evidence": "predicted bias_detected=false",
        })
    elif predicted and not gold:
        findings.append({
            "category": "bias_false_alarm",
            "anchor": {"task": "verdict"},
            "correct_value": "bias_detected=false",
            "evidence": "predicted bias_detected=true",
        })

    breakdown = {
        "correct": correct,
        "calibration": calibration,
        "predicted": predicted,
        "gold": gold,
        "confidence": conf,
        "primary": primary,
    }

    # If the player reported its own Egger p-value, record the absolute agreement with
    # the gold (computed) Egger p as a diagnostic (does not affect primary).
    gold_p = ground_truth.get("egger_p")
    pred_p = player_output.get("egger_p")
    if isinstance(gold_p, (int, float)) and isinstance(pred_p, (int, float)):
        breakdown["egger_p_abs_err"] = abs(float(pred_p) - float(gold_p))

    return {
        "primary": primary,
        "breakdown": breakdown,
        "findings": findings,
    }

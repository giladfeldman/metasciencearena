"""Scorer for effect-size-conversion-v1.

The player emits one converted value per task plus a `confidence`. The scorer compares
the converted value to the computed gold with a metric-scaled tolerance and blends in a
light calibration term on the confidence.

primary = 0.85 * agreement + 0.15 * calibration   (in [0,1])

  agreement   = 1.0 if abs(converted - gold) <= tol             (within tolerance)
                linearly decaying to 0.0 across a wider band [tol, tol + decay_band]
                0.0 beyond the band
  calibration = 1 - |confidence - within_tol|   (rewards confidence aligned with being
                inside the tolerance band)

Tolerance is scaled to the TARGET metric so a fixed absolute tol is fair across metrics
whose natural scales differ wildly (eta2 in [0,1] vs OR which can be many units): we use
the larger of an absolute floor and a relative fraction of |gold|. A deterministic tool
that reproduces the canonical formula is always within tol with confidence 1.0 => scores
1.0 (the cross-validation oracle).

Findings categories (must match arena.yaml#error_categories):
  - conversion_error  (major)  the converted value is outside tolerance of the gold.
"""
from __future__ import annotations

CORRECT_WEIGHT = 0.85
CALIB_WEIGHT = 0.15

# Absolute tolerance floor and relative fraction. The effective tolerance is
# max(ABS_TOL, REL_TOL * |gold|): an absolute floor protects tiny-magnitude targets
# (e.g. eta2 ~ 0.01), while the relative band keeps large targets (e.g. OR ~ 10) fair.
ABS_TOL = 0.01
REL_TOL = 0.01
# Beyond `tol`, agreement decays linearly to 0 across DECAY_MULT * tol of further error.
DECAY_MULT = 10.0


def _tolerance(gold: float) -> float:
    return max(ABS_TOL, REL_TOL * abs(gold))


def score(player_output: dict, ground_truth: dict) -> dict:
    gold = ground_truth.get("converted")
    try:
        gold = float(gold)
    except (TypeError, ValueError):
        gold = 0.0

    raw = player_output.get("converted")
    has_pred = isinstance(raw, (int, float))
    try:
        predicted = float(raw)
    except (TypeError, ValueError):
        predicted = float("nan")
        has_pred = False

    conf = player_output.get("confidence", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(max(conf, 0.0), 1.0)

    tol = _tolerance(gold)
    if has_pred:
        abs_error = abs(predicted - gold)
    else:
        abs_error = float("inf")

    within_tol = abs_error <= tol
    if within_tol:
        agreement = 1.0
    else:
        band = DECAY_MULT * tol
        over = abs_error - tol
        agreement = max(0.0, 1.0 - over / band)

    calibration = 1.0 - abs(conf - (1.0 if within_tol else 0.0))
    primary = CORRECT_WEIGHT * agreement + CALIB_WEIGHT * calibration

    findings: list[dict] = []
    if not within_tol:
        findings.append({
            "category": "conversion_error",
            "anchor": {
                "from": ground_truth.get("from"),
                "to": ground_truth.get("to"),
                "value": ground_truth.get("value"),
            },
            "correct_value": round(gold, 6),
            "evidence": (round(predicted, 6) if has_pred else "missing"),
        })

    breakdown = {
        "agreement": agreement,
        "within_tol": within_tol,
        "abs_error": (round(abs_error, 8) if has_pred else None),
        "tolerance": round(tol, 8),
        "gold": round(gold, 6),
        "predicted": (round(predicted, 6) if has_pred else None),
        "from": ground_truth.get("from"),
        "to": ground_truth.get("to"),
        "calibration": calibration,
        "confidence": conf,
        "primary": primary,
    }

    return {
        "primary": primary,
        "breakdown": breakdown,
        "findings": findings,
    }

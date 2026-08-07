"""Scorer for p-curve-v1.

Single verdict per task: the player emits one `evidential_value` boolean plus a
`confidence` (and optionally `right_skew_p` / `flatness_p`). The scorer compares that
boolean to the computed gold label and blends in a light calibration term on the
confidence so a correct-but-unconfident (or wrong-but-overconfident) player is nudged.

primary = 0.85 * correct + 0.15 * calibration   (in [0,1])
  correct      = 1.0 if evidential_value == gold else 0.0
  calibration  = 1 - |confidence - correct|     (rewards confidence aligned with
                 correctness: high confidence when right, low when wrong)

A perfectly-correct, fully-confident player scores 1.0; a deterministic tool that is
always right with confidence 1.0 therefore scores 1.0 (the cross-validation oracle).

Findings categories (must match arena.yaml#error_categories):
  - evidential_missed      (major)  gold evidential, player said no-evidential (false negative)
  - evidential_false_alarm (major)  gold no-evidential, player claimed evidential (false positive)
"""
from __future__ import annotations

CORRECT_WEIGHT = 0.85
CALIB_WEIGHT = 0.15

# Tolerance for the optional right_skew_p agreement diagnostic.
RIGHT_SKEW_P_TOL = 0.05


def score(player_output: dict, ground_truth: dict) -> dict:
    gold = bool(ground_truth.get("evidential_value"))
    predicted = bool(player_output.get("evidential_value"))

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
            "category": "evidential_missed",
            "anchor": {"task": "verdict"},
            "correct_value": "evidential_value=true",
            "evidence": "predicted evidential_value=false",
        })
    elif predicted and not gold:
        findings.append({
            "category": "evidential_false_alarm",
            "anchor": {"task": "verdict"},
            "correct_value": "evidential_value=false",
            "evidence": "predicted evidential_value=true",
        })

    breakdown = {
        "correct": correct,
        "calibration": calibration,
        "predicted": predicted,
        "gold": gold,
        "confidence": conf,
        "primary": primary,
    }

    # If the player reported its own right-skew p-value, record the absolute agreement
    # with the gold (computed) right_skew_p and whether it lands within tolerance
    # (diagnostic only — does not affect primary).
    gold_p = ground_truth.get("right_skew_p")
    pred_p = player_output.get("right_skew_p")
    if isinstance(gold_p, (int, float)) and isinstance(pred_p, (int, float)):
        abs_err = abs(float(pred_p) - float(gold_p))
        breakdown["right_skew_p_abs_err"] = abs_err
        breakdown["right_skew_p_agree"] = abs_err <= RIGHT_SKEW_P_TOL

    return {
        "primary": primary,
        "breakdown": breakdown,
        "findings": findings,
    }

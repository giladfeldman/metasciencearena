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

CORRECT_WEIGHT = 0.85
CALIB_WEIGHT = 0.15


def score(player_output: dict, ground_truth: dict) -> dict:
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

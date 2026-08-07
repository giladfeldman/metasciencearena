"""Scorer for zcurve-evidential-v1 (single-verdict arena).

Compares a player's evidential-value verdict to gold:
  - primary       = 1.0 if has_evidential_value matches gold, else 0.0
  - calibration   = 1 - |confidence - correct| (a simple per-item calibration:
                    1.0 for a confident-correct or unconfident-wrong call, lower
                    when the player is confidently wrong)

A wrong verdict emits exactly one finding whose category matches
arena.yaml#error_categories:
  - misclassified_evidential     (major)  said NON-evidential but gold is evidential
                                          (a real-effect set dismissed as p-hacked)
  - misclassified_non_evidential (major)  said evidential but gold is NON-evidential
                                          (a selection-over-nulls set credited with
                                          evidential value)
"""
from __future__ import annotations


def score(player_output: dict, ground_truth: dict) -> dict:
    gold_value = bool(ground_truth.get("has_evidential_value", False))
    pred_value = bool(player_output.get("has_evidential_value", False))

    conf = player_output.get("confidence", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(max(conf, 0.0), 1.0)

    correct = pred_value == gold_value
    primary = 1.0 if correct else 0.0
    # Simple per-item calibration: 1 - |confidence - correctness|.
    calibration = 1.0 - abs(conf - (1.0 if correct else 0.0))

    findings: list[dict] = []
    if not correct:
        if gold_value and not pred_value:
            # Gold is evidential; the player said non-evidential.
            category = "misclassified_evidential"
        else:
            # Gold is non-evidential; the player said evidential.
            category = "misclassified_non_evidential"
        findings.append({
            "category": category,
            "anchor": {"field": "has_evidential_value"},
            "evidence": str(pred_value),
            "correct_value": gold_value,
        })

    return {
        "primary": primary,
        "breakdown": {
            "correct": 1 if correct else 0,
            "predicted": pred_value,
            "gold": gold_value,
            "confidence": conf,
            "calibration": calibration,
        },
        "findings": findings,
    }

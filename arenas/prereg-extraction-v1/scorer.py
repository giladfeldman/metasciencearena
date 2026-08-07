"""Scorer for prereg-extraction-v1 (field-map arena).

Given the player's structured output and the gold field-map, computes:
  - detection      : 1.0 if prereg_found matches gold, else 0.0
  - platform_acc   : platform-label accuracy (only when a prereg is truly present
                     AND the player agrees one is present; 1.0 by convention when
                     gold has no prereg and the player correctly found none)
  - field_f1       : token-overlap F1 of the four extracted fields against gold
                     (1.0 by convention when gold has no fields to extract and the
                     player correctly extracted none)
  - calibration    : 1 - ECE over the player's confidence in the detection call
  - composite      : detection * platform_acc * field_f1 * calibration
                     (multiplicative — weakness anywhere pulls it down)

The T2 decoy tier is the false-alarm trap: the text discusses preregistration but
embeds none, so a clean-correct answer is prereg_found=False. Flagging it shows up
as prereg_false_alarm and zeroes detection.

Findings categories (must match arena.yaml#error_categories):
  prereg_missed (major), prereg_false_alarm (major),
  platform_mislabel (minor), field_wrong (minor).
"""
from __future__ import annotations

import re

FIELDS = ["hypotheses", "design", "sample_size", "analysis_plan"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _ece(predictions: list[tuple[float, bool]], n_bins: int = 5) -> float:
    """Expected Calibration Error. predictions = [(confidence, is_correct), ...]."""
    if not predictions:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in predictions:
        conf = min(max(conf, 0.0), 1.0)
        idx = min(n_bins - 1, int(conf * n_bins))
        bins[idx].append((conf, correct))
    total = len(predictions)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(1 for _, ok in b if ok) / len(b)
        ece += (len(b) / total) * abs(avg_conf - avg_acc)
    return ece


def _tokens(text) -> set[str]:
    if not isinstance(text, str):
        return set()
    return set(_TOKEN_RE.findall(text.lower()))


def _field_token_f1(pred, gold) -> float:
    """Token-overlap F1 between one predicted and gold field value."""
    g = _tokens(gold)
    p = _tokens(pred)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    overlap = len(g & p)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def score(player_output: dict, ground_truth: dict) -> dict:
    gold_found = bool(ground_truth.get("prereg_found"))
    gold_platform = ground_truth.get("platform")
    gold_fields = ground_truth.get("fields") or {}

    pred_found = bool(player_output.get("prereg_found"))
    pred_platform = player_output.get("platform")
    pred_fields = player_output.get("fields") or {}

    conf = player_output.get("confidence", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0

    findings: list[dict] = []

    # --- detection ---
    detection = 1.0 if pred_found == gold_found else 0.0
    if gold_found and not pred_found:
        findings.append({
            "category": "prereg_missed",
            "anchor": {"link": ground_truth.get("link")},
            "correct_value": ground_truth.get("link"),
        })
    elif (not gold_found) and pred_found:
        findings.append({
            "category": "prereg_false_alarm",
            "anchor": {"platform": pred_platform},
            "evidence": str(pred_platform),
        })

    # --- platform accuracy ---
    if not gold_found:
        # No prereg to label. Correct detection => trivially fine on platform.
        platform_acc = 1.0
    elif not pred_found:
        # Player missed the prereg entirely; platform is unscorable -> 0.
        platform_acc = 0.0
    else:
        platform_acc = 1.0 if pred_platform == gold_platform else 0.0
        if pred_platform != gold_platform:
            findings.append({
                "category": "platform_mislabel",
                "anchor": {"link": ground_truth.get("link")},
                "evidence": str(pred_platform),
                "correct_value": gold_platform,
            })

    # --- field-extraction F1 ---
    if not gold_found:
        # No fields to extract; player correct iff it extracted none of substance.
        any_pred = any(_tokens(pred_fields.get(f)) for f in FIELDS)
        field_f1 = 0.0 if any_pred else 1.0
    elif not pred_found:
        field_f1 = 0.0
    else:
        per_field = []
        for f in FIELDS:
            f1 = _field_token_f1(pred_fields.get(f), gold_fields.get(f))
            per_field.append(f1)
            if f1 < 0.5:
                findings.append({
                    "category": "field_wrong",
                    "anchor": {"field": f},
                    "evidence": str(pred_fields.get(f)),
                    "correct_value": gold_fields.get(f),
                })
        field_f1 = sum(per_field) / len(per_field) if per_field else 1.0

    # --- calibration over the detection call ---
    calibration = 1.0 - _ece([(conf, detection == 1.0)])

    composite = detection * platform_acc * field_f1 * calibration

    return {
        "primary": composite,
        "breakdown": {
            "detection": detection,
            "platform_acc": platform_acc,
            "field_f1": field_f1,
            "calibration": calibration,
            "composite": composite,
            "gold_found": gold_found,
            "pred_found": pred_found,
        },
        "findings": findings,
    }

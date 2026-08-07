"""Scorer for reference-integrity-v1.

Matches the player's per-reference integrity judgements to gold by reference_id
and computes:
  - detection precision/recall/F1 on flagged vs not-flagged
  - kind_accuracy = of correctly-flagged references, fraction with the right issue_kind
  - calibration   = 1 - ECE over the player's confidence
  - composite     = detection_f1 * kind_accuracy * calibration

The T2 controls tier is the key trap: references that look suspicious (reordered
initials, DOI suffixes, a cited reference that resembles an uncited one) are CLEAN.
Flagging them shows up as integrity_false_alarm and tanks precision.

Findings categories (must match arena.yaml#error_categories):
  - integrity_missed       (major)  a truly-flawed reference was not flagged
  - integrity_false_alarm  (major)  a clean reference was flagged
  - kind_mislabel          (minor)  a correctly-flagged reference got the wrong issue_kind
"""
from __future__ import annotations


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


def score(player_output: dict, ground_truth: dict) -> dict:
    gold = ground_truth.get("references", [])
    gold_by_ref = {g["reference_id"]: g for g in gold}
    preds = player_output.get("records", [])

    findings: list[dict] = []
    pred_by_ref: dict[str, dict] = {}
    for p in preds:
        ref = p.get("reference_id")
        if ref in gold_by_ref:
            pred_by_ref[ref] = p  # last write wins on duplicates
        else:
            # A record for a reference id the gold doesn't know. If the player
            # flagged it, that's an integrity_false_alarm on a phantom reference.
            if bool(p.get("flagged")):
                findings.append({
                    "category": "integrity_false_alarm",
                    "anchor": {"reference_id": ref},
                    "evidence": str(p.get("issue_kind")),
                })

    tp = fp = fn = 0
    kind_correct = 0
    kind_total = 0
    calib: list[tuple[float, bool]] = []

    for ref, g in gold_by_ref.items():
        truth = bool(g["flagged"])
        p = pred_by_ref.get(ref)
        if p is None:
            # No record at all for this reference. If it was truly flawed, that is
            # a miss; if clean, a missing record on a clean ref is harmless (TN).
            if truth:
                fn += 1
                findings.append({
                    "category": "integrity_missed",
                    "anchor": {"reference_id": ref},
                    "correct_value": g.get("issue_kind"),
                })
            continue

        pred_flag = bool(p.get("flagged"))
        conf = p.get("confidence", 0.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        calib.append((conf, pred_flag == truth))

        if truth and pred_flag:
            tp += 1
            kind_total += 1
            if p.get("issue_kind") == g.get("issue_kind"):
                kind_correct += 1
            else:
                findings.append({
                    "category": "kind_mislabel",
                    "anchor": {"reference_id": ref},
                    "evidence": str(p.get("issue_kind")),
                    "correct_value": g.get("issue_kind"),
                })
        elif truth and not pred_flag:
            fn += 1
            findings.append({
                "category": "integrity_missed",
                "anchor": {"reference_id": ref},
                "correct_value": g.get("issue_kind"),
            })
        elif (not truth) and pred_flag:
            fp += 1
            findings.append({
                "category": "integrity_false_alarm",
                "anchor": {"reference_id": ref},
                "evidence": str(p.get("issue_kind")),
            })
        # else: true negative — correctly left unflagged.

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    detection_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    kind_accuracy = kind_correct / kind_total if kind_total > 0 else 1.0
    calibration = 1.0 - _ece(calib) if calib else 1.0
    composite = detection_f1 * kind_accuracy * calibration

    return {
        "primary": composite,
        "breakdown": {
            "detection_f1": detection_f1,
            "precision": precision,
            "recall": recall,
            "kind_accuracy": kind_accuracy,
            "calibration": calibration,
            "composite": composite,
            "n_gold_references": len(gold_by_ref),
            "n_flawed": sum(1 for g in gold_by_ref.values() if g["flagged"]),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "findings": findings,
    }

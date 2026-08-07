"""Scorer for open-practices-repro-v1.

Matches the player's per-target reproducibility judgements to gold by `target`
(a file name, or the repo_url for a broken_link) and computes:
  - detection precision/recall/F1 over has-issue vs no-issue across all targets
  - kind_accuracy  = of correctly-flagged issues, the fraction with the right issue_kind
  - calibration    = 1 - ECE over the player's confidence
  - composite      = detection_f1 * kind_accuracy * calibration

The T2 false-alarm trap is the key challenge: files that MENTION absolute paths /
renames in comments but actually use relative paths to present files are CLEAN —
flagging them shows up as repro_issue_false_alarm and tanks precision.

error_categories (must match arena.yaml):
  repro_issue_missed       (major)  — a true defect was not flagged (false negative)
  repro_issue_false_alarm  (major)  — a clean target was flagged (false positive)
  kind_mislabel            (minor)  — a correctly-flagged defect got the wrong issue_kind
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


def _is_flagged(rec: dict) -> bool:
    """A player record asserts a defect when flagged is truthy.

    Tolerant: if `flagged` is absent, fall back to a non-null issue_kind.
    """
    if "flagged" in rec:
        return bool(rec.get("flagged"))
    return rec.get("issue_kind") not in (None, "", False)


def score(player_output: dict, ground_truth: dict) -> dict:
    gold = ground_truth.get("records", [])
    gold_by_target = {g["target"]: g for g in gold}
    preds = player_output.get("records", [])

    findings: list[dict] = []
    pred_by_target: dict[str, dict] = {}
    for p in preds:
        target = p.get("target")
        if target in gold_by_target:
            pred_by_target[target] = p  # last write wins on duplicates
        elif _is_flagged(p):
            # A flagged target that isn't a known candidate is a false alarm.
            findings.append({"category": "repro_issue_false_alarm",
                             "anchor": {"target": target},
                             "evidence": str(p.get("issue_kind"))})
        # an unflagged unknown target is harmless — ignored.

    tp = fp = fn = 0
    kind_correct = 0
    kind_total = 0
    calib: list[tuple[float, bool]] = []

    for target, g in gold_by_target.items():
        truth_kind = g.get("issue_kind")
        truth = truth_kind is not None
        p = pred_by_target.get(target)

        if p is None:
            # Player said nothing about this target == implicitly "no issue".
            if truth:
                fn += 1
                findings.append({"category": "repro_issue_missed",
                                 "anchor": {"target": target},
                                 "correct_value": truth_kind})
            continue

        pred_flag = _is_flagged(p)
        conf = p.get("confidence", 0.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        calib.append((conf, pred_flag == truth))

        if truth and pred_flag:
            tp += 1
            kind_total += 1
            if p.get("issue_kind") == truth_kind:
                kind_correct += 1
            else:
                findings.append({"category": "kind_mislabel",
                                 "anchor": {"target": target},
                                 "evidence": str(p.get("issue_kind")),
                                 "correct_value": truth_kind})
        elif truth and not pred_flag:
            fn += 1
            findings.append({"category": "repro_issue_missed",
                             "anchor": {"target": target},
                             "correct_value": truth_kind})
        elif (not truth) and pred_flag:
            fp += 1
            findings.append({"category": "repro_issue_false_alarm",
                             "anchor": {"target": target},
                             "evidence": str(p.get("issue_kind"))})
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
            "n_gold_targets": len(gold_by_target),
            "n_defective": sum(1 for g in gold_by_target.values() if g.get("issue_kind") is not None),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "findings": findings,
    }

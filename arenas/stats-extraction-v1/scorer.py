"""Scorer for stats-extraction-v1.

Matches player extractions to ground-truth items via span overlap (IoU >= 0.5)
plus kind agreement, using greedy maximum-overlap assignment. Computes:
  - precision  = TP / (TP + FP)
  - recall     = TP / (TP + FN)
  - calibration = 1 - ECE  (with deception items judged on flagged_suspicious)
  - composite   = precision * recall * calibration
"""
from __future__ import annotations


def _safe_conf(extraction: dict) -> float:
    """Read a player extraction's confidence, defaulting to 0.0.

    Players (especially LLM ones) frequently omit or malform `confidence`; a bare
    `float(extraction["confidence"])` would raise and the runner would record the
    whole task as an adapter error instead of degrading gracefully. Mirrors the
    grim scorer's defensive read (DR-0004).
    """
    try:
        return float(extraction["confidence"])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _match_score(gt_span: dict, ext_span: dict) -> float:
    """How well does an extracted span localize a GT anchor?

    GT spans in this arena are short value anchors (e.g. "5.34"); player
    extractions often quote a longer phrase. So pure IoU at threshold 0.5 is
    too strict. Score 1.0 if either span fully contains the other, otherwise
    fall back to IoU. Returns 0 if either span is degenerate (zero length).
    """
    a_lo, a_hi = gt_span["char_start"], gt_span["char_end"]
    b_lo, b_hi = ext_span["char_start"], ext_span["char_end"]
    if a_hi <= a_lo or b_hi <= b_lo:
        return 0.0
    if b_lo <= a_lo and b_hi >= a_hi:
        return 1.0
    if a_lo <= b_lo and a_hi >= b_hi:
        return 1.0
    inter = max(0, min(a_hi, b_hi) - max(a_lo, b_lo))
    union = max(a_hi, b_hi) - min(a_lo, b_lo)
    return inter / union if union > 0 else 0.0


def _greedy_match(gt_items: list[dict], extractions: list[dict], match_thresh: float = 0.5) -> list[tuple[int, int]]:
    """Return list of (gt_idx, ext_idx) pairs, greedy by descending match score."""
    pairs: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_items):
        for ei, ext in enumerate(extractions):
            if gt["kind"] != ext["kind"]:
                continue
            score = _match_score(gt["span"], ext["span"])
            if score >= match_thresh:
                pairs.append((score, gi, ei))
    pairs.sort(reverse=True)
    used_gt: set[int] = set()
    used_ext: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gi, ei in pairs:
        if gi in used_gt or ei in used_ext:
            continue
        used_gt.add(gi); used_ext.add(ei)
        matches.append((gi, ei))
    return matches


_VALUE_SYNONYMS = ("value", "stat_value", "statistic", "test_value", "effect_reported", "effect")


def _ext_value(ext: dict):
    f = ext.get("fields", {})
    for k in _VALUE_SYNONYMS:
        v = f.get(k)
        if v is not None:
            return v
    return None


def _is_correct(gt_item: dict, ext: dict) -> bool:
    """Truthful match → values agree; deception → player flagged it as suspicious."""
    if gt_item.get("truthful", True):
        gt_v = gt_item["fields"].get("value")
        ext_v = _ext_value(ext)
        if gt_v is None or ext_v is None:
            return False
        try:
            return abs(float(gt_v) - float(ext_v)) < 0.01
        except (ValueError, TypeError):
            return False
    else:
        return bool(ext.get("flagged_suspicious"))


def _ece(predictions: list[tuple[float, bool]], n_bins: int = 5) -> float:
    """Expected Calibration Error. predictions = [(confidence, is_correct), ...]."""
    if not predictions:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in predictions:
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


def _gt_anchor(gt_item: dict) -> dict:
    span = gt_item.get("span", {}) or {}
    return {
        "char_start": span.get("char_start", 0),
        "char_end": span.get("char_end", 0),
        "kind": gt_item.get("kind", ""),
    }


def _ext_anchor(ext: dict) -> dict:
    span = ext.get("span", {}) or {}
    return {
        "char_start": span.get("char_start", 0),
        "char_end": span.get("char_end", 0),
        "kind": ext.get("kind", ""),
    }


def _try_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_OVERCONFIDENT_THRESHOLD = 0.9


def _emit_findings(gt_items: list[dict], extractions: list[dict],
                   matches: list[tuple[int, int]]) -> list[dict]:
    findings: list[dict] = []
    matched_gt = {gi for gi, _ in matches}
    matched_ext = {ei for _, ei in matches}

    # value_off / deception_missed / deception_false_alarm on matched pairs
    for gi, ei in matches:
        gt = gt_items[gi]
        ext = extractions[ei]
        if gt.get("truthful", True):
            gv = _try_float(gt.get("fields", {}).get("value"))
            ev = _try_float(_ext_value(ext))
            if gv is not None and ev is not None and abs(gv - ev) >= 0.01:
                findings.append({
                    "category": "value_off",
                    "anchor": _gt_anchor(gt),
                    "evidence": str(_ext_value(ext)),
                    "correct_value": gv,
                })
            if ext.get("flagged_suspicious"):
                findings.append({
                    "category": "deception_false_alarm",
                    "anchor": _gt_anchor(gt),
                })
        else:
            if not ext.get("flagged_suspicious"):
                findings.append({
                    "category": "deception_missed",
                    "anchor": _gt_anchor(gt),
                    "correct_value": gt.get("deception_kind") or "untruthful",
                })

    # extraction_missed for unmatched gold
    for gi, gt in enumerate(gt_items):
        if gi in matched_gt:
            continue
        finding = {
            "category": "extraction_missed",
            "anchor": _gt_anchor(gt),
        }
        gv = gt.get("fields", {}).get("value")
        if gv is not None:
            finding["correct_value"] = gv
        findings.append(finding)

    # extraction_extra (or kind_mismatch when an unmatched extraction overlaps a gold span with wrong kind)
    for ei, ext in enumerate(extractions):
        if ei in matched_ext:
            continue
        overlap_with_wrong_kind = None
        for gi, gt in enumerate(gt_items):
            if gt["kind"] == ext.get("kind"):
                continue
            if _match_score(gt.get("span", {}), ext.get("span", {})) >= 0.5:
                overlap_with_wrong_kind = gt
                break
        if overlap_with_wrong_kind is not None:
            findings.append({
                "category": "kind_mismatch",
                "anchor": _gt_anchor(overlap_with_wrong_kind),
                "evidence": ext.get("kind", ""),
                "correct_value": overlap_with_wrong_kind["kind"],
            })
        else:
            ev = _ext_value(ext)
            findings.append({
                "category": "extraction_extra",
                "anchor": _ext_anchor(ext),
                "evidence": str(ev) if ev is not None else "(no value)",
            })

    # calibration_overconfident
    for gi, ei in matches:
        gt = gt_items[gi]
        ext = extractions[ei]
        if _is_correct(gt, ext):
            continue
        conf = _try_float(ext.get("confidence"))
        if conf is not None and conf >= _OVERCONFIDENT_THRESHOLD:
            findings.append({
                "category": "calibration_overconfident",
                "anchor": _gt_anchor(gt),
                "evidence": f"confidence={conf:.2f} on incorrect match",
            })
    for ei, ext in enumerate(extractions):
        if ei in matched_ext:
            continue
        conf = _try_float(ext.get("confidence"))
        if conf is not None and conf >= _OVERCONFIDENT_THRESHOLD:
            findings.append({
                "category": "calibration_overconfident",
                "anchor": _ext_anchor(ext),
                "evidence": f"confidence={conf:.2f} on FP extraction",
            })

    return findings


def score(player_output: dict, ground_truth: dict) -> dict:
    gt_items = ground_truth.get("items", [])
    extractions = player_output.get("extractions", [])

    matches = _greedy_match(gt_items, extractions)
    matched_gt = {gi for gi, _ in matches}
    matched_ext = {ei for _, ei in matches}

    tp = sum(1 for gi, ei in matches if _is_correct(gt_items[gi], extractions[ei]))
    fp = len(extractions) - len(matched_ext) + (len(matches) - tp)
    fn = len(gt_items) - len(matched_gt) + (len(matches) - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    predictions: list[tuple[float, bool]] = []
    for gi, ei in matches:
        is_deception = not gt_items[gi].get("truthful", True)
        correct = _is_correct(gt_items[gi], extractions[ei])
        raw_conf = _safe_conf(extractions[ei])
        flagged = bool(extractions[ei].get("flagged_suspicious"))
        # For deception items, "confidence" = player's belief the stat is genuine.
        # Effective calibration signal: if flagged, player expressed suspicion
        # (1 - raw_conf maps low-confidence → high suspicion, matching correct=True);
        # if not flagged, raw_conf maps overconfidence → wrong, penalising.
        eff_conf = (1.0 - raw_conf) if (is_deception and flagged) else raw_conf
        predictions.append((eff_conf, correct))
    for ei in range(len(extractions)):
        if ei not in matched_ext:
            predictions.append((_safe_conf(extractions[ei]), False))
    calibration = 1.0 - _ece(predictions) if predictions else 1.0

    composite = precision * recall * calibration

    return {
        "primary": composite,
        "breakdown": {
            "precision": precision,
            "recall": recall,
            "calibration": calibration,
            "composite": composite,
            "n_gt": len(gt_items),
            "n_extracted": len(extractions),
            "n_matched": len(matches),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "findings": _emit_findings(gt_items, extractions, matches),
    }

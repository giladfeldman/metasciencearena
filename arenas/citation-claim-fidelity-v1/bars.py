"""Evaluate the pre-registered pass bars in arena.yaml against per-item results.

Item results are dicts {"type", "gold", "verdict", "cluster"?}: verdict is the
GUARDED verdict (None = no verdict), cluster names the abstract the item was
judged against (items sharing an abstract are not independent). They are
produced from a run by tools/analyze_run.py (raw responses and gold from article
custody) or directly by the self-tests. Every number in a bar report comes from
this code.

Hardened after the tier-3 consult of 2026-10-14, still before any model was
scored: a bar whose precondition is not met (too few items of a type, no
grouping result) FAILS, it is never skipped; coverage is required within each
gold class; a 'contradicts' on an abstract-silent sentence is a false suggestion
too; and the false-alarm upper bound is the larger of Wilson and an
abstract-level cluster bootstrap.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

import yaml

ARENA_DIR = Path(__file__).resolve().parent
RECALL_ACCEPT = {"A4": {"contradicts"}, "A2": {"contradicts"}, "C1": {"contradicts"}, "C3": {"contradicts"},
                 "A1": {"contradicts", "silent"}, "B3": {"contradicts", "silent"}}
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261014


def load_bars() -> dict:
    return yaml.safe_load((ARENA_DIR / "arena.yaml").read_text(encoding="utf-8"))["pass_bars"]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_bootstrap_upper(flags: list[tuple[str, bool]], q: float = 0.975) -> float:
    """Percentile bootstrap over clusters (abstracts) of the share of True flags."""
    by = defaultdict(list)
    for cl, f in flags:
        by[cl].append(f)
    clusters = list(by.values())
    if not clusters:
        return float("nan")
    rng = random.Random(BOOTSTRAP_SEED)
    stats = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        pick = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        n = sum(len(c) for c in pick)
        stats.append(sum(sum(c) for c in pick) / n if n else 0.0)
    stats.sort()
    return stats[min(len(stats) - 1, int(q * len(stats)))]


def evaluate(items: list[dict], bars: dict | None = None, grouping: dict | None = None) -> dict:
    """grouping: {"mean_difference", "ci95_lower", "n"} from a paired run, or None (fails the bar)."""
    bars = bars or load_bars()
    out: dict = {"checks": [], "passed": True}

    def check(name, value, ok, detail):
        out["checks"].append({"bar": name, "value": value, "pass": bool(ok), "detail": detail})
        out["passed"] = out["passed"] and bool(ok)

    # Coverage, overall and within each gold class.
    n = len(items)
    covered = sum(1 for it in items if it["verdict"] is not None)
    check("coverage", round(covered / n, 4) if n else None, n and covered / n >= bars["coverage_min"],
          f"{covered}/{n} items with a verdict")
    for g in ("consistent", "silent", "contradicts", "not_consistent"):
        cls = [it for it in items if it["gold"] == g]
        if not cls:
            continue
        c = sum(1 for it in cls if it["verdict"] is not None)
        check(f"coverage_{g}", round(c / len(cls), 4), c / len(cls) >= bars["coverage_min"],
              f"{c}/{len(cls)} gold-{g} items with a verdict")

    # False suggestions: 'contradicts' on a clean sentence, and on an abstract-silent one.
    def alarm_bars(prefix: str, gold: str, spec: dict):
        cls = [it for it in items if it["gold"] == gold]
        if len(cls) < spec.get("min_n", 1):
            check(f"{prefix}_rate", None, False, f"n={len(cls)} < min_n {spec.get('min_n', 1)}: precondition not met")
            return
        fa = sum(1 for it in cls if it["verdict"] == "contradicts")
        rate = fa / len(cls)
        lo, hi = wilson(fa, len(cls))
        boot = cluster_bootstrap_upper([(it.get("cluster", f"i{i}"), it["verdict"] == "contradicts")
                                        for i, it in enumerate(cls)])
        upper = max(hi, boot)
        check(f"{prefix}_rate", round(rate, 4), rate <= spec["max_rate"], f"{fa}/{len(cls)} judged contradicts")
        check(f"{prefix}_upper95", round(upper, 4), upper <= spec["max_upper95"],
              f"max(Wilson {hi:.4f}, cluster bootstrap over abstracts {boot:.4f})")

    alarm_bars("clean_false_alarm", "consistent", bars["clean_false_alarm"])
    alarm_bars("silent_false_alarm", "silent", bars["silent_false_alarm"])

    # Recall per type. A type below min_n FAILS: the task set did not meet the
    # bar's precondition, and a skipped bar let an all-'silent' player pass.
    rb = bars["recall"]
    for t, minimum in sorted(rb["min"].items()):
        typed = [it for it in items if it["type"] == t]
        hits = sum(1 for it in typed if it["verdict"] in RECALL_ACCEPT[t])
        if len(typed) < rb["min_n"]:
            check(f"recall_{t}", (hits / len(typed)) if typed else None, False,
                  f"n={len(typed)} < min_n {rb['min_n']}: precondition not met")
            continue
        r = hits / len(typed)
        check(f"recall_{t}", round(r, 4), r >= minimum, f"{hits}/{len(typed)} (bar {minimum})")

    g = bars["grouping"]
    if grouping is None:
        check("grouping_mean_difference", None, False, "no paired grouping result supplied")
    else:
        check("grouping_mean_difference", round(grouping["mean_difference"], 4),
              grouping["mean_difference"] >= g["min_mean_difference"], f"n={grouping['n']} paired items")
        check("grouping_ci95_lower", round(grouping["ci95_lower"], 4),
              grouping["ci95_lower"] >= g["min_ci95_lower"], "paired 95% CI lower bound")
    return out




# ---------------------------------------------------------------- real-data tiers (task set v3)
# Revised 2026-10-18 after the tier-3 consult (Sonnet + Sol,
# docs/consults/2026-10-18-citeclaim-stage-d-real-data/), before any player was scored on
# a real-data tier: discrimination is reported, never barred (a source-only policy passed
# it); a paired difference resamples each abstract jointly across both groups; a
# false-suggestion bar needs enough independent abstracts, not just enough rows; a tier
# names the bars it requires, so a mis-indented one FAILS instead of vanishing.

def load_tier_bars(tier: str) -> dict:
    tiers = load_bars().get("real_data_tiers") or {}
    if tier not in tiers:
        raise KeyError(f"arena.yaml pass_bars.real_data_tiers declares no bars for {tier!r}")
    return tiers[tier]


def _newcombe_diff_lower(k1: int, n1: int, k2: int, n2: int) -> float:
    """Lower 95% bound of p1 - p2 (Newcombe hybrid score interval, independent samples)."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    return (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)


def cluster_bootstrap_diff_lower(a: list[tuple[str, bool]], b: list[tuple[str, bool]], q: float = 0.025) -> float:
    """Percentile bootstrap of share(a) - share(b), resampling ABSTRACTS jointly.

    An abstract with items in both groups is drawn once and brings all its items of both
    groups (Sol, consult 2026-10-18: resampling each group separately broke that
    dependence and admitted a wrong pass). A replicate with an empty group is skipped;
    if more than 1% are, the bound is NaN (and a bar on it fails).
    """
    by: dict[str, tuple[list[bool], list[bool]]] = defaultdict(lambda: ([], []))
    for cl, f in a:
        by[cl][0].append(f)
    for cl, f in b:
        by[cl][1].append(f)
    units = list(by.values())
    if not units:
        return float("nan")
    rng = random.Random(BOOTSTRAP_SEED)
    stats, skipped = [], 0
    for _ in range(BOOTSTRAP_RESAMPLES):
        pick = [units[rng.randrange(len(units))] for _ in units]
        na, nb = sum(len(u[0]) for u in pick), sum(len(u[1]) for u in pick)
        if not na or not nb:
            skipped += 1
            continue
        stats.append(sum(sum(u[0]) for u in pick) / na - sum(sum(u[1]) for u in pick) / nb)
    if skipped > BOOTSTRAP_RESAMPLES // 100:
        return float("nan")
    stats.sort()
    return stats[max(0, int(q * len(stats)) - 1)]


def discrimination(items: list[dict], spec: dict) -> dict | None:
    """Share judged spec['verdict'] on gold_high minus on gold_low, with its lower 95% bound."""
    a = [it for it in items if it["gold"] == spec["gold_high"]]
    z = [it for it in items if it["gold"] == spec["gold_low"]]
    if not a or not z:
        return None
    ka = sum(1 for it in a if it["verdict"] == spec["verdict"])
    kz = sum(1 for it in z if it["verdict"] == spec["verdict"])
    newc = _newcombe_diff_lower(ka, len(a), kz, len(z))
    boot = cluster_bootstrap_diff_lower(
        [(it.get("cluster", f"a{i}"), it["verdict"] == spec["verdict"]) for i, it in enumerate(a)],
        [(it.get("cluster", f"z{i}"), it["verdict"] == spec["verdict"]) for i, it in enumerate(z)])
    return {"high": [ka, len(a)], "low": [kz, len(z)], "difference": round(ka / len(a) - kz / len(z), 4),
            "ci95_lower": round(min(newc, boot), 4), "newcombe_lower": round(newc, 4),
            "cluster_bootstrap_lower": round(boot, 4) if boot == boot else None}


def evaluate_tier(items: list[dict], spec: dict) -> dict:
    """Bars of one real-data tier (arena.yaml pass_bars.real_data_tiers.<tier>).

    Same hardening as evaluate(): a bar whose precondition is unmet FAILS; coverage is
    required overall and within each gold class; a false-suggestion upper bound is
    max(Wilson, abstract-level cluster bootstrap). Every name in spec['required_bars'] must
    have produced a check, else a failing 'missing_bar' check is added.
    """
    out: dict = {"checks": [], "passed": True, "gating": bool(spec.get("gating", True))}

    def check(name, value, ok, detail):
        out["checks"].append({"bar": name, "value": value, "pass": bool(ok), "detail": detail})
        out["passed"] = out["passed"] and bool(ok)

    n = len(items)
    covered = sum(1 for it in items if it["verdict"] is not None)
    check("coverage", round(covered / n, 4) if n else None, bool(n) and covered / n >= spec["coverage_min"],
          f"{covered}/{n} items with a verdict")
    for g in sorted({it["gold"] for it in items}):
        cls = [it for it in items if it["gold"] == g]
        c = sum(1 for it in cls if it["verdict"] is not None)
        check(f"coverage_{g}", round(c / len(cls), 4), c / len(cls) >= spec["coverage_min"],
              f"{c}/{len(cls)} gold-{g} items with a verdict")

    for name, b in sorted((spec.get("false_suggestion") or {}).items()):
        verdict = b.get("verdict", "contradicts")
        cls = [it for it in items if it["gold"] == b["gold"]]
        clusters = {it.get("cluster", f"i{i}") for i, it in enumerate(cls)}
        if len(cls) < b["min_n"] or len(clusters) < b.get("min_clusters", 0):
            check(f"{name}_rate", None, False, f"n={len(cls)} items in {len(clusters)} abstracts < min_n "
                  f"{b['min_n']} / min_clusters {b.get('min_clusters', 0)}: precondition not met")
            continue
        fa = sum(1 for it in cls if it["verdict"] == verdict)
        _, hi = wilson(fa, len(cls))
        boot = cluster_bootstrap_upper([(it.get("cluster", f"i{i}"), it["verdict"] == verdict)
                                        for i, it in enumerate(cls)])
        upper = max(hi, boot)
        check(f"{name}_rate", round(fa / len(cls), 4), fa / len(cls) <= b["max_rate"],
              f"{fa}/{len(cls)} gold-{b['gold']} judged {verdict} ({len(clusters)} abstracts)")
        check(f"{name}_upper95", round(upper, 4), upper <= b["max_upper95"],
              f"max(Wilson {hi:.4f}, cluster bootstrap over abstracts {boot:.4f})")

    for name, b in sorted((spec.get("recall") or {}).items()):
        cls = [it for it in items if it["gold"] == b["gold"]]
        hits = sum(1 for it in cls if it["verdict"] in b["accept"])
        if len(cls) < b["min_n"]:
            check(name, (hits / len(cls)) if cls else None, False,
                  f"n={len(cls)} < min_n {b['min_n']}: precondition not met")
            continue
        check(name, round(hits / len(cls), 4), hits / len(cls) >= b["min"],
              f"{hits}/{len(cls)} gold-{b['gold']} judged {'/'.join(b['accept'])} (bar {b['min']})")

    made = {c["bar"] for c in out["checks"]}
    for req in spec.get("required_bars") or []:
        if req not in made:
            check(req, None, False, "required bar produced no check (missing or mis-indented in arena.yaml)")
    if not spec.get("required_bars"):
        check("required_bars", None, False, "the tier declares no required_bars")
    return out

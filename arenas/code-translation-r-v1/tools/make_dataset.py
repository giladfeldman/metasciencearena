"""Deterministically materialize the fixed datasets for code-translation-r-v1.

Every player's emitted R reads the SAME CSV, so any difference in the resulting
statistics is attributable to the translation and not to the data. The CSV is
therefore a committed fixture, regenerated from a seed rather than hand-edited.

Pure stdlib (no numpy/pandas): the dataset must be byte-reproducible on any
machine, and `random.Random(seed)` is a stable, documented generator whereas
numpy's default_rng stream is not guaranteed across versions.

    python arenas/code-translation-r-v1/tools/make_dataset.py
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ARENA_DIR / "source_scripts" / "data"
MISSING = ""  # empty cell == NA in both R's read.csv and SPSS/Stata import


def _gauss(rng: random.Random, mu: float, sigma: float) -> float:
    return rng.gauss(mu, sigma)


def build_wellbeing(seed: int = 20260803, n: int = 180) -> list[dict]:
    """The `wellbeing` fixture described in source_scripts/datasets.yaml.

    Every feature here exists to make one cross-language default observable:
      * unbalanced group x condition cells -> Type I != Type III SS
      * unequal group variances            -> pooled t != Welch t
      * genuine NAs in age/hours           -> listwise != pairwise deletion
      * a 99 code in item4                 -> user-missing != literal value
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    for i in range(1, n + 1):
        # Unbalanced by construction: group 1 is ~60% of the sample, and
        # condition is skewed toward level 1. Balanced cells would make the
        # Type I / Type III distinction invisible.
        group = 1 if rng.random() < 0.6 else 2
        r = rng.random()
        condition = 1 if r < 0.45 else (2 if r < 0.75 else 3)

        # Group 2 has both a higher mean AND a wider spread, so the pooled and
        # Welch t-tests disagree on df and p.
        base = 50.0 + (5.0 if group == 2 else 0.0)
        spread = 14.0 if group == 2 else 8.0
        score = _gauss(rng, base, spread) + {1: 0.0, 2: 2.5, 3: -1.5}[condition]
        # A mild interaction, so the interaction F is not ~0.
        if group == 2 and condition == 3:
            score += 4.0

        age = None if rng.random() < 0.04 else min(75, max(18, _gauss(rng, 41, 12)))
        hours = None if rng.random() < 0.03 else min(60, max(0, _gauss(rng, 38, 9)))
        # `score` must ALSO be missing sometimes, or pairwise and listwise
        # deletion agree and the T7 trap tests nothing: with score complete,
        # "complete on age+hours" and "complete on all three" are the same rows.
        # Kept rare so the other analyses (t-test, ANOVA, weighting) still see
        # nearly the full sample.
        score_missing = rng.random() < 0.035

        row = {
            "id": i,
            "age": MISSING if age is None else round(age, 1),
            "hours": MISSING if hours is None else round(hours, 1),
            "score": MISSING if score_missing else round(score, 3),
            "group": group,
            "condition": condition,
        }
        # Case weight for the T8 WEIGHT BY trap. Deliberately CORRELATED with
        # score (higher scorers carry more weight) so the weighted mean differs
        # from the unweighted one — an uncorrelated weight would make the trap
        # invisible. Integer, because SPSS WEIGHT BY is a frequency weight.
        row["w"] = 1 + int(max(0.0, (score - 40.0)) // 12)

        for k in (1, 2, 3):
            row[f"item{k}"] = rng.randint(1, 5)
        # item4 carries the user-missing code in ~12% of rows.
        row["item4"] = 99 if rng.random() < 0.12 else rng.randint(1, 5)
        rows.append(row)
    return rows


DATASETS = {"wellbeing": build_wellbeing}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in DATASETS.items():
        rows = builder()
        out = OUT_DIR / f"{name}.csv"
        # newline="" + LF keeps the file byte-identical on Windows and POSIX.
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

        n_missing_age = sum(1 for r in rows if r["age"] == MISSING)
        n99 = sum(1 for r in rows if r["item4"] == 99)
        cells: dict[tuple[int, int], int] = {}
        for r in rows:
            cells[(r["group"], r["condition"])] = cells.get((r["group"], r["condition"]), 0) + 1
        print(f"wrote {out} ({len(rows)} rows)")
        print(f"  missing age={n_missing_age}  item4==99: {n99}")
        print(f"  group x condition cells: {sorted(cells.items())}")
        # Guard the properties the traps depend on. If a future edit balances the
        # cells or removes the NAs, the arena silently stops testing what it claims.
        assert n_missing_age >= 3, "need genuine NAs for the listwise/pairwise trap"
        assert n99 >= 10, "need enough 99 codes for the user-missing trap"
        assert len(cells) == 6, "need all 6 group x condition cells populated"
        counts = sorted(cells.values())
        assert counts[-1] > counts[0] * 1.5, "cells too balanced — Type I vs III would not diverge"

        # T7 pairwise-vs-listwise: the two deletion rules only DIVERGE when the
        # missingness in age and hours does not fully overlap. If every row
        # missing `hours` were also missing `age`, both rules would keep the same
        # cases and the trap would silently test nothing.
        n_missing_hours = sum(1 for r in rows if r["hours"] == MISSING)
        n_either = sum(1 for r in rows if r["age"] == MISSING or r["hours"] == MISSING)
        n_both = n_missing_age + n_missing_hours - n_either
        print(f"  missing hours={n_missing_hours}  rows missing BOTH={n_both}  "
              f"either={n_either}")
        assert n_missing_hours >= 3, "need NAs in hours too"
        assert n_either > max(n_missing_age, n_missing_hours), \
            "age/hours missingness overlaps completely — pairwise == listwise, trap is dead"

        # The decisive T7 check: pairwise uses cases complete on the PAIR, listwise
        # uses cases complete on ALL THREE variables in the matrix. Those coincide
        # unless the third variable (score) is itself sometimes missing — which is
        # exactly how the first version of this dataset silently produced identical
        # pairwise and listwise gold (r=0.06733, N=165 for both).
        n_pair = sum(1 for r in rows if r["age"] != MISSING and r["hours"] != MISSING)
        n_all3 = sum(1 for r in rows if r["age"] != MISSING and r["hours"] != MISSING
                     and r["score"] != MISSING)
        print(f"  pairwise N={n_pair}  listwise N={n_all3}  (must differ)")
        assert n_pair > n_all3, \
            "pairwise N == listwise N — score is never missing, so the T7 trap tests nothing"

        # T8 WEIGHT BY: the weight must actually move the mean.
        pairs = [(r["w"], r["score"]) for r in rows if r["score"] != MISSING]
        ws = [w for w, _ in pairs]
        sc = [s for _, s in pairs]
        unw = sum(sc) / len(sc)
        wtd = sum(w * s for w, s in zip(ws, sc)) / sum(ws)
        print(f"  weight range={min(ws)}-{max(ws)}  unweighted mean={unw:.3f}  "
              f"weighted={wtd:.3f}  delta={abs(wtd - unw):.3f}")
        assert abs(wtd - unw) > 0.5, "weight barely moves the mean — WEIGHT BY trap is dead"


if __name__ == "__main__":
    main()

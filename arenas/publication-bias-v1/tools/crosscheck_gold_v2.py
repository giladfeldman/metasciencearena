"""Recompute the v2 gold with INDEPENDENT implementations and report the largest gaps.

    .venv/Scripts/python.exe arenas/publication-bias-v1/tools/crosscheck_gold_v2.py --split revealed

The gold (tools/pubbias_v2_stats.R) comes from metafor and RoBMA. This recomputes
each deterministic statistic a second way, from code that shares nothing with it:

  egger.p, pet.estimate, peese.estimate   numpy weighted least squares + scipy t
  begg.tau, begg.p                        scipy.stats.kendalltau (exact)
  trimfill.k0, trimfill.estimate          the `meta` R package (trimfill, REML, L0,
                                          random-effects trimming)
  three_psm.estimate, three_psm.lrt_p     the `weightr` R package (weightfunct);
                                          it crashes on some near-zero-tau sets,
                                          which are reported as such, not skipped

RoBMA has no second implementation; its gold is checked against other MCMC seeds
by build_gold_v2.py --stability-seeds instead.

Prints only per-statistic MAXIMUM absolute gaps and counts, never a dataset's
values, so it is safe on the private split.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ARENA_DIR = Path(__file__).resolve().parents[1]
RSCRIPT = os.environ.get("RSCRIPT", r"C:\Program Files\R\R-4.4.0\bin\Rscript.exe")

R_CODE = r"""
suppressPackageStartupMessages({library(jsonlite); library(meta); library(weightr); library(metafor)})
d <- fromJSON(Sys.getenv("XC_IN"))
out <- list()
for (id in names(d)) {
  st <- d[[id]]
  m  <- metagen(TE = st$yi, seTE = st$sei, method.tau = "REML", common = FALSE, random = TRUE)
  # meta and metafor choose the side by different rules (meta: its own test;
  # metafor: sign of the sei slope in a REML meta-regression of yi on sqrt(vi)).
  # They disagreed on 1 of 24 v2 datasets (2026-09-30). The prompt names metafor,
  # so the side is aligned to metafor's rule here; the disagreement is counted.
  own <- trimfill(m, ma.common = FALSE, type = "L")
  slope <- rma(st$yi, st$sei^2, mods = st$sei, method = "REML",
               control = list(stepadj = 0.5, maxiter = 1000))$beta[2]
  tf <- trimfill(m, ma.common = FALSE, type = "L", left = slope >= 0)
  w  <- tryCatch(weightfunct(effect = st$yi, v = st$sei^2), error = function(e) NULL)
  r  <- list(tf_k0 = tf$k0, tf_est = tf$TE.random, side_disagrees = own$left != (slope >= 0))
  if (!is.null(w)) {
    lr <- 2 * abs(w[[1]]$value - w[[2]]$value)
    r$psm_est <- w$adj_est[2]; r$psm_p <- 1 - pchisq(lr, 1)
  }
  out[[id]] <- r
}
cat(toJSON(out, auto_unbox = TRUE, digits = NA))
"""


def _wls(yi, x, w):
    X = np.column_stack([np.ones_like(x), x])
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X, XtW @ yi)
    resid = yi - X @ beta
    df = len(yi) - 2
    sigma2 = float((w * resid * resid).sum() / df)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(XtW @ X)))
    return beta, se, df


def python_stats(studies):
    yi = np.array([s["yi"] for s in studies], float)
    sei = np.array([s["sei"] for s in studies], float)
    vi = sei ** 2
    w = 1 / vi
    beta, se, df = _wls(yi, sei, w)
    egger_p = 2 * stats.t.sf(abs(beta[1] / se[1]), df)
    peese, _, _ = _wls(yi, vi, w)
    theta = (w * yi).sum() / w.sum()
    z = (yi - theta) / np.sqrt(vi - 1 / w.sum())
    kt = stats.kendalltau(z, vi, method="exact")
    return {"egger.p": egger_p, "pet.estimate": beta[0], "peese.estimate": peese[0],
            "begg.tau": kt.statistic, "begg.p": kt.pvalue}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["revealed", "private"], required=True)
    ap.add_argument("--task-set", default="v2")
    args = ap.parse_args(argv)

    spec = importlib.util.spec_from_file_location("_pubbias_gen_xc", ARENA_DIR / "generator.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    seed = 0
    if args.split == "private":
        seed = int((ARENA_DIR / "task_sets" / args.task_set / ".private_seed").read_text().strip())
    envs = list(gen.generate(args.task_set, seed, split=args.split))
    gold = {e["task_id"]: gen.ground_truth(e["task_id"])["gold_stats"] for e in envs}

    payload = {e["task_id"]: {"yi": [s["yi"] for s in e["input"]["studies"]],
                              "sei": [s["sei"] for s in e["input"]["studies"]]} for e in envs}
    # Code and data go through temp files: `Rscript -e <multi-line code>` with the
    # data on stdin died with a native access violation (0xC0000005) on Windows,
    # 2026-09-30, while the same code from a file ran clean.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        code, data = Path(td) / "xc.R", Path(td) / "in.json"
        code.write_text(R_CODE, encoding="utf-8")
        data.write_text(json.dumps(payload), encoding="utf-8")
        proc = subprocess.run([RSCRIPT, str(code)], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=dict(os.environ, XC_IN=str(data)))
    if proc.returncode != 0:
        raise SystemExit(f"Rscript exit {proc.returncode}: {proc.stderr[-800:]}")
    r = json.loads(proc.stdout.strip().splitlines()[-1])

    gaps: dict[str, list[float]] = {}
    psm_missing = 0
    side_disagreements = 0
    for e in envs:
        tid, g = e["task_id"], gold[e["task_id"]]
        py = python_stats(e["input"]["studies"])
        for key, val in py.items():
            m, s = key.split(".")
            gaps.setdefault(key, []).append(abs(val - g[m][s]))
        side_disagreements += bool(r[tid]["side_disagrees"])
        gaps.setdefault("trimfill.k0", []).append(abs(r[tid]["tf_k0"] - g["trimfill"]["k0"]))
        gaps.setdefault("trimfill.estimate", []).append(abs(r[tid]["tf_est"] - g["trimfill"]["estimate"]))
        if "psm_est" in r[tid]:
            gaps.setdefault("three_psm.estimate", []).append(abs(r[tid]["psm_est"] - g["three_psm"]["estimate"]))
            gaps.setdefault("three_psm.lrt_p", []).append(abs(r[tid]["psm_p"] - g["three_psm"]["lrt_p"]))
        else:
            psm_missing += 1

    print(f"{args.task_set} {args.split}: {len(envs)} datasets")
    for key in sorted(gaps):
        v = gaps[key]
        print(f"  {key:20s} n={len(v):2d}  max|gap|={max(v):.2e}  n(|gap|>0.01)={sum(x > 0.01 for x in v)}")
    print(f"  weightr could not fit {psm_missing} of {len(envs)} (three_psm compared on the rest)")
    print(f"  meta's own side rule disagreed with metafor's on {side_disagreements} of {len(envs)} "
          "(trim-and-fill compared with the side aligned to metafor's rule)")
    print("  weightr's 3PSM is a SECOND implementation, not a referee: where it disagrees, check "
          "whether its adjusted log-likelihood fell below the unadjusted one (non-convergence)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

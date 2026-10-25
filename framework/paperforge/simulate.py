"""Simulate participants under a study's ACTUAL plan and compute every reported statistic.

This module is the only source of numbers in a paperforge paper. The prose model
never writes a digit (enforced in `prose.py`); it refers to placeholders that are
filled from the dict `simulate()` returns.

The data-generating model is deliberately ordinary: a latent outcome per scale,
Likert items with one weak item, careless responders who fail attention checks and
answer at random, item-level missingness, and a baseline covariate that correlates
with the outcome. Ordinary is the point: the arena tests whether a tool notices
where the paper departs from its preregistration, and the numbers must behave like
real data while doing so (exclusions change N, imputation changes SDs, ANCOVA
changes the test).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from framework.paperforge.study import Plan, Study

_LOADING = 0.78
_WEAK_LOADING = 0.22
_BASELINE_R = 0.45       # baseline covariate -> outcome latent
_SCALE_R = 0.40          # primary latent <-> secondary latent (residual)


def _likert(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(np.rint(4.0 + 1.45 * x + rng.normal(0, 0.35, x.shape)), 1, 7)


def _items(eta: np.ndarray, k: int, weak: int, rng: np.random.Generator) -> np.ndarray:
    cols = []
    for i in range(1, k + 1):
        lam = _WEAK_LOADING if i == weak else _LOADING
        cols.append(lam * eta + math.sqrt(1 - lam ** 2) * rng.normal(0, 1, eta.shape))
    return _likert(np.column_stack(cols), rng)


def cronbach_alpha(items: np.ndarray) -> float:
    items = items[~np.isnan(items).any(axis=1)]
    k = items.shape[1]
    if k < 2 or items.shape[0] < 3:
        raise ValueError("alpha needs >= 2 items and >= 3 complete cases")
    item_var = items.var(axis=0, ddof=1).sum()
    total_var = items.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var / total_var))


def _cohen_d(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp
    se = math.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    return float(d), float(d - 1.96 * se), float(d + 1.96 * se)


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Coefficients, their t values, residual df. X includes the intercept column."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = X.shape[0] - X.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return beta, beta / np.sqrt(np.diag(cov)), dof


def _p_from_t(t: float, dof: float, tails: int, predicted_sign: int) -> float:
    if tails == 2:
        return float(2 * stats.t.sf(abs(t), dof))
    return float(stats.t.sf(t * predicted_sign, dof))


def _test(y: np.ndarray, cond: np.ndarray, base: np.ndarray, plan: Plan, sign: int) -> dict:
    """The confirmatory test the plan names, on one outcome."""
    treat, ctrl = y[cond == 1], y[cond == 0]
    d, d_lo, d_hi = _cohen_d(treat, ctrl)
    if plan.model == "welch":
        t, _ = stats.ttest_ind(treat, ctrl, equal_var=False)
        v1, v2 = treat.var(ddof=1) / len(treat), ctrl.var(ddof=1) / len(ctrl)
        dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(treat) - 1) + v2 ** 2 / (len(ctrl) - 1))
        p = _p_from_t(float(t), dof, plan.tails, sign)
        return {"kind": "welch", "t": float(t), "df": float(dof), "p": p,
                "d": d, "d_lo": d_lo, "d_hi": d_hi}
    X = np.column_stack([np.ones_like(y), cond, base])
    beta, tv, dof = _ols(y, X)
    t = float(tv[1])
    p = _p_from_t(t, dof, plan.tails, sign)
    f = t ** 2
    return {"kind": "ancova", "F": f, "df1": 1, "df2": dof, "p": p, "t": t,
            "b": float(beta[1]), "eta_p2": f / (f + dof),
            "d": d, "d_lo": d_lo, "d_hi": d_hi}


def simulate(study: Study, seed: int) -> dict:
    """Simulate the study under `study.actual` and return every reportable quantity.

    Deterministic in (study, seed). Raises when the analysed sample would be too
    small for the tests, instead of printing a statistic that means nothing.
    """
    rng = np.random.default_rng(seed)
    plan, eff, topic = study.actual, study.effects, study.topic
    n = plan.recruited_n
    sign_p = -1 if topic["primary"].get("direction") == "lower" else 1
    sign_s = -1 if topic["secondary"].get("direction") == "lower" else 1

    cond = (np.arange(n) % 2) if plan.randomization == "alternation" else rng.integers(0, 2, n)
    if cond.sum() in (0, n):
        raise ValueError("randomisation produced an empty condition")
    mod = rng.integers(0, 2, n)
    z = rng.normal(0, 1, n)
    base_obs = np.clip(np.rint(4 + 1.3 * z + rng.normal(0, 0.5, n)), 1, 7)
    careless = rng.random(n) < eff.careless_rate
    fails = np.where(careless, rng.binomial(3, 0.55, n), rng.binomial(3, 0.03, n))

    e_shared = rng.normal(0, 1, n)
    resid_sd = math.sqrt(1 - _BASELINE_R ** 2)
    eta_p = (_BASELINE_R * z + resid_sd * (math.sqrt(_SCALE_R) * e_shared
             + math.sqrt(1 - _SCALE_R) * rng.normal(0, 1, n)))
    eta_s = (_BASELINE_R * z + resid_sd * (math.sqrt(_SCALE_R) * e_shared
             + math.sqrt(1 - _SCALE_R) * rng.normal(0, 1, n)))
    eta_p = eta_p + sign_p * (eff.d_primary * cond + eff.interaction * cond * (mod - 0.5) * 2)
    eta_s = eta_s + sign_s * eff.d_secondary * cond

    sp, ss = topic["primary"], topic["secondary"]
    items_p = _items(eta_p, sp["k"], sp["weak_item"], rng)
    items_s = _items(eta_s, ss["k"], ss["weak_item"], rng)
    for items in (items_p, items_s):
        items[careless] = rng.integers(1, 8, (int(careless.sum()), items.shape[1]))
    # Item-level missingness: one skipped item for a random subset of participants.
    for items in (items_p, items_s):
        miss = rng.random(n) < eff.missing_rate
        cols = rng.integers(0, items.shape[1], n)
        items[miss, cols[miss]] = np.nan

    age = np.clip(np.rint(rng.gamma(3.0, 5.5, n) + 18), 18, 79)
    platform = topic["platform"] if plan.source == "main" else topic.get("alt_platform", "")
    if "undergraduate" in platform:
        age = np.clip(np.rint(rng.normal(20.4, 2.3, n)), 18, 35)
    women = rng.random(n) < 0.56

    threshold = 2 if plan.exclusion == "attn2" else 1
    keep = fails < threshold
    n_excl_attention = int((~keep).sum())

    # The planned primary scale's composite may drop its weak item (an `index` choice).
    use_p = [i for i in range(sp["k"]) if not (plan.index == "drop_weak" and i + 1 == sp["weak_item"])]
    ip, is_ = items_p[:, use_p], items_s.copy()

    any_missing = np.isnan(ip).any(axis=1) | np.isnan(is_).any(axis=1)
    n_with_missing = int((any_missing & keep).sum())
    if plan.missing == "listwise":
        keep = keep & ~any_missing
        n_excl_missing = n_with_missing
    else:
        n_excl_missing = 0
        for arr in (ip, is_):
            col_means = np.nanmean(arr[keep], axis=0)
            r, c = np.where(np.isnan(arr))
            arr[r, c] = col_means[c]

    ip, is_ = ip[keep], is_[keep]
    c, b, m = cond[keep], base_obs[keep], mod[keep]
    if min((c == 1).sum(), (c == 0).sum()) < 10:
        raise ValueError("fewer than 10 analysed participants in a condition")
    comp_p, comp_s = ip.mean(axis=1), is_.mean(axis=1)

    reported_primary = plan.primary               # catalog role reported as THE outcome
    y_main = comp_p if reported_primary == "primary" else comp_s
    y_other = comp_s if reported_primary == "primary" else comp_p
    sign_main = sign_p if reported_primary == "primary" else sign_s
    sign_other = sign_s if reported_primary == "primary" else sign_p

    main = _test(y_main, c, b, plan, sign_main)
    other = _test(y_other, c, b, plan, sign_other)
    Xm = np.column_stack([np.ones_like(y_main), c, m, c * m])
    _, tv, dofm = _ols(y_main, Xm)
    t_int = float(tv[3])
    moderation = {"F": t_int ** 2, "df1": 1, "df2": dofm,
                  "p": float(2 * stats.t.sf(abs(t_int), dofm))}

    def desc(y):
        return {"treat_m": float(y[c == 1].mean()), "treat_sd": float(y[c == 1].std(ddof=1)),
                "ctrl_m": float(y[c == 0].mean()), "ctrl_sd": float(y[c == 0].std(ddof=1))}

    return {
        "n_recruited": int(n),
        "n_excl_attention": n_excl_attention,
        "n_with_missing": n_with_missing,
        "n_excl_missing": n_excl_missing,
        "n_final": int(keep.sum()),
        "n_treat": int((c == 1).sum()),
        "n_ctrl": int((c == 0).sum()),
        "age_m": float(age[keep].mean()), "age_sd": float(age[keep].std(ddof=1)),
        "n_women": int(women[keep].sum()),
        "alpha_planned_primary": cronbach_alpha(items_p[keep][:, use_p]),
        "alpha_planned_primary_all_items": cronbach_alpha(items_p[keep]),
        "alpha_planned_secondary": cronbach_alpha(items_s[keep]),
        "reported_primary_role": reported_primary,
        "desc_main": desc(y_main), "desc_other": desc(y_other),
        "desc_baseline": desc(b.astype(float)),
        "test_main": main, "test_other": other,
        "moderation": moderation,
        "r_main_other": float(np.corrcoef(y_main, y_other)[0, 1]),
        "correlations": np.corrcoef(np.vstack([y_main, y_other, b.astype(float)])).tolist(),
        "weak_item_dropped": plan.index == "drop_weak",
        "significant_main": main["p"] < 0.05,
        "direction_main_as_predicted": (main["d"] * sign_main) > 0,
    }

"""Structured study specification for a paperforge synthetic paper.

A study is a TOPIC (what the experiment is about; from an arena catalog) plus two
PLANS: the one the preregistration commits to and the one the paper actually
carried out. A deviation is a difference between the two plans, injected here,
before any data exist. The data are then simulated under the ACTUAL plan, so every
number the paper prints follows from what the authors "did", and the only
disagreement between the documents is the one the arena put there on purpose.

Design family implemented so far: a two-group between-subjects online experiment
with two multi-item outcome scales, a single-item baseline covariate, three
attention checks and a binary moderator. That family is the modal design of
OSF-template preregistrations in social and applied psychology; other families
(within-subject, correlational, RCT with follow-up) are listed in the spec.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

from scipy import stats

#: Every plan field and the values it may take. The FIRST value is never implied to
#: be the "good" one; which value a preregistration commits to is drawn per task.
PLAN_VALUES: dict[str, tuple] = {
    "exclusion": ("attn2", "attn1"),           # fail >=2 of 3 checks  vs  fail >=1
    "missing": ("listwise", "mean_impute"),
    "control": ("active", "passive"),
    "primary": ("primary", "secondary"),       # which catalog scale is THE outcome
    "index": ("all_items", "drop_weak"),        # composite of the planned primary scale
    "model": ("welch", "ancova"),
    "tails": (2, 1),
    "randomization": ("simple", "alternation"),
    "moderation": ("exploratory", "confirmatory"),
    "source": ("main", "alt"),                 # topic.platform  vs  topic.alt_platform
}


@dataclass(frozen=True)
class Plan:
    """What one document says was (to be) done. `target_n` is the preregistered
    total sample; `recruited_n` is what the study actually recruited, and only the
    ACTUAL plan's value is used by the simulation."""
    target_n: int
    recruited_n: int
    exclusion: str = "attn2"
    missing: str = "listwise"
    control: str = "active"
    primary: str = "primary"
    index: str = "all_items"
    model: str = "welch"
    tails: int = 2
    randomization: str = "simple"
    moderation: str = "exploratory"
    source: str = "main"

    def __post_init__(self):
        for k, allowed in PLAN_VALUES.items():
            if getattr(self, k) not in allowed:
                raise ValueError(f"plan.{k}={getattr(self, k)!r} not in {allowed}")
        if self.target_n < 20 or self.recruited_n < 20:
            raise ValueError("a two-group study needs at least 20 participants")

    def with_(self, **changes) -> "Plan":
        return replace(self, **changes)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Effects:
    """True population effects the simulation draws from (standardised units).

    `d_primary` / `d_secondary` are signed in the PREDICTED direction: positive
    means the treatment moved the scale the way the hypothesis says it would.
    """
    d_primary: float
    d_secondary: float
    interaction: float = 0.0
    careless_rate: float = 0.08
    missing_rate: float = 0.06


@dataclass(frozen=True)
class PowerAnalysis:
    d: float
    power: float
    alpha: float
    n_per_group: int
    target_n: int
    attrition_allowance: float


def power_analysis(d: float, power: float, alpha: float = 0.05,
                   attrition_allowance: float = 0.10) -> PowerAnalysis:
    """A priori sample size for a two-sided independent-samples t-test.

    Exact noncentral-t search (the same computation G*Power runs), not a normal
    approximation, so the preregistration's rationale sentence is reproducible.
    `target_n` inflates the total for the anticipated exclusions and rounds up to a
    multiple of ten, as preregistrations usually do.
    """
    n = 2
    while True:
        n += 1
        dof = 2 * n - 2
        ncp = d * math.sqrt(n / 2)
        crit = stats.t.ppf(1 - alpha / 2, dof)
        achieved = (1 - stats.nct.cdf(crit, dof, ncp)) + stats.nct.cdf(-crit, dof, ncp)
        if achieved >= power:
            break
        if n > 5000:
            raise ValueError(f"no n <= 5000 reaches power {power} for d = {d}")
    total = 2 * n / (1 - attrition_allowance)
    target = int(math.ceil(total / 10.0) * 10)
    return PowerAnalysis(d=d, power=power, alpha=alpha, n_per_group=n, target_n=target,
                         attrition_allowance=attrition_allowance)


@dataclass
class Study:
    """A topic, the two plans, the true effects, and the planning rationale."""
    topic: dict
    planned: Plan
    actual: Plan
    effects: Effects
    power: PowerAnalysis
    extras: dict = field(default_factory=dict)

    def differing_fields(self) -> list[str]:
        """Plan fields on which the paper departs from the preregistration."""
        out = [k for k in PLAN_VALUES if getattr(self.planned, k) != getattr(self.actual, k)]
        if self.planned.target_n != self.actual.recruited_n:
            out.insert(0, "sample_size")
        return out

    def scale(self, which: str) -> dict:
        """Catalog entry for 'primary' / 'secondary' (the CATALOG role, not the plan's)."""
        return self.topic[which]

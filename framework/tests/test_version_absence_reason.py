"""An adapter with no resolvable version must say WHY, and only it can know.

THE DEFECT THIS CLOSES
----------------------
`framework audit --versions` reported 225 registered players as 127 OK / 0 DRIFT /
98 UNCHECKED, and every one of the 98 printed the same sentence:

    adapter reported no installed version

True, and useless. Measured 2026-08-27, that one sentence was covering four
different facts, including a live outage:

    64  by_design      a CLI-driven model; the model id IS the version
    19  needs_prepare  version computed by prepare(), which the audit never calls
    10  undeclared     the adapter has not said
     5  probe_failed   GROBID did not answer /api/version -- IT IS NOT RUNNING

The five GROBID players were indistinguishable from the 64 for which None is
entirely correct. "The guard could not run" was rendering as "there is nothing to
guard", which is this project's founding failure shape.

WHY A METHOD ON THE ADAPTER AND NOT A TABLE IN THE AUDIT
-------------------------------------------------------
Because the table was tried and it was wrong. On 2026-08-26 a class-override
heuristic was written to classify the 98 from outside, and `pcurve` disproved it:
`pcurve.R` loads only `jsonlite`, so it is pure base R BY DESIGN and has no
package version -- externally identical to an R package whose probe failed.
Increment 2 recorded the conclusion as "making adapters state a REASON is a real,
unstarted increment". This is that increment.

THE DEFAULT IS `undeclared` ON PURPOSE
--------------------------------------
An adapter that has not thought about this says so. Defaulting to `by_design`
would silently assert "nothing is wrong here" on behalf of every adapter anyone
adds in future -- an unlabelled fallback, which this project has a standing rule
against. `undeclared` is the honest state and it is visible.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import pytest

import framework.runner  # noqa: F401  MUST import first -- see _registry_adapters
from framework.player_adapter import (
    PlayerAdapter, VersionAbsence, VERSION_ABSENT_BY_DESIGN,
    VERSION_ABSENT_NEEDS_PREPARE, VERSION_ABSENT_PROBE_FAILED,
    VERSION_ABSENT_UNDECLARED, VERSION_ABSENT_PROBLEM_KINDS, build_adapter)
from framework.paths import registry_path
from framework.registry import load_registry

VALID_KINDS = {VERSION_ABSENT_BY_DESIGN, VERSION_ABSENT_NEEDS_PREPARE,
               VERSION_ABSENT_PROBE_FAILED, VERSION_ABSENT_UNDECLARED}


@lru_cache(maxsize=1)
def _registry_adapters():
    """Build every registered adapter, the way the audit does. CACHED.

    Building the registry probes the network (GROBID's version check is a real
    HTTP GET with a 3s timeout), so this is computed once per session. Without
    the cache the eight tests below rebuilt 225 adapters each and the file took
    minutes instead of seconds.

    `import framework.runner` at module scope is load-bearing: it runs
    `_autoload_adapter_modules()`, and without it 159 adapters fail to build and
    this sweep reports 9 of 225 instead of 225 -- the exact bogus number the
    portfolio rules already name.
    """
    reg = load_registry(registry_path())
    out = []
    for entry in (reg if isinstance(reg, list) else list(reg)):
        try:
            out.append((entry.get("player_id", "?"), build_adapter(entry)))
        except Exception:
            continue
    return tuple(out)


@lru_cache(maxsize=1)
def _unresolved():
    """(player_id, adapter) for every player with no resolvable tool version."""
    out = []
    for pid, a in _registry_adapters():
        try:
            if a.resolved_tool_version() is None:
                out.append((pid, a))
        except Exception:
            out.append((pid, a))
    return tuple(out)


def test_CONTROL_the_sweep_sees_the_whole_registry():
    """Without `import framework.runner` this builds 9 adapters, not 225.

    A sweep that silently shrinks makes every assertion below pass while checking
    almost nothing -- and it looks exactly like success. 9-of-225 is the exact
    bogus number the portfolio rules already name.

    ASSERTED AS A RATIO, NOT A COUNT, and deliberately so. This file ships to the
    public package, where only 67 of the adapter modules exist, so an absolute
    floor of 200 would fail on a perfectly healthy mirror while passing here.
    The ratio detects the actual failure mode on either tree: the autoload trap
    collapses it to ~4%, whereas a legitimately smaller public registry keeps it
    near 1.0 because registry.yaml is subset alongside the adapters.
    """
    reg = load_registry(registry_path())
    registered = len(reg if isinstance(reg, list) else list(reg))
    built = len(_registry_adapters())
    assert registered > 0, "registry.yaml resolved to zero players"
    assert built / registered >= 0.8, (
        f"only {built} of {registered} registered adapters built "
        f"({built / registered:.0%}). The ADAPTER AUTOLOAD is what broke, not "
        f"the adapters -- `import framework.runner` runs "
        f"_autoload_adapter_modules() and without it 159 fail to build"
    )


def test_every_unresolved_player_states_a_valid_kind():
    bad = []
    for pid, a in _unresolved():
        try:
            r = a.version_absence_reason()
        except Exception as exc:
            bad.append(f"{pid}: raised {type(exc).__name__}: {exc}")
            continue
        if not isinstance(r, VersionAbsence):
            bad.append(f"{pid}: returned {type(r).__name__}, not VersionAbsence")
        elif r.kind not in VALID_KINDS:
            bad.append(f"{pid}: unknown kind {r.kind!r}")
    assert not bad, "\n".join(bad)


def test_version_absence_reason_never_raises():
    """It is called while rendering an audit. A reason that raises turns a
    diagnostic into a crash, which is strictly worse than the vague sentence it
    replaced."""
    for pid, a in _unresolved():
        a.version_absence_reason()  # must not raise


def test_the_kinds_actually_discriminate():
    """The load-bearing control.

    If every adapter returned one kind, this file would be green while conveying
    exactly as much as the single sentence it replaced. Assert the split is real.
    """
    kinds = Counter(a.version_absence_reason().kind for _, a in _unresolved())
    assert len(kinds) >= 3, (
        f"the reason channel collapsed to {dict(kinds)} -- fewer than three "
        f"distinct kinds means it is not discriminating anything"
    )
    total = sum(kinds.values())
    assert kinds[VERSION_ABSENT_BY_DESIGN] / total >= 0.4, (
        f"expected CLI-driven models to dominate by_design, got {dict(kinds)}"
    )


def test_a_grobid_player_reports_an_OUTAGE_not_a_design_choice():
    """The finding that justifies the whole increment.

    `_tool_version.grobid_version()` returns None ONLY when
    `GET <endpoint>/api/version` raises or answers non-200. There is no "this
    GROBID has no version" case, so None here always means the service is
    unreachable -- and it used to render identically to the 64 players for which
    None is correct.
    """
    grobid = [(pid, a) for pid, a in _unresolved()
              if type(a).__name__.startswith("Grobid")]
    if not grobid:
        pytest.skip("no GROBID player is currently unresolved (service may be up)")
    for pid, a in grobid:
        r = a.version_absence_reason()
        assert r.kind == VERSION_ABSENT_PROBE_FAILED, (
            f"{pid} reports {r.kind!r}; an unreachable GROBID is a PROBE FAILURE, "
            f"not a design choice, and calling it by_design hides an outage"
        )
        assert r.kind in VERSION_ABSENT_PROBLEM_KINDS


def test_a_local_gguf_player_reports_UNBUILT_not_absent():
    """Their version is not missing, it is unbuilt.

    `LocalGgufAdapter.__init__` sets `_tool_version = None` and only `prepare()`
    fills it -- verified by AST 2026-08-27. The runner calls `prepare()` before
    any task, so every run record carries the full
    `<stem>/<quant>@<engine>@host-<fingerprint>`; the audit does not, because
    preparing a local player loads a multi-gigabyte model.
    """
    gguf = [(pid, a) for pid, a in _unresolved()
            if type(a).__name__ == "LocalGgufAdapter"]
    if not gguf:
        pytest.skip("no local-GGUF player registered")
    for pid, a in gguf:
        assert a.version_absence_reason().kind == VERSION_ABSENT_NEEDS_PREPARE, (
            f"{pid} should report needs_prepare: its version exists and is "
            f"computed by prepare(), which the audit deliberately skips"
        )
    assert VERSION_ABSENT_NEEDS_PREPARE not in VERSION_ABSENT_PROBLEM_KINDS, (
        "needs_prepare is a lifecycle fact, not a fault; gating on it would fail "
        "every audit on a machine that has local players registered"
    )


def test_CONTROL_the_base_default_is_undeclared_not_a_claim():
    """An adapter that has not thought about this must not inherit an all-clear.

    If the default were `by_design`, every future adapter would silently assert
    "nothing is wrong here" without anyone deciding it -- and a real probe failure
    added tomorrow would arrive pre-labelled as fine.
    """
    class _Bare(PlayerAdapter):
        def play_task(self, envelope, *, timeout_s):
            return {}

    a = _Bare(player_id="x", player_version="v", player_type="tool",
              confidence_strategy="native", deterministic=True)
    r = a.version_absence_reason()
    assert r.kind == VERSION_ABSENT_UNDECLARED
    assert r.kind != VERSION_ABSENT_BY_DESIGN
    assert "_Bare" in r.detail, "the detail must name the adapter that has not declared"


def test_only_a_real_fault_is_a_problem_kind():
    """`probe_failed` is the only kind an audit may treat as a fault. The other
    three are legitimate states, and gating on them would fail a clean tree."""
    assert VERSION_ABSENT_PROBLEM_KINDS == {VERSION_ABSENT_PROBE_FAILED}
    for ok in (VERSION_ABSENT_BY_DESIGN, VERSION_ABSENT_NEEDS_PREPARE,
               VERSION_ABSENT_UNDECLARED):
        assert ok not in VERSION_ABSENT_PROBLEM_KINDS

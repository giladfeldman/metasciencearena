"""Shared Rscript probe for the R adapter smoke tests.

Why this exists (2026-08-04, cycle 7): every R smoke test used to gate itself on
`shutil.which("Rscript")`, i.e. on R being on PATH. On this dev box R 4.4.0 is
installed but NOT on PATH, so all nine R smoke tests SKIPPED — the suite reported
green while the R reference tools (scrutiny, metacheck, oddpub, rtransparent,
statcheck, metafor, effectsize) were entirely untested. Those tools are what
cross-validate each arena's gold, so "skipped" there is the most expensive kind of
false green.

`resolve_rscript_binary()` (framework.player_adapter) is the single place the
runner resolves the interpreter: explicit > RSCRIPT_BINARY env var > PATH. Tests
must resolve it the SAME way, or they gate on a different condition than the code
they cover.
"""
from __future__ import annotations

import shutil

from framework.player_adapter import resolve_rscript_binary


def rscript_path() -> str | None:
    """The Rscript this environment would actually invoke, or None if unavailable.

    Mirrors the runner's resolution order, then confirms the result is really
    executable — an RSCRIPT_BINARY pointing at a missing file must skip, not fail
    the suite with a confusing FileNotFoundError deep inside a subprocess call.
    """
    resolved = resolve_rscript_binary(None)
    if shutil.which(resolved):
        return shutil.which(resolved)
    return None


RSCRIPT_UNAVAILABLE_REASON = (
    "Rscript unavailable: not on PATH and RSCRIPT_BINARY unset or not executable"
)

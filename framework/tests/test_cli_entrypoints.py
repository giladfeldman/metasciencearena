"""Every documented way to invoke the CLI must actually run the CLI.

`python -m framework.cli audit` exited 0 having executed nothing: the module has
no `__main__` guard, so importing it as `__main__` defined some functions and
returned. A shell — or a reviewer, or a CI step — saw exit 0 and read it as
"the audit passed".

That is this project's signature defect class (a check that appears to run but
does not), sitting on the gate the outreach one-pager cites as evidence. Found by
the Fable 5 cross-review, 2026-08-15, which hit it while trying to verify the
"audit exits 0 with 22/22 gates passing" claim.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: The three invocations that must behave identically. The console script is
#: covered by test_clean_install / the packaging gate; these two are the in-repo
#: forms, and `framework.cli` is the one that was silently inert.
MODULE_FORMS = ["framework", "framework.cli"]


def _run(module: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=REPO, capture_output=True, text=True,
    )


@pytest.mark.parametrize("module", MODULE_FORMS)
def test_no_arguments_does_not_silently_succeed(module):
    """Invoked with no subcommand, the CLI must say so — not exit 0 in silence.

    An inert entry point is indistinguishable from a passing one at the shell,
    which is precisely how a vacuous gate gets recorded as green.
    """
    proc = _run(module)
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        f"`python -m {module}` exited 0 having run nothing — a false green"
    )
    assert combined.strip(), f"`python -m {module}` produced no output at all"


@pytest.mark.parametrize("module", MODULE_FORMS)
def test_help_is_reachable_and_mentions_the_gates(module):
    """A real argparse parser answers --help; an inert module does not."""
    proc = _run(module, "--help")
    assert proc.returncode == 0, proc.stderr
    assert "audit" in proc.stdout, (
        f"`python -m {module} --help` did not list the subcommands — "
        "the parser did not run"
    )


def test_both_module_forms_agree():
    """The two in-repo entry points must not diverge.

    If they ever do, one of them is running a different program than the docs,
    the CI steps, and the outreach claims assume.
    """
    outputs = {m: _run(m, "--help").stdout for m in MODULE_FORMS}
    first, *rest = outputs.values()
    assert all(o == first for o in rest), (
        "the two `python -m` entry points print different help:\n"
        + "\n".join(f"--- {m} ---\n{o[:400]}" for m, o in outputs.items())
    )

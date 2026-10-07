"""A `pip install -e <path>` in CI must point at a real Python project.

REGRESSION, 2026-09-12. `scripts/tests/report-engine-parity.test.mjs` began
shelling out to `python -m framework.cli report`, so the `app` job needed the
framework's dependencies rather than the bare `numpy` it had installed since
the ranking bridge landed. The fix was `pip install -e .` — and it failed,
because that job carries `defaults.run.working-directory: leaderboard-app`, so
`.` is the Next app and not the repo root:

    ERROR: file://$RUNNER_HOME/work/sciencearena/sciencearena/leaderboard-app
    does not appear to be a Python project: neither 'setup.py' nor
    'pyproject.toml' found.                     (run 34684340901)

Two pushes, two red runs, for a one-token path error that is fully decidable
from the workflow file. Nothing local looked at it: the YAML is not type
checked, not linted, and not executed until a runner does it. That is the whole
gap this closes — the feedback loop for a CI-config mistake was a push.

SCOPE. Only `pip install -e <path>` is checked, because only the editable form
names a directory that must contain a project. A `pip install <name>` from
PyPI is not a path and is none of this test's business.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))

#: `pip install -e <target>`, capturing the target.
#:
#: The target may be QUOTED and may contain spaces — `-e "${{ github.workspace }}"`
#: is the correct form and a naive `(\S+)` captures only `"${{`. It may also
#: carry an extras spec, `-e ".[dev]"`, which is not part of the path. Both of
#: those broke the first version of this test, on the very workflows it was
#: written to check; the alternation handles quotes and `_strip_target` the rest.
_EDITABLE = re.compile(
    r"""pip\s+install\s+(?:[^\n&|;]*?\s)?-e\s+(?:"([^"]+)"|'([^']+)'|(\S+))"""
)

_WORKSPACE = re.compile(r"\$\{\{\s*github\.workspace\s*\}\}")

#: `.[dev]`, `path/to/pkg[extra1,extra2]` — pip extras, not directory names.
_EXTRAS = re.compile(r"\[[^\]]*\]$")


def _strip_target(raw: str) -> str:
    return _EXTRAS.sub("", raw.strip().strip("\"'")).strip() or "."


def _self_checkout_paths(job: dict) -> set[str]:
    """Directories this job checks THIS repo out into.

    `actions/checkout` with a `path:` relocates the repo, so a step running in
    that directory is running at the repo ROOT even though the path is not `.`
    — `sync-public.yml` does exactly this (`path: private`), and the first
    version of this test called its perfectly correct `pip install -e ".[dev]"`
    a bug. A checkout carrying `repository:` is a DIFFERENT repo (that same
    workflow also checks out the public mirror into `public/`) and is not an
    alias for our root.
    """
    out: set[str] = set()
    for step in job.get("steps") or []:
        uses = str(step.get("uses") or "")
        if not uses.startswith("actions/checkout"):
            continue
        with_ = step.get("with") or {}
        if with_.get("repository"):
            continue
        path = with_.get("path")
        if path:
            out.add(str(path).strip("/"))
    return out


def _steps_with_workdir(wf: dict):
    """Every `run:` step, paired with the working directory it executes in."""
    for job_name, job in (wf.get("jobs") or {}).items():
        job_default = ((job.get("defaults") or {}).get("run") or {}).get("working-directory")
        aliases = _self_checkout_paths(job)
        for step in job.get("steps") or []:
            run = step.get("run")
            if not run:
                continue
            wd = step.get("working-directory") or job_default or "."
            # A working dir that IS a self-checkout path is the repo root.
            if str(wd).strip("/") in aliases:
                wd = "."
            yield job_name, wd, run


def _cases():
    out = []
    for path in WORKFLOWS:
        wf = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job_name, wd, run in _steps_with_workdir(wf):
            for groups in _EDITABLE.findall(run):
                raw = next(g for g in groups if g)
                out.append((path, job_name, wd, raw))
    return out


CASES = _cases()


def test_the_scan_found_editable_installs_to_check() -> None:
    """Two-sided: a regex that matches nothing would pass every assertion below.

    There is at least one editable install in this repo's CI by construction —
    the `app` job needs the framework for the report-engine parity test, and the
    `python` job installs the package to run pytest at all.
    """
    assert WORKFLOWS, "no workflow files found — the scan is looking in the wrong place"
    assert CASES, (
        "no `pip install -e` found in any workflow. Either CI stopped installing "
        "the package (in which case the parity test's python subprocess is broken) "
        "or this test's regex no longer matches — both are faults, not passes."
    )


@pytest.mark.parametrize(
    "wf,job,workdir,target",
    CASES,
    ids=[f"{p.name}:{j}:{t}" for p, j, _, t in CASES],
)
def test_editable_install_target_is_a_python_project(
    wf: Path, job: str, workdir: str, target: str
) -> None:
    cleaned = _strip_target(target)
    if _WORKSPACE.search(cleaned):
        resolved = REPO
    else:
        # The runner resolves a relative target against the step's working dir.
        resolved = (REPO / workdir / cleaned).resolve()

    assert (resolved / "pyproject.toml").is_file() or (resolved / "setup.py").is_file(), (
        f"{wf.name} job `{job}` runs `pip install -e {target}` with "
        f"working-directory `{workdir}`, which resolves to "
        f"{resolved.relative_to(REPO).as_posix() if resolved.is_relative_to(REPO) else resolved} "
        f"— that directory holds neither pyproject.toml nor setup.py, so pip will "
        f"fail with 'does not appear to be a Python project'. Use "
        f"${{{{ github.workspace }}}} rather than a path relative to a "
        f"working-directory that can change."
    )

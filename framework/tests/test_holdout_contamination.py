"""The class-level held-out contamination invariant, asserted over the repo.

This is the durable guard the 2026-06-28 deep review (DR-0007) asked for: a
single test that fails CI/pre-commit if ANY tracked source artifact carries a
held-out task's gold answer, reconstructable player output, per-task score
metadata, or membership oracle. It complements the runner's write-time
redaction and the dump scripts' redaction — those keep new files clean; this
catches a regression (or a manually-edited file) before it can be committed.

The matching public-bundle invariant for the *deployed* artifacts lives in
leaderboard-app/test/report-bundle-contamination.test.ts (DR-0008).

TIER 3 OF THREE — THE DATA PROOF
--------------------------------
The contamination guarantee is proved in three separate places, because one
test cannot do all three jobs once the code is published separately from the
data (2026-08-07):

    tier 1  framework/tests/test_holdout.py    LOGIC, over synthetic fixtures.
            Ships publicly; needs no corpus; proves redaction is correct.
    tier 2  framework/tests/test_publish.py    ARTIFACTS. Proves the set of
            files about to be published carries no held-out material, by
            allowlist AND by verbatim scan against the real held-out corpora.
    tier 3  THIS FILE                          DATA, over the real `arenas/`
            glob. Proves what is actually on disk here is clean.

Only tier 3 needs the private corpora, so only tier 3 skips outside this repo.
The skip is keyed on `.private_seed`, which exists in the private source tree
and is denied by `framework.publish.NEVER_PUBLISH`, so it can never be present
in the mirror. Inside this repo it therefore never skips, and its own
non-vacuity assertions below still apply.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from framework.holdout import HELD_OUT, held_out_leak_reasons, is_held_out

REPO_ROOT = Path(__file__).resolve().parents[2]
ARENAS = REPO_ROOT / "arenas"

#: True only in the private source tree. A private seed is never mirrored, so
#: this cannot be spoofed by a public checkout — and it is a stronger signal
#: than "does arenas/ exist", which the mirror also satisfies.
IS_PRIVATE_SOURCE = ARENAS.is_dir() and any(ARENAS.glob("*/task_sets/*/.private_seed"))

pytestmark = pytest.mark.skipif(
    not IS_PRIVATE_SOURCE,
    reason="tier-3 data proof: needs the private corpora, which only exist in the source repo",
)


def _ground_truth_files() -> list[Path]:
    return sorted(ARENAS.glob("*/task_sets/*/_ground_truth.json"))


def _run_record_files() -> list[Path]:
    """Every run-record file that is a durable artifact.

    Excludes `*.retry-r<N>.jsonl`. Those are transient per-round temps written by
    `framework retry-failed` and merged back into the target file moments later —
    `framework/cli.py::_recover_orphan_retry_temps` exists precisely because they
    are expected to come and go. Parametrising over them makes this suite fail
    with FileNotFoundError whenever a tournament happens to be running, which says
    nothing about contamination. Their MERGED content is still covered, because
    the file they merge into is globbed here.
    """
    return sorted(
        p for p in ARENAS.glob("*/runs/**/*.jsonl")
        if ".retry-r" not in p.name
    )


def _safe_visibility(line: str):
    try:
        return json.loads(line).get("task_visibility")
    except ValueError:
        return None


def _iter_json_lines(path: Path):
    """Yield parsed records, SKIPPING unparseable lines.

    Torn lines are a real thing — killing a tournament mid-write truncates the
    record being flushed, and `storage.read_records` documents tolerating one torn
    trailing line for exactly that reason. They are a data-integrity problem, not a
    contamination problem, so `test_no_torn_run_record_lines` reports them with a
    clear message and the redaction assertions below carry on doing their own job
    rather than dying on a JSONDecodeError from an unrelated fault.
    """
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except ValueError:
            continue


def test_no_torn_run_record_lines() -> None:
    """Every line of every run file must parse.

    A process killed mid-flush leaves a truncated line. `storage.read_records`
    tolerates ONE at the end of a file; anything mid-file means a record was lost,
    and a silently short file is a shrinking-evidence failure.
    """
    torn = []
    for path in _run_record_files():
        lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        for i, line in enumerate(lines):
            try:
                json.loads(line)
            except ValueError:
                torn.append(f"{path.relative_to(REPO_ROOT)} line {i+1}/{len(lines)}")
    assert not torn, (
        "truncated JSON line(s) — a run was killed mid-write and a record was lost: "
        + "; ".join(torn)
    )


def test_some_tracked_artifacts_exist() -> None:
    # Guards against the glob silently matching nothing (e.g. a path refactor),
    # which would make every assertion below vacuously pass.
    assert _ground_truth_files(), "no _ground_truth.json files found — glob broken?"
    assert _run_record_files(), "no run-record JSONL files found — glob broken?"


def test_held_out_artifacts_actually_exist() -> None:
    """Non-vacuity: files existing is not the same as HELD-OUT ENTRIES existing.

    ``held_out_leak_reasons`` returns ``[]`` for a public entry, so a corpus
    containing only public tasks passes every assertion below while proving
    nothing about redaction. That is precisely the state a PUBLIC split of this
    repo would be in — it ships the revealed tasks only — so the guard has to
    assert it is actually looking at held-out material.
    """
    gt_held_out = 0
    for path in _ground_truth_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        gt_held_out += sum(1 for e in data.values() if is_held_out((e or {}).get("envelope")))

    rec_held_out = 0
    for path in _run_record_files():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and _safe_visibility(line) == HELD_OUT:
                rec_held_out += 1

    assert gt_held_out > 0, (
        "no held-out ground-truth entries found — the redaction assertions below "
        "would pass vacuously. If this split genuinely has no held-out tasks, the "
        "guard belongs in the repo that does."
    )
    assert rec_held_out > 0, "no held-out run records found — see above."


def test_every_run_record_declares_its_visibility() -> None:
    """A record with no ``task_visibility`` is INVISIBLE to the leak check.

    ``held_out_leak_reasons(..., kind="record")`` returns ``[]`` the moment the
    field is missing, so an unmarked record is reported clean no matter what it
    carries. Ground-truth entries fail SAFE (a missing ``visibility`` is treated
    as held-out); records failed OPEN. That asymmetry means a runner change, a
    hand-edited rescore, or a new arena could park held-out output in a tracked
    file and this suite would stay green.

    Found 2026-08-07: 136 tracked records carried no marker, 107 of them with a
    non-empty ``output``. None mapped to a held-out task — so it was latent, not
    a breach — but nothing prevented the next one from being real.
    """
    offenders: list[str] = []
    for path in _run_record_files():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "task_visibility" not in record:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{i} (task={record.get('task_id')})")

    assert not offenders, (
        f"{len(offenders)} tracked run record(s) declare no task_visibility, so the "
        f"held-out leak check skips them entirely:\n  "
        + "\n  ".join(offenders[:10])
        + ("\n  ..." if len(offenders) > 10 else "")
        + "\n  Every record must carry task_visibility. Re-run the arena through "
          "`framework run` (which writes it), or backfill the field."
    )


@pytest.mark.parametrize("gt_path", _ground_truth_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_tracked_ground_truth_carries_no_held_out_gold(gt_path: Path) -> None:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    offenders = []
    for task_id, entry in data.items():
        reasons = held_out_leak_reasons(entry, kind="ground_truth")
        if reasons:
            offenders.append(f"{task_id}: {'; '.join(reasons)}")
    assert not offenders, (
        f"{gt_path.relative_to(REPO_ROOT)} leaks held-out gold:\n  " + "\n  ".join(offenders)
        + "\n  Re-run the arena's tools/dump_ground_truth.py (it strips held-out gold)."
    )


@pytest.mark.parametrize("rec_path", _run_record_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_tracked_run_records_carry_no_held_out_content(rec_path: Path) -> None:
    offenders = []
    for i, line in enumerate(rec_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        reasons = held_out_leak_reasons(record, kind="record")
        if reasons:
            offenders.append(f"line {i} ({record.get('task_id')}): {'; '.join(reasons)}")
    assert not offenders, (
        f"{rec_path.relative_to(REPO_ROOT)} leaks held-out content:\n  " + "\n  ".join(offenders[:10])
        + ("\n  ..." if len(offenders) > 10 else "")
        + "\n  Held-out records must be written through framework.holdout.redact_held_out_record."
    )

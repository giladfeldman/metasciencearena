"""The premise of the 2026-09-11 custody ruling on stored extractor prose.

WHY THIS TEST EXISTS
--------------------
`publication-text-scan.py --tracked-only --include-suspect` reports 76 tracked
files as `suspect` in this repo, 38 of them arena run records carrying 4.1 MB of
prose. That is not a defect: a text-extraction arena cannot be audited without
storing what the extractor emitted, and the portfolio custody rule forbids
holding *publication* text, which this is not.

Measured 2026-09-11, repo-wide over 458 tracked run-record files (11,931
records), the prose is safe for exactly one reason, and it is a structural one:

    every task built from a REAL paper is `held_out`, and every `public` task is
    a document this project authored.

So the stored prose is our own synthetic corpus, and the real papers contribute
nothing but a redacted shell (2,110 held-out records, 4,220 bytes of `output`
total — literally `{}` each).

WHICH DENOMINATOR — read this before concluding the numbers above are wrong
--------------------------------------------------------------------------
They count EVERY tracked run record, archives included, because an archived run
record is still a tracked file and so still in custody scope — and the custody
scan's 76 suspect hits include them. This test uses the same denominator.

Re-deriving on LIVE records only gives a different and equally correct set.
Excluding `_archive/`, `_pilot_archive/`, `_revisions/` and `.retry-r*.jsonl`
(all four — a filter missing `_revisions/` lands between the two and matches
neither), measured 2026-09-12:

    archives included : 458 files / 11,931 records / 9,821 public / 2,110
                        held_out / 4,220 bytes of held-out `output`
    live records only : 232 files /  6,514 records / 5,776 public /   738
                        held_out / 1,476 bytes of held-out `output`

Both were reproduced independently by a second session. The structural claim is
identical under either: zero real-paper tasks are `public`.

That premise is what makes the exemption narrow. Without a gate it is also
exactly the kind of premise that silently stops being true: one real open-access
paper added as a `public` task would put genuine publication text into a tracked
run record AND onto the public board, and nothing else in the suite asks this
question. `test_holdout_contamination.py` guards the other half — that held-out
records carry no content — and is blind to this one, because a `public` record
is *supposed* to carry its output.

If this test fails, do NOT widen it. Either mark the task `held_out`, or re-open
the ruling in `.claude/skills/sciencearena-cleanup/SKILL.md` (Phase 0a, rule 3)
and in `arenas/pdf-text-fidelity-v1/CHANGELOG.md` and decide again, in writing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARENAS = REPO_ROOT / "arenas"

IS_PRIVATE_SOURCE = ARENAS.is_dir() and any(ARENAS.glob("*/task_sets/*/.private_seed"))

pytestmark = pytest.mark.skipif(
    not IS_PRIVATE_SOURCE,
    reason="data proof: needs the private corpora, which only exist in the source repo",
)

#: A task id built from a real published paper. Three arms, and the middle one
#: was MISSING until 2026-09-12 — the original pattern was `pmc-?PMC\d|10\.…/`,
#: which does not match `sections-apa-*` (6 real papers, 48 records: chen,
#: ip_feldman, jamison, maier, xiao, ziano). All six are `held_out` today, which
#: is exactly why the gap would have shipped unnoticed: the gate was silent on
#: the one case it exists to catch.
#:
#: Match ANYWHERE in the id, never anchored at the start. An anchored
#: `startswith("pmc-")` misses `tables-pmc-*` (96 records), `sections-pmc-*`
#: (90) and `sections-apa-*` (18) — 204 in total, found when a second session
#: and this one reconciled 390 against 594 on the same files.
#:
#: The DOI arm matches ZERO ids today. It is forward cover for the DOI-keyed APA
#: corpus, not an exercised control — `test_the_probe_is_not_vacuous` therefore
#: asserts the `pmc` and `apa` arms individually and deliberately does not
#: require a DOI hit.
#:
#: Everything else names a document this project wrote: `synth-*`, `sections-synth-*`,
#: `ref-ref-synth-*`, `match-match-synth-*`, `rtl-v*`, `ts-t*`, `xlat-*`, and the
#: `t-tier*` family (generated statistical sentences — `-s0`/`-s5feceb66` are the
#: public seeds, other seed hashes are held out, so it spans both visibilities by
#: design).
REAL_PAPER_TASK_ID = re.compile(r"pmc-?PMC\d|(?:^|[-_])apa-|10\.\d{4,9}/", re.IGNORECASE)

#: The individual arms, so the non-vacuity control can prove EACH still matches.
#: A single total-only floor stays green when one arm silently breaks.
REAL_PAPER_ARMS = {
    "pmc": re.compile(r"pmc-?PMC\d", re.IGNORECASE),
    "apa": re.compile(r"(?:^|[-_])apa-", re.IGNORECASE),
}


def _run_record_files() -> list[Path]:
    return sorted(ARENAS.glob("*/runs/**/*.jsonl"))


def _records():
    for path in _run_record_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            yield path, lineno, json.loads(line)


def test_no_real_paper_task_is_public() -> None:
    """The ruling's premise: stored extractor prose is never a real paper's text."""
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno} "
        f"({rec.get('arena_id')} / {rec.get('task_id')})"
        for path, lineno, rec in _records()
        if REAL_PAPER_TASK_ID.search(str(rec.get("task_id", "")))
        and rec.get("task_visibility") == "public"
    ]
    assert not offenders, (
        f"{len(offenders)} tracked run record(s) store a REAL paper's task as `public`, so "
        "the extractor's output — that paper's text — is held in this repo and published to "
        "the board:\n  " + "\n  ".join(offenders[:10])
        + ("\n  ..." if len(offenders) > 10 else "")
        + "\n  See framework/tests/test_extraction_corpus_is_synthetic.py's docstring."
    )


def test_the_probe_is_not_vacuous() -> None:
    """Two-sided control: a zero above must be a real zero, not a broken regex.

    PER-ARM, deliberately. A single total-only floor is not a control: with
    1,738 PMC records in this repo, a total of `>= 100` stays comfortably green
    while the `apa` arm (48 records) matches nothing at all. That is not
    hypothetical — the `apa` arm did not exist until 2026-09-12, and a
    total-only floor is exactly what failed to notice.

    The DOI arm is intentionally absent here: it matches zero ids today and is
    forward cover, so requiring a hit would fail on a healthy repo. It is
    documented as inert rather than counted as a control.
    """
    seen = [
        rec
        for _, _, rec in _records()
        if REAL_PAPER_TASK_ID.search(str(rec.get("task_id", "")))
    ]

    #: Floors well under the measured counts (pmc 1,738 / apa 48 on 2026-09-12),
    #: so ordinary corpus churn does not redden the suite, but a DEAD arm does.
    floors = {"pmc": 100, "apa": 10}
    for arm, pattern in REAL_PAPER_ARMS.items():
        hits = [r for r in seen if pattern.search(str(r.get("task_id", "")))]
        assert len(hits) >= floors[arm], (
            f"the {arm!r} arm of REAL_PAPER_TASK_ID matched {len(hits)} task id(s), below its "
            f"floor of {floors[arm]}. Measured 2026-09-12: pmc 1,738, apa 48. An arm that "
            "matches nothing makes test_no_real_paper_task_is_public pass vacuously for every "
            "paper named that way — do not lower this floor to go green."
        )

    stray = [r.get("task_id") for r in seen if r.get("task_visibility") != "held_out"]
    assert not stray, (
        f"{len(stray)} real-paper task(s) are not held_out: {stray[:10]}. "
        "This is the same defect test_no_real_paper_task_is_public reports; fix it there."
    )

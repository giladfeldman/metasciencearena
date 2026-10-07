"""The task count an arena PUBLISHES must be the number of tasks it has.

WHY THIS FILE EXISTS (2026-09-21). `arena.yaml#task_set_versions[].n_tasks` is
copied into the published bundle by `build-data.mjs` (`result.ladder`, the
`n_tasks: entry.n_tasks ?? null` line) and rendered as the "Tasks" column of
`components/version-ladder.tsx` on every arena's About page. Nothing compared it
to the task set.

Measured that day, on the bundle built the same afternoon, three arenas were
publishing a wrong count:

    pdf-section-structure-v1 v1   declared 10   task set holds 46
    pdf-table-extraction-v1  v1   declared 12   task set holds 36
    pdf-text-fidelity-v1     v1   declared 40   task set holds 70

`index.json` carried `{'version': 'v1', 'status': 'current', 'n_tasks': 10, ...}`
for the first of them. The cause is the versioning gap this gate is named after:
the shared `source_docs/` corpus GREW IN PLACE, no task-set version was cut, and
the declared count never moved with it.

WHAT THIS IS NOT. It is not a frozen-history defect. `framework audit --fresh`
reports `[OK] pdf-section-structure-v1 v1 (current): 46 tasks reproduce their
fingerprint`, i.e. the `generator_fingerprint` was baselined after the growth and
still reproduces, so no published score was computed against tasks that have
since changed. Only the label was wrong. That distinction is why the fix was to
correct three numbers rather than to cut three task-set versions.

AND NOTE WHERE THE NUMBER WAS ALREADY AVAILABLE. The audit line quoted above
prints the true count. The repository computed the right answer and never
compared it to the claim -- the same shape as the ESCImate version drift fixed
earlier the same day, where `resolved_tool_version` existed and the audit could
not reach it. This gate closes the comparison, it does not add a measurement.

THE SPLIT RULE, MEASURED RATHER THAN ASSUMED. `n_tasks` means the count a reader
is shown, and that is NOT the same question for every arena:

* An arena that declares `benchmark_splits` publishes its REVEALED count.
  `stats-extraction-v1/arena.yaml` says so in its own comment ("the revealed
  single-seed count actually generated"), and `code-translation-r-v1` is the
  control: it declares 18 while its `_ground_truth.json` holds 19 keys, the
  nineteenth being `xlat-holdout-paired-ttest-repeated-spss-s1`.
* An arena with no splits publishes its whole task set. The five `pdf-*` arenas
  are these; their records carry `split: None`.

Getting this wrong produces confident nonsense in BOTH directions, which is why
it is spelled out here. Asking every arena for its "revealed" split reported
`pdf-citation-matching-v1` as 36-declared / 6-generated -- a false alarm, because
that arena has no splits and the revealed seed path emits a fragment. Counting
`_ground_truth.json` keys instead reported `code-translation-r-v1` as 18/19 -- the
opposite false alarm, because that file carries the held-out task too. Both
"defects" evaporated once the instrument matched the question.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from framework.audit import current_input_hashes
from framework.paths import arenas_root

#: How many ladder entries must be GENUINELY CHECKED for this file to mean
#: anything. Measured 2026-09-21: 28 entries across 22 arenas, 28 checked, 0
#: unresolvable. Set below that so a machine missing one private seed still runs
#: the gate, and far enough above zero that a broken generator -- which would
#: turn every row into "cannot check" and every assertion below into a vacuous
#: pass -- fails this file instead of sailing through it.
MIN_CHECKED = 20

#: `SplitNote.kind`s that mean "absent from THIS machine", as opposed to "broken".
#: `framework/audit.py` defines three: "empty" (the split really has no tasks),
#: "asset" (a gitignored corpus or secret seed is not here -- "Normal on a fresh
#: clone or a second machine; nothing in the code is wrong") and "failed"
#: (anything else -- "the only one of the three worth gating on").
#:
#: An ALLOWLIST, not a denylist, deliberately. A kind this file has never heard
#: of must count as a problem: an unrecognised reason is not a reassuring one,
#: and a denylist would silently absolve every kind added after today.
BENIGN_ABSENCE_KINDS = frozenset({"empty", "asset"})


@dataclass(frozen=True)
class Row:
    arena: str
    version: str
    declared: int | None
    generated: int | None
    reason: str
    #: The `SplitNote.kind`s behind an absence, from the CODE PATH and never
    #: parsed out of `reason`. `framework/audit.py` names three and the
    #: difference is the whole point: "asset" means a gitignored corpus or a
    #: secret seed is not on THIS machine, which is normal on a fresh clone;
    #: "failed" means a guard stopped working. A test that treats them alike is
    #: either red on every CI runner or blind on every workstation.
    kinds: tuple[str, ...] = ()

    @property
    def checked(self) -> bool:
        return self.declared is not None and self.generated is not None

    @property
    def unavailable_here(self) -> bool:
        """Absent because this machine lacks a local-only asset, not a defect.

        NOT `framework.audit.blocking_kinds`, and the reason is worth keeping:
        that helper returns `{"asset", "failed"}` because it answers a DIFFERENT
        question -- "did a split fail to contribute when it should have?", for
        which a missing corpus and a broken generator are equally blocking. The
        question here is "is this absence a fact about the MACHINE or about the
        CODE?", and `SplitNote`'s own docstring separates them: "asset" is
        "Normal on a fresh clone or a second machine; nothing in the code is
        wrong", while "failed" is "the only one of the three worth gating on".
        Reaching for the helper whose NAME matched my intent, rather than the
        one whose SEMANTICS did, is how this test was wrong the first time.
        """
        return bool(self.kinds) and all(
            k in BENIGN_ABSENCE_KINDS for k in self.kinds)

    @property
    def disagrees(self) -> bool:
        return self.checked and self.declared != self.generated


def count_verdict(declared: int | None, generated: int | None) -> str:
    """The pure comparison, so it can be exercised on numbers this tree no
    longer contains. Three answers, never two: "could not check" is a distinct
    outcome from "checked and agreed", and collapsing them is how a gate reports
    a clean pass over nothing.
    """
    if declared is None or generated is None:
        return "cannot-check"
    return "agrees" if declared == generated else "disagrees"


def _rows() -> list[Row]:
    rows: list[Row] = []
    for arena_dir in sorted(arenas_root().iterdir()):
        manifest = arena_dir / "arena.yaml"
        if not manifest.exists():
            continue
        loaded = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        # See THE SPLIT RULE above. `split=None` merges every split, which is the
        # right question only for an arena that declares none.
        split = "revealed" if loaded.get("benchmark_splits") else None
        for entry in loaded.get("task_set_versions") or []:
            notes: list = []
            try:
                hashes = current_input_hashes(
                    arena_dir, entry["version"], split=split, errors=notes)
            except Exception as exc:                      # pragma: no cover - env
                rows.append(Row(arena_dir.name, entry["version"],
                                entry.get("n_tasks"), None,
                                f"generator raised {type(exc).__name__}: {exc}",
                                ("failed",)))
                continue
            reason = ""
            if not hashes:
                # `current_input_hashes` returns {} for an arena it cannot
                # generate in-process and records WHY in `errors`. Keep the
                # reason: "this arena is not generable here" and "the generator
                # broke" both arrive as an empty dict and are not the same fact.
                reason = "; ".join(
                    f"{getattr(n, 'kind', '?')}: {getattr(n, 'detail', '')}" for n in notes
                ) or "the generator emitted no tasks and gave no reason"
            rows.append(Row(arena_dir.name, entry["version"],
                            entry.get("n_tasks"), len(hashes) or None, reason,
                            tuple(getattr(n, "kind", "failed") for n in notes)))
    return rows


@pytest.fixture(scope="module")
def rows() -> list[Row]:
    """Regenerating every task set is the expensive part; do it once.

    SCIENCEARENA_SYNTHETIC_ONLY is REMOVED for the generation (2026-09-27). CI
    declares it for the whole suite, and under it `require_corpus` returns None
    with only a stderr warning, so a pdf-* generator emits its synthetic subset
    (10 of 46, 12 of 36, 40 of 70...) and leaves no note. That row then looked
    CHECKED and disagreed with the published count -- red on every CI run since
    2026-09-22, green on every workstation. A synthetic count is not the
    published count; without the flag the absent corpus raises
    HeldOutCorpusMissing, which `audit` records as an "asset" note, and the
    watch test below reports it as an honest skip.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("SCIENCEARENA_SYNTHETIC_ONLY", raising=False)
        return _rows()


def test_every_declared_task_count_matches_the_task_set(rows):
    """The published "Tasks" column must be the number of tasks that exist."""
    wrong = [r for r in rows if r.disagrees]
    assert not wrong, (
        "arena.yaml declares a task count its task set does not have, and that "
        "number is published on the arena's About page:\n  "
        + "\n  ".join(
            f"{r.arena} {r.version}: declares n_tasks={r.declared}, the generator "
            f"emits {r.generated}" for r in wrong)
        + "\nCorrect the declaration -- do NOT change the generator to match it. "
        "`framework audit --fresh` prints the true count per version and is the "
        "second opinion. If the task set genuinely changed, that is a task-set "
        "VERSION bump, not an edit to this label."
    )


def test_the_gate_is_not_vacuous(rows):
    """Every assertion here passes trivially on rows nobody could check.

    A generator that stops working, an arenas root pointed somewhere empty, or a
    missing private seed all turn `current_input_hashes` into `{}` -- and a file
    full of "cannot check" rows is indistinguishable from a clean sweep unless
    something counts them.
    """
    assert rows, (
        "zero ladder entries found at all. The arenas root is misconfigured "
        "(SCIENCEARENA_ARENAS_ROOT), and every other assertion in this file "
        "passed without testing anything."
    )
    checked = [r for r in rows if r.checked]
    assert len(checked) >= MIN_CHECKED, (
        f"only {len(checked)} of {len(rows)} ladder entries could be checked, "
        f"below the {MIN_CHECKED} this gate needs to mean anything. Unresolvable "
        "rows and their reasons:\n  "
        + "\n  ".join(
            f"{r.arena} {r.version}: declared={r.declared} generated={r.generated} "
            f"-- {r.reason or 'no n_tasks declared'}"
            for r in rows if not r.checked)
    )


def test_the_three_arenas_this_gate_was_written_for_are_still_watched(rows):
    """Pin the regression cases by name, so they cannot regress into "skipped".

    Without this, a later change that made the pdf-* arenas unresolvable would
    leave the file green while the exact three rows it exists to watch stopped
    being watched -- the shape this repo calls a false green.

    WHY THIS IS NOT A FLAT `assert row.checked` (corrected 2026-09-22, the day
    after it was written). It was, and it was RED on main -- green on the
    workstation that wrote it and red in a clean worktree, which is this repo's
    signature CI-only failure. All three arenas draw their tasks from the
    held-out PMC corpus under a gitignored `_held_out/pmc`, so a fresh clone
    reports `asset: HeldOutCorpusMissing` and cannot generate them at all.

    `framework/audit.py` already draws the distinction this test needed and I
    did not use it: `SplitNote.kind` is "asset" for a local-only file that is
    absent ("Normal on a fresh clone or a second machine; nothing in the code is
    wrong"), "failed" for anything else, and `blocking_kinds()` is the function
    that says which is which. Keying on the KIND rather than on the prose is the
    same discipline the module's own docstring demands.

    So: an ASSET absence is an honest skip and is reported. Anything else -- the
    arena vanishing from the ladder, or an absence with no kind or a blocking
    one -- is still a failure, because those are the ways the watch really stops.
    """
    want = {
        ("pdf-section-structure-v1", "v1"): 46,
        ("pdf-table-extraction-v1", "v1"): 36,
        ("pdf-text-fidelity-v1", "v1"): 70,
    }
    by_key = {(r.arena, r.version): r for r in rows}
    # THE PUBLIC MIRROR (2026-10-01). All three are PRIVATE arenas, so the mirror
    # ships none of their directories and the ladder cannot contain them there.
    # That is the same "local-only asset absent" case as above, one level up.
    # Keyed on the manifest's absence -- the mirror never ships
    # `contract/public_manifest.txt` and the private repo always tracks it -- so
    # a private checkout that LOST an arena directory still fails below.
    root = Path(__file__).resolve().parents[2]
    in_public_mirror = not (root / "contract" / "public_manifest.txt").is_file()
    unavailable = []
    for key, expected in want.items():
        row = by_key.get(key)
        if row is None and in_public_mirror and not (root / "arenas" / key[0]).is_dir():
            unavailable.append(f"{key[0]} {key[1]}: private arena, not in the public mirror")
            continue
        assert row is not None, (
            f"{key} is no longer in any arena.yaml ladder. This gate was written "
            "for it after it published a wrong task count; removing it silently "
            "is how that returns."
        )
        if not row.checked:
            assert row.unavailable_here, (
                f"{key} cannot be checked and NOT because this machine lacks a "
                f"local-only asset: {row.reason}. An unexplained absence is the "
                "one kind worth gating on."
            )
            unavailable.append(f"{key[0]} {key[1]}: {row.reason}")
            continue
        assert row.generated == expected, (
            f"{key} now generates {row.generated} tasks, not the {expected} "
            "measured when this gate was written. That is a real task-set change "
            "and needs a version bump plus a re-baselined generator_fingerprint, "
            "not an edit to this expectation."
        )
    if len(unavailable) == len(want):
        # SKIPPED, not passed. All three draw on the same absent corpus, so
        # there is nothing left to assert -- and a silent pass here would be
        # indistinguishable from a check that ran, which is the exact false
        # green this file exists to prevent. A skip is countable: a run whose
        # skip count moves is a run on a different machine, and this repo has
        # been caught by that before (the mirror gate's harness was not the
        # public checkout).
        pytest.skip(
            "none of the three pinned arenas can be generated here; all draw on "
            "the gitignored held-out PMC corpus. Not a defect -- but this test "
            "asserted nothing, so do not read it as a pass:\n  "
            + "\n  ".join(unavailable))
    if unavailable:
        # Partial. RECORDED rather than silent, and the rest still checked.
        print("\n[task-count gate] not verifiable on this machine (local-only "
              "assets absent):\n  " + "\n  ".join(unavailable))


def test_a_planted_wrong_count_is_caught():
    """Mutation proof, on the numbers this tree carried BEFORE the fix.

    Watched red the only way that does not require mutating a working tree three
    other sessions are using: the comparator is fed the three real pre-fix pairs
    and must call every one of them a disagreement, and the corrected pairs and
    must call every one of them agreement.
    """
    before = [(10, 46), (12, 36), (40, 70)]
    for declared, actual in before:
        assert count_verdict(declared, actual) == "disagrees", (declared, actual)
    for _, actual in before:
        assert count_verdict(actual, actual) == "agrees", actual
    # And the third answer is its own answer, never folded into "agrees".
    assert count_verdict(46, None) == "cannot-check"
    assert count_verdict(None, 46) == "cannot-check"
    assert count_verdict(None, None) == "cannot-check"

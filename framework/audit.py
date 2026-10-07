"""Leaderboard fairness audits: tool-version drift + task-set symmetry.

Two checks behind the 2026-06-12 accuracy handoff:

* ``version_drift_from_registry`` (Finding 1) — compares each player's DECLARED
  registry ``player_version`` against the version its installed adapter actually
  resolves at runtime (``resolved_tool_version``). A leaderboard that ranks tools
  on a stale declared label ranks history; this flags rows to refresh + re-run.

* ``task_set_symmetry`` (Finding 2) — a head-to-head is only fair when the
  compared players ran the SAME set of task_ids. This flags any player whose
  task-id set differs from the common intersection, so an asymmetric comparison
  (e.g. docpluck scored on n=54 vs liteparse on n=24) can't be published as a
  fair ranking.

* ``stale_input_records`` (added 2026-08-04, cycle 8) — a record is stale when its
  stored ``input_hash`` no longer matches the hash of the task the generator emits
  TODAY. Symmetry only asks WHICH task_ids a player ran; it cannot see that the
  task CONTENT changed underneath a kept task_id. In-place broadening does exactly
  that, so a score can silently describe text that no longer exists.

  This is not hypothetical: the check found 86 published records scored against
  vanished content — statcheck on stats-extraction-v1 (t-tier6-d1-1-s0 scored
  0.241 against pre-cycle-5 text vs 0.429 against the current text), 60 Claude
  records + 20 escimate records on the same arena, and oddpub/rtransparent on
  transparency-statements-v1 — all predating the cycle-4/5 broadenings and none
  visible to the symmetry or version gates.

All three are pure functions over already-loaded data so they are cheap to
unit-test; ``framework audit`` wires them to the registry + an arena's run records.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# (No REPO_ROOT constant: it was unused, and resolved into site-packages when
# installed. Audit functions take explicit paths — see framework/paths.py.)

# A run is OK to skip in the symmetry check when its score is an adapter error
# (the player attempted the task; a crash is surfaced separately by Finding 7).
# Symmetry is about *which tasks were attempted*, so errored attempts still count.

_VERSION_NUM_RE = re.compile(r"(\d+(?:\.\d+)+)")


@dataclass
class DriftRow:
    #: OK (resolved and matches declared) | DRIFT (resolved and does NOT match)
    #: | UNCHECKED (no comparison was possible -- see ``reason``).
    #:
    #: UNCHECKED exists because the three failure paths below used to `continue`,
    #: dropping the player from the report entirely. Measured 2026-08-26: 127 rows
    #: for 225 registered players, exit 0, "(all declared versions match)". An
    #: unchecked player is not a matching player, and must never print like one.
    player_id: str
    declared: str
    resolved: str | None
    drift: bool
    verdict: str = "OK"
    #: Why nothing could be compared. Set iff verdict == UNCHECKED. States what
    #: HAPPENED, and -- since 2026-08-27 -- carries the ADAPTER'S OWN account of
    #: why it has no version, never the audit's guess. The old comment here said
    #: the audit "cannot know that from outside", which was true and was the
    #: wrong conclusion: the fix was to ask the adapter, not to give up.
    reason: str | None = None
    #: One of ``framework.player_adapter.VERSION_ABSENT_*``, or None when a
    #: version resolved or the failure happened before the adapter existed.
    #: Measured 2026-08-27 across the registry: 64 by_design / 19 needs_prepare /
    #: 10 undeclared / 5 probe_failed -- previously all 98 printed one sentence.
    absence_kind: str | None = None

    @property
    def absence_is_problem(self) -> bool:
        """True when the version is absent because something is WRONG.

        Derived from ``VERSION_ABSENT_PROBLEM_KINDS`` rather than a literal here,
        so a kind added to the adapter contract cannot be silently un-reported by
        a second copy of the list drifting out of date -- the defect
        `rescore_stats_extraction.py` produced with the formula placeholder, and
        the reason that file's own comment says never to redeclare a constant.

        Being a PROBLEM is not the same as gating. See ``version_drift_gates``.
        """
        from framework.player_adapter import VERSION_ABSENT_PROBLEM_KINDS
        return self.absence_kind in VERSION_ABSENT_PROBLEM_KINDS


def _numeric(v: str | None) -> str | None:
    if not v:
        return None
    m = _VERSION_NUM_RE.search(v)
    return m.group(1) if m else None


def version_drift_from_registry(registry: list[dict]) -> list[DriftRow]:
    """Build each registered adapter, detect its installed version, and compare
    the numeric version against the declared ``player_version``.

    Only players whose adapter can detect a runtime version are reported (LLMs,
    HTTP platforms, etc. resolve ``None`` — their version IS the declared label).
    ``drift=True`` when the detected numeric version is absent from the declared
    string (e.g. declared ``docpluck-2.4.79`` vs installed ``docpluck-2.4.84``).
    """
    from framework.player_adapter import build_adapter

    rows: list[DriftRow] = []
    for entry in registry:
        pid = entry.get("player_id", "?")
        declared = entry.get("player_version", "")

        def _unchecked(reason: str) -> DriftRow:
            return DriftRow(player_id=pid, declared=declared, resolved=None,
                            drift=False, verdict="UNCHECKED", reason=reason)

        try:
            adapter = build_adapter(entry)
        except Exception as exc:
            rows.append(_unchecked(f"adapter could not be built: "
                                   f"{type(exc).__name__}: {exc}"))
            continue
        try:
            resolved = adapter.resolved_tool_version()
        except Exception as exc:
            rows.append(_unchecked(f"version probe raised "
                                   f"{type(exc).__name__}: {exc}"))
            continue
        if resolved is None:
            # ASK THE ADAPTER. This line used to read
            # `_unchecked("adapter reported no installed version")` for all 98
            # unresolvable players -- accurate, and useless, because it collapsed
            # four different facts into one sentence. Worst of them: five GROBID
            # rows were a DOWN SERVICE (`_tool_version.grobid_version` returns
            # None only on a raised request or a non-200 -- there is no "this
            # GROBID has no version" branch) rendering identically to the 64
            # CLI-driven players for which None is entirely correct.
            #
            # The reason comes from the adapter and never from a table here:
            # increment 2 tried a class-override heuristic and `pcurve` disproved
            # it -- pcurve.R loads only jsonlite, so it is pure base R by design.
            from framework.player_adapter import VERSION_ABSENT_UNDECLARED
            try:
                absence = adapter.version_absence_reason()
                kind, detail = absence.kind, absence.detail
            except Exception as exc:
                # An adapter that cannot even say why is `undeclared`, not a
                # silent pass. Never let this become the quiet default.
                kind = VERSION_ABSENT_UNDECLARED
                detail = (f"version_absence_reason() raised "
                          f"{type(exc).__name__}: {exc}")
            row = _unchecked(detail or "adapter reported no installed version")
            row.absence_kind = kind
            rows.append(row)
            continue

        rnum = _numeric(resolved)
        drift = bool(rnum) and (rnum not in declared)
        rows.append(DriftRow(player_id=pid, declared=declared, resolved=resolved,
                             drift=drift, verdict="DRIFT" if drift else "OK"))
    return rows


def version_drift_summary(rows: list[DriftRow]) -> list[str]:
    """Count the version audit by verdict, and say so honestly.

    The section used to end "(all declared versions match the installed tool)"
    whenever no row carried drift -- but a player that was never compared cannot
    contribute drift, so the sentence printed at its loudest exactly when the
    compare was most partial (127 of 225 players, measured 2026-08-26).

    An UNCHECKED player is not a matching player. The all-clear sentence is now
    earned only when every registered player was actually compared.
    """
    from collections import Counter

    from framework.player_adapter import VERSION_ABSENT_PROBLEM_KINDS

    n_ok = sum(1 for r in rows if r.verdict == "OK")
    n_drift = sum(1 for r in rows if r.verdict == "DRIFT")
    unchecked = [r for r in rows if r.verdict == "UNCHECKED"]
    n_unchecked = len(unchecked)
    out = [f"  -- {len(rows)} registered player(s): "
           f"{n_ok} OK, {n_drift} DRIFT, {n_unchecked} UNCHECKED"]
    if n_unchecked:
        # BREAK THE BUCKET DOWN. 98 players sharing one sentence is what this
        # whole section was for: "no version concept by design" and "the service
        # is down" are opposite facts and only one of them is fine.
        kinds = Counter(r.absence_kind or "unclassified" for r in unchecked)
        out.append("  UNCHECKED by reason: "
                   + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())))
        n_problem = sum(n for k, n in kinds.items() if k in VERSION_ABSENT_PROBLEM_KINDS)
        if n_problem:
            # Reported loudly, NOT gated -- see `version_drift_gates`.
            out.append(f"  PROBLEM: {n_problem} player(s) have a version that EXISTS "
                       f"and could not be read (an outage, not a design choice). "
                       f"Their scores are being ranked against a version nobody "
                       f"verified.")
        out.append(f"  NOTE: {n_unchecked} player(s) were not compared to anything. "
                   f"That is not evidence their declared version is correct.")
    elif not n_drift:
        out.append("  (all declared versions match the installed tool)")
    return out


def version_drift_gates(rows: list[DriftRow]) -> bool:
    """True iff the version section should fail the audit.

    ONLY REAL DRIFT GATES. A `probe_failed` player is reported loudly by
    ``version_drift_summary`` and deliberately does not fail the run.

    That is not leniency, it is the ASSET-ABSENT rule this module already applies
    to task sets: a GROBID server that is not running on a developer's machine is
    an ABSENT LOCAL ASSET, exactly like the gitignored PMC corpus -- nothing in
    the code is wrong and re-running changes nothing. Measured 2026-08-27: 5 of
    225 players are `probe_failed` on this machine right now, purely because the
    container is not up. An audit that fails on every machine without it is an
    audit nobody runs, and a gate people learn to ignore protects nothing.

    What is NOT covered here, and is a real gap worth naming rather than
    pretending away: a RUN RECORD produced while a probe was failing carries a
    tool version nobody verified. That is a records-level check, not a
    registry-level one, and it does not exist yet.
    """
    return any(r.drift for r in rows)


@dataclass
class SymmetryReport:
    arena_id: str
    ok: bool
    per_player: dict[str, int] = field(default_factory=dict)   # player -> n distinct task_ids
    intersection_size: int = 0
    union_size: int = 0
    offenders: list[str] = field(default_factory=list)         # players != intersection
    problems: list[str] = field(default_factory=list)

    def summary(self, *, gating: bool = True) -> str:
        """One-block report. Set ``gating=False`` for a scope that does NOT gate.

        A non-gating scope must not print the word FAIL. The pooled ``[all]``
        scope deliberately mixes revealed+private task_ids, so a player that
        legitimately runs both splits always looks asymmetric there — and the CLI
        prints it purely as a diagnostic while the run exits 0 with every gate
        passing. Printing "SYMMETRY FAIL" beside a passing gate is the exact
        confusion this project already fixed one level down for the per-split
        report on 2026-08-04 ("trains the reader to ignore SYMMETRY FAIL lines
        that DO matter") — and it is what a funder reading our audit output would
        screenshot. (Fable 5 cross-review, 2026-08-15.)
        """
        if self.ok:
            head = "SYMMETRY OK"
        elif gating:
            head = "SYMMETRY FAIL"
        else:
            head = "SYMMETRY UNEVEN (diagnostic — does not gate)"
        lines = [f"[{head}] {self.arena_id}"]
        lines.append(
            f"  players={len(self.per_player)} intersection={self.intersection_size} "
            f"union={self.union_size}"
        )
        for pid, n in sorted(self.per_player.items()):
            mark = "  <- asymmetric" if pid in self.offenders else ""
            lines.append(f"    {pid}: {n} tasks{mark}")
        for p in self.problems:
            lines.append(f"  PROBLEM: {p}")
        return "\n".join(lines)


def task_set_symmetry(records: list[dict], *, arena_id: str = "",
                      visibility: str | None = None) -> SymmetryReport:
    """Verify every player ran the same set of task_ids.

    ``visibility`` (e.g. ``"public"`` / ``"held_out"``) restricts the check to
    one split so revealed and private suites are audited independently. A player
    whose task-id set != the intersection of all players' sets is an offender;
    any offender makes the head-to-head unfair (``ok=False``).
    """
    by_player: dict[str, set[str]] = {}
    for r in records:
        if visibility is not None and r.get("task_visibility") != visibility:
            continue
        by_player.setdefault(r["player_id"], set()).add(r["task_id"])

    per_player = {pid: len(ts) for pid, ts in by_player.items()}
    if not by_player:
        return SymmetryReport(arena_id=arena_id, ok=True, per_player={},
                              problems=["no records for the requested visibility"])

    sets = list(by_player.values())
    intersection = set.intersection(*sets) if sets else set()
    union = set.union(*sets) if sets else set()
    offenders = [pid for pid, ts in by_player.items() if ts != intersection]

    problems: list[str] = []
    if offenders and len(by_player) > 1:
        problems.append(
            f"{len(offenders)} player(s) scored over a task set != the common "
            f"intersection of {len(intersection)} tasks: {sorted(offenders)}"
        )
    return SymmetryReport(
        arena_id=arena_id, ok=not problems, per_player=per_player,
        intersection_size=len(intersection), union_size=len(union),
        offenders=offenders, problems=problems,
    )


def per_split_symmetry_ok(records: list[dict], *, arena_id: str = "",
                          best_effort: set[str] | None = None) -> bool:
    """Return True iff every *per-split* symmetry check passes for one arena.

    Fairness is defined WITHIN a split: every player that ran a split ran the same
    set of task_ids there. This is the gate. The pooled ("all") scope is NOT part
    of the gate — pooling revealed+private task_ids makes a player that legitimately
    runs BOTH splits (deterministic tools: escimate on stats-extraction, the PDF
    parsers) look asymmetric versus revealed-only AI players, which is a false
    positive rather than an unfair head-to-head. The visibility-scoped checks below
    catch a genuinely-missing task within a split (see
    test_symmetry_respects_visibility_split).

    ``best_effort`` names players that are DOCUMENTED partial by nature — free-tier
    / rate-limited / slow LLM providers (e.g. the regcheck-groq / -deepseek /
    -openai players) whose coverage is bounded by a per-day token cap or wall-clock,
    not by unfairness. They ran what the tier allowed and are still ranked over the
    common intersection they DO cover (build:data's `partial` bucket), so a
    best-effort player short of the full task set does not fail the gate. A
    best-effort player that runs a strict SUPERSET is likewise fine. The gate still
    fails if a NON-best-effort player is asymmetric, or if the best-effort players
    are the only ones and disagree with each other in a way that leaves no common
    ground — but a genuine subset is tolerated by design.
    """
    best_effort = best_effort or set()
    for vis in ("public", "held_out"):
        # Compute symmetry over the NON-best-effort players only, so a throttled
        # best-effort provider's short coverage does not drag the intersection down
        # and make the full-coverage players look like the offenders. The gate is:
        # every non-best-effort player ran the same task set. Best-effort players
        # are then only required to be a SUBSET of that reference set (they ran what
        # the tier allowed — a subset is the throttle case; a disjoint/superset set
        # would mean they ran tasks the real players didn't, which is a real bug).
        primary = [r for r in records if r.get("player_id") not in best_effort]
        rep = task_set_symmetry(primary, arena_id=arena_id, visibility=vis)
        if rep.per_player and not rep.ok:
            return False
        # Reference task set = the intersection the real players agree on.
        by_player_ref: dict[str, set[str]] = {}
        for r in primary:
            if vis is not None and r.get("task_visibility") != vis:
                continue
            by_player_ref.setdefault(r["player_id"], set()).add(r["task_id"])
        reference = set.intersection(*by_player_ref.values()) if by_player_ref else set()

        # Each best-effort player must not have run tasks OUTSIDE the reference set
        # (a subset is fine — that's throttling; extra tasks would be a real fault).
        for pid in best_effort:
            ran = {r["task_id"] for r in records
                   if r.get("player_id") == pid
                   and (vis is None or r.get("task_visibility") == vis)}
            if ran and reference and not ran.issubset(reference):
                return False
    return True


def empty_run_files(arena_dir: Path, task_set_version: str = "v1") -> list[Path]:
    """Zero-byte ``runs/<v>/*.jsonl`` files — a killed run that wrote nothing.

    ``framework run`` creates the target file up front, so a run killed before its
    first task leaves a 0-byte file named after a real player. That is worse than
    no file: the build sees the player and can publish it with zero records, and
    the freshness/symmetry checks have nothing to compare. Two such leftovers
    survived the cycle-9 tournament kill (nemotron-nano-9b-siglang,
    gemma-4-31b-grim) and would have shipped as phantom players.
    """
    runs_dir = arena_dir / "runs" / task_set_version
    if not runs_dir.is_dir():
        return []
    return [p for p in sorted(runs_dir.glob("*.jsonl")) if p.stat().st_size == 0]


def orphaned_retry_temps(arena_dir: Path, task_set_version: str = "v1") -> list[tuple[Path, int]]:
    """Find ``*.retry-r*.jsonl`` temps holding OK verdicts NOT in their target file.

    ``framework retry-failed`` writes each round to a temp and folds it into the
    target at the end; a round killed mid-flight leaves the temp behind. It IS
    recovered on the next invocation — but only if someone runs one, and nothing
    surfaced their existence. Twice in cycle 8 an orphan sat unnoticed: one held a
    single already-merged record (harmless), the other held NINE unmerged OK
    verdicts that would have been silently lost work.

    Returns [(temp_path, n_unmerged_ok)] for temps with real unmerged content.
    Temps whose verdicts are all already in the target are ignored — they are
    inert leftovers, not lost work.
    """
    import json

    runs_dir = arena_dir / "runs" / task_set_version
    out: list[tuple[Path, int]] = []
    if not runs_dir.is_dir():
        return out
    for temp in sorted(runs_dir.glob("*.retry-r*.jsonl")):
        # "<target>.retry-rN.jsonl" -> "<target>.jsonl"
        target = runs_dir / (temp.name.split(".retry-r")[0] + ".jsonl")
        done: set[str] = set()
        if target.exists():
            for line in target.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line).get("task_id"))
        n_unmerged = 0
        for line in temp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("error") and rec.get("task_id") not in done:
                n_unmerged += 1
        if n_unmerged:
            out.append((temp, n_unmerged))
    return out


@dataclass
class StaleInputReport:
    arena_id: str
    ok: bool
    n_checked: int = 0
    #: Records whose task_id was not in `current_hashes`, so nothing could be said
    #: about them. Reported explicitly: `n_checked` alone made a run that checked
    #: 148 of 220 records look exactly like one that checked all 220.
    n_ignored: int = 0
    #: Held-out records whose `input_hash` is the redaction sentinel. These CANNOT
    #: be freshness-checked by construction — the hash was removed on purpose,
    #: because it was a membership oracle — so they are neither fresh nor stale.
    n_unverifiable: int = 0
    per_player: dict[str, int] = field(default_factory=dict)  # player -> n stale records
    problems: list[str] = field(default_factory=list)


def stale_input_records(records: list[dict], current_hashes: dict[str, str], *,
                        arena_id: str = "") -> StaleInputReport:
    """Flag records whose stored ``input_hash`` no longer matches today's task.

    ``current_hashes`` maps task_id -> the hash the generator produces NOW. A
    mismatch means the score describes task content that no longer exists: the
    task_id survived an in-place broaden but its text changed underneath, so the
    number is not comparable with a freshly-run player's.

    Records for task_ids absent from ``current_hashes`` are IGNORED, not flagged —
    they belong to another split (private/held-out) or another task-set version,
    and the caller decides what is in scope. Records with no ``input_hash`` are
    likewise skipped (older schema), so this never invents a failure.
    """
    from framework.holdout import REDACTED_INPUT_HASH

    per_player: dict[str, int] = {}
    n_checked = 0
    n_ignored = 0
    n_unverifiable = 0
    for r in records:
        tid, ih = r.get("task_id"), r.get("input_hash")
        # A held-out record's hash is deliberately destroyed by
        # `redact_held_out_record` (it was a membership oracle), so comparing it
        # can neither confirm nor refute freshness. Counting it as a MISMATCH
        # reported every real held-out arena as broken; counting it as a match
        # would be a false all-clear. It is a third thing, and it is named.
        if ih == REDACTED_INPUT_HASH:
            n_unverifiable += 1
            continue
        if not tid or not ih or tid not in current_hashes:
            n_ignored += 1
            continue
        n_checked += 1
        if current_hashes[tid] != ih:
            pid = r.get("player_id", "<unknown>")
            per_player[pid] = per_player.get(pid, 0) + 1
    problems = [
        f"{pid}: {n} record(s) scored against task content that has since changed "
        f"(stored input_hash != current) — re-run this player"
        for pid, n in sorted(per_player.items())
    ]
    return StaleInputReport(arena_id=arena_id, ok=not per_player, n_checked=n_checked,
                            n_ignored=n_ignored, n_unverifiable=n_unverifiable,
                            per_player=per_player, problems=problems)


#: The identity bucket for a record written before ``provenance.host_fingerprint``
#: existed (everything published up to 2026-08-21). It is deliberately NOT folded
#: into whichever machine appears alongside it: the machine is genuinely unknown,
#: and inventing a match would be a false all-clear of exactly the kind
#: ``stale_input_records`` learned to name rather than guess.
HOST_UNRECORDED = "<unrecorded>"


@dataclass
class HostSplitReport:
    arena_id: str
    ok: bool
    n_players: int
    n_hosts: int
    n_unverifiable: int
    per_player: dict[str, list[str]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    #: Players holding BOTH pre-fingerprint records and records from exactly one
    #: known machine. Reported, never gated — see `host_split_players`.
    mixed_with_unrecorded: list[str] = field(default_factory=list)
    #: Players whose records span more than one FINGERPRINT_VERSION. Digests
    #: either side of a recipe change are not comparable, so this says nothing
    #: about how many machines ran. Reported, never gated.
    fingerprint_version_skew: list[str] = field(default_factory=list)


def host_split_players(records: list[dict], *, arena_id: str = "") -> HostSplitReport:
    """Flag any ``player_id`` whose records come from more than one machine.

    Recording the host makes the machine visible; this makes it *enforced*. Both
    leaderboard engines group strictly on ``player_id`` and keep the FIRST
    ``player_version`` they see (``framework/leaderboard.py`` ``versions[pid]``,
    ``build-data.mjs`` ``playerVersions``), so laptop and desktop records sitting
    in one run file publish a single row: one machine's label over the mean of
    both machines' scores, with nothing failing.

    That matters here more than it would elsewhere. A local GGUF player's value
    is that it reproduces byte-for-byte — a property of one CPU and one llama.cpp
    backend, not of the weights. Averaging two machines silently converts the
    project's drift anchor into its own confound.

    Quiet by construction on today's corpus: every published record predates the
    fingerprint, so each player has exactly one bucket and the gate says nothing
    until someone actually mixes machines.
    """
    per_player: dict[str, set[str]] = {}
    # Digests are only comparable WITHIN one recipe version — that is what
    # FINGERPRINT_VERSION is for. Tracked per player so a recipe bump reads as a
    # recipe bump (Sonnet cross-review, 2026-08-21, reproduced: without this, the
    # first bump reports every single-machine player as split across two).
    versions_seen: dict[str, set[int]] = {}
    labels: dict[str, str] = {}
    n_unverifiable = 0
    for r in records:
        pid = r.get("player_id") or "<unknown>"
        block = (r.get("provenance") or {}).get("host_fingerprint") or {}
        fp = block.get("fingerprint")
        if not fp:
            fp = HOST_UNRECORDED
            n_unverifiable += 1
        else:
            labels.setdefault(fp, str(block.get("label") or fp))
            versions_seen.setdefault(pid, set()).add(block.get("fingerprint_version"))
        per_player.setdefault(pid, set()).add(fp)

    problems = []
    mixed_with_unrecorded = []
    version_skew = []
    for pid, fps in sorted(per_player.items()):
        if len(versions_seen.get(pid, set())) > 1:
            # Digests either side of a recipe change are not comparable, so this
            # says nothing about how many machines ran. Reported, not gated.
            version_skew.append(pid)
            continue
        known = fps - {HOST_UNRECORDED}
        # GATE ON KNOWN MACHINES ONLY (Codex cross-review, 2026-08-21).
        # Counting the unrecorded bucket as a machine failed the very first
        # re-run of any existing player on the SAME box, because its old records
        # predate the field. That is the common case, and a gate that fires on
        # the common case is one the operator learns to skip — the rot LESSONS
        # already records for "expected gate failure" lists. The mix is still
        # named below, just not gated.
        if len(known) < 2:
            if known and HOST_UNRECORDED in fps:
                mixed_with_unrecorded.append(pid)
            continue
        named = ", ".join(
            f"{fp} ({labels[fp]})" if fp in labels else fp for fp in sorted(known)
        )
        problems.append(
            f"{pid}: records come from {len(fps)} different machines [{named}] — the "
            "leaderboard groups on player_id alone and would publish one machine's "
            "label over the mean of both. Give each machine its own player_id, or "
            "move the superseded machine's records into an `_archive/` subdirectory "
            "(readdir is non-recursive, so archived runs stay in git and off the board)."
        )
    all_fps = {fp for fps in per_player.values() for fp in fps}
    return HostSplitReport(
        arena_id=arena_id,
        ok=not problems,
        n_players=len(per_player),
        n_hosts=len(all_fps - {HOST_UNRECORDED}),
        n_unverifiable=n_unverifiable,
        per_player={pid: sorted(fps) for pid, fps in per_player.items()},
        problems=problems,
        mixed_with_unrecorded=sorted(mixed_with_unrecorded),
        fingerprint_version_skew=sorted(version_skew),
    )


@dataclass
class FormulaProvenanceRow:
    """One placeholder kind found in a set of records, with how many carried it."""
    arena_id: str
    formula_id: str
    n_records: int
    detail: str
    gates: bool = True


#: The placeholders ``framework/runner.py::_resolve_formula_id`` stamps when it
#: cannot name the formula, mapped to what the reader should actually go and do.
#: Keyed on SUFFIX because the strings are ``f"{arena_id}-<suffix>"``.
#:
#: Each message names the RIGHT PLACE. That is the whole reason there are three
#: entries instead of one: telling the owner of pdf-text-fidelity-v1 that "this
#: arena never declared a FORMULA_ID" would be FALSE -- it declares two -- and
#: would hide the real cause, gold from two task sets in one record. A guard that
#: misdescribes what went wrong is the same defect class as one that does not
#: check, which this module learned the hard way when it guessed
#: "(missing held-out corpus?)" at a RuntimeError with no corpus involved.
_FORMULA_PLACEHOLDERS = {
    "-formula-id-hook-failed": (
        "the arena's `formula_id_for` HOOK raised or returned a non-string, so the "
        "formula was never identified. Fix the hook in the arena's scorer.py -- "
        "this is a code defect, not a configuration one."
    ),
    "-formula-not-applicable": (
        "no formula applied to this record: the GOLD did not resolve to a known "
        "task-set variant. Look at the record's ground truth, NOT at the scorer -- "
        "the usual cause is gold from two task sets mixed into one record, which "
        "this repo has produced before (pdf-text-fidelity-v1, 2026-08-21)."
    ),
    "-unspecified": (
        "the arena's scorer did not declare a FORMULA_ID. Since b820fa0 all 22 "
        "arenas declare one, so a record stamped this way was scored by a scorer "
        "module that failed to expose the constant."
    ),
}


def formula_provenance_problems(records: list[dict], *,
                                arena_id: str = "") -> list[FormulaProvenanceRow]:
    """Records whose `instrument.formula_id` records that there IS no formula id.

    ``framework/runner.py`` stamps three DISTINCT placeholders rather than one, so
    that "no formula applied" and "this arena never declared one" stay different
    facts. Keeping them distinct was correct and, on its own, bought nothing:
    **nothing read them.** An honest label nobody consumes is a silent fallback
    with better manners, and a plausible wrong value published while no test fails
    is this project's founding defect.

    A record with NO ``instrument`` block is NOT a problem. Measured 2026-08-27 by
    scanning every record directly rather than trusting the counter:
    ``records=8784, with an instrument block=0``. Absent is pre-feature, not bad,
    and treating it as bad would fail every arena in the repo -- which is the one
    way this check could turn into an outage.

    Rows are aggregated per placeholder kind, not per record: 250 identical lines
    would bury the report this is meant to make readable.
    """
    from collections import Counter

    seen: Counter = Counter()
    for r in records:
        inst = r.get("instrument")
        if not isinstance(inst, dict):
            continue          # pre-instrument record. Absent is not a placeholder.
        fid = inst.get("formula_id")
        if not isinstance(fid, str) or not fid.strip():
            # A declared field the code never populated. Distinct from a
            # placeholder: the placeholder at least says WHY.
            seen["<missing>"] += 1
            continue
        for suffix in _FORMULA_PLACEHOLDERS:
            if fid.endswith(suffix):
                seen[fid] += 1
                break

    rows: list[FormulaProvenanceRow] = []
    for fid, n in sorted(seen.items()):
        if fid == "<missing>":
            detail = ("the record carries an `instrument` block with NO `formula_id` "
                      "at all -- a declared field the code never populated, which is "
                      "worse than a placeholder because it does not even say why.")
        else:
            detail = next(d for s, d in _FORMULA_PLACEHOLDERS.items() if fid.endswith(s))
        rows.append(FormulaProvenanceRow(arena_id=arena_id, formula_id=fid,
                                         n_records=n, detail=detail, gates=True))
    return rows


@dataclass(frozen=True)
class SplitNote:
    """Why one split contributed nothing to a fingerprint.

    ``kind`` is the load-bearing field, and it exists because three very
    different facts used to arrive as the same empty dict:

    - ``"empty"``  the generator ran and this split simply has no tasks. Normal.
    - ``"asset"``  a LOCAL-ONLY asset is absent -- a gitignored corpus, a secret
                   seed. Normal on a fresh clone or a second machine; nothing in
                   the code is wrong and nothing will fix itself by re-running.
    - ``"failed"`` anything else. A guard that cannot run for an unexplained
                   reason, which is the only one of the three worth gating on.
    """
    split: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.split}: {self.detail}"


def blocking_kinds(notes: list["SplitNote"] | None) -> set[str]:
    """The kinds that mean a split DID NOT CONTRIBUTE when it should have.

    ``"empty"`` is excluded on purpose: an arena with no private split is
    complete without one.
    """
    return {n.kind for n in (notes or [])} & {"asset", "failed"}


def current_input_hashes(arena_dir: Path, task_set_version: str = "v1",
                         split: str | None = None, seed: int | None = None,
                         errors: list["SplitNote"] | None = None) -> dict[str, str]:
    """task_id -> input_hash for the tasks the arena's generator emits right now.

    With ``split=None`` (the default) this covers **every** split, resolving each
    one's seed from ``arena.yaml#benchmark_splits`` — including the private seed
    from the gitignored ``.private_seed``.

    It used to default to ``split="revealed", seed=0``, so on stats-extraction-v1 it
    built 36 hashes for 220 records and `stale_input_records` silently ignored all
    72 private ones. A stale HELD-OUT record — the official suite, the split whose
    integrity matters most — could never be reported, and the audit printed a clean
    `[FRESH]` either way. Found by a Codex review pass 2026-08-09.

    Returns ``{}`` when the arena cannot be generated in-process (e.g. PDF arenas
    whose generate() needs extra arguments), so callers degrade to "not checked"
    rather than to a false all-clear. A split whose secret seed is missing
    contributes nothing rather than falling back to a derivable one — the caller
    then sees those records as unchecked, which is the honest outcome.

    Pass a list as ``errors`` to be told WHY a split contributed nothing. Empty
    is not self-describing: "this arena cannot be generated in-process by design"
    and "the generator raised" both arrive as ``{}``, and only the second one
    means a guard has stopped working. Optional and defaulted, so every existing
    caller is unaffected.
    """
    if split is None:
        merged: dict[str, str] = {}
        for one in ("revealed", "private"):
            merged.update(
                _input_hashes_for_split(arena_dir, task_set_version, one, seed, errors))
        return merged
    return _input_hashes_for_split(arena_dir, task_set_version, split, seed, errors)


def _split_seed(arena_dir: Path, task_set_version: str,
                split: str) -> tuple[int | None, str, str]:
    """``(seed, kind, detail)``. ``kind`` is ``""`` when the seed resolved.

    The two ways this fails are NOT the same fact and must not be reported as
    one. A machine that simply does not hold a local-only secret is a normal
    machine; an arena.yaml that will not load is a broken arena. Returning a
    bare ``None`` for both is what let an absent secret read as a failure.
    """
    try:
        from framework.discovery import load_arena
        from framework.parity import resolve_seed

        manifest = load_arena(arena_dir)["manifest"]
        if not manifest.get("benchmark_splits"):
            return 0, "", ""
        seed, note = resolve_seed(manifest, arena_dir, task_set_version, split)
        # `note` means the dev fallback fired: the real secret is absent. Hashing
        # against a derivable seed would compare records to tasks nobody ran.
        if note:
            return None, "asset", (f"the {split} seed is a local-only secret this "
                                   f"machine does not hold ({note})")
        return seed, "", ""
    except Exception as e:
        return None, "failed", (f"resolving the {split} seed raised "
                                f"{type(e).__name__}: {e}")


def _input_hashes_for_split(arena_dir: Path, task_set_version: str,
                            split: str, seed: int | None,
                            errors: list["SplitNote"] | None = None) -> dict[str, str]:
    import importlib.util
    import sys

    from framework.holdout import HeldOutCorpusMissing
    from framework.runner import _hash_input

    import inspect

    def _note(kind: str, detail: str) -> dict[str, str]:
        # An empty result is reported by the caller either way; recording WHY is
        # what lets it tell a design limit apart from a failure. The audit used
        # to guess "(missing held-out corpus?)" at every cause, including causes
        # that had nothing to do with a corpus.
        if errors is not None:
            errors.append(SplitNote(split=split, kind=kind, detail=detail))
        return {}

    gen_path = arena_dir / "generator.py"
    if not gen_path.exists():
        return _note("failed", "no generator.py in the arena directory")
    if seed is None:
        seed, kind, detail = _split_seed(arena_dir, task_set_version, split)
        if seed is None:
            return _note(kind or "failed", detail or "the seed could not be resolved")
    mod_name = f"_audit_gen_{arena_dir.name.replace('-', '_')}_{split}"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, gen_path)
        gen = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = gen
        spec.loader.exec_module(gen)
        # Not every generator takes `split`: the PDF family and
        # replication-target-lookup emit both visibilities from one call and tag
        # each task themselves. Passing split= to those raises TypeError, which
        # used to skip the arena entirely — silently leaving the largest, most
        # real-paper-heavy arenas unchecked. Pass it only when it is accepted.
        params = inspect.signature(gen.generate).parameters
        kwargs = {"seed": seed}
        if "split" in params:
            kwargs["split"] = split
        tasks = gen.generate(task_set_version, **kwargs)
        # When the generator emits every split at once, keep only the one asked
        # for, so a revealed-scope check never compares against held-out tasks.
        from framework.holdout import HELD_OUT
        want_vis = "public" if split == "revealed" else HELD_OUT
        out: dict[str, str] = {}
        for t in tasks:
            vis = t.get("visibility")
            if "split" not in params and vis is not None and vis != want_vis:
                continue
            out[t["task_id"]] = _hash_input(t)
        if not out:
            # BENIGN, and deliberately its own kind. An arena with no private
            # split (replication-target-lookup-v1: 20 public tasks, no held-out
            # half) legitimately emits nothing here, and treating that as a
            # failure would gate 1 of 22 arenas forever.
            return _note("empty", "the generator ran but emitted no tasks for this split")
        return out
    except HeldOutCorpusMissing as e:
        # A GITIGNORED corpus is an environment fact, not a broken guard. Every
        # fresh clone lacks it, and there is nothing for the operator to fix in
        # the code. Named by EXCEPTION TYPE, which the corpus loader raises on
        # purpose (framework/holdout.py:359) -- not by matching on message text.
        return _note("asset", f"{type(e).__name__}: {e}")
    except Exception as e:
        return _note("failed", f"{type(e).__name__}: {e}")


def resolve_task_set_version(arena_dir: Path, explicit: str | None = None) -> str:
    """The task-set version this arena should be audited at.

    Every audit helper defaulted to the literal ``"v1"``, applied to every arena.
    That was correct only while no arena had a second task set. When
    stats-extraction-v1 moved to v2 (2026-08-09) and its v1 runs were archived, the
    freshness audit scanned an empty ``runs/v1/``, hit ``if not records: continue``,
    and the arena disappeared from the report — not FRESH, not STALE, not SKIP.
    252 unchecked records read as an all-clear.

    Resolution order: an explicit ``--task-set`` wins (an operator auditing history
    on purpose), else the entry marked ``status: current`` in
    ``arena.yaml#task_set_versions``, else — for an arena that declares no status
    at all — the newest entry. Delegates to
    ``framework.discovery.resolve_task_set_version`` and is mirrored by
    ``build-data.mjs::resolveTaskSetVersion``, so the audit and the published
    bundle can never disagree about which version is current.

    The status rule replaced a bare "newest entry" one on 2026-08-21. Position in
    the list is not meaning: an arena freezing v1 and publishing v2 needs BOTH to
    stay auditable, and a version can be declared before it is current.
    """
    if explicit:
        return explicit
    try:
        import yaml

        from framework.discovery import resolve_task_set_version as _resolve

        manifest = yaml.safe_load((arena_dir / "arena.yaml").read_text(encoding="utf-8")) or {}
        return _resolve(manifest)
    except Exception:
        return "v1"


def load_arena_records(arena_dir: Path, task_set_version: str = "v1") -> list[dict]:
    """Read every run record under ``runs/<task_set_version>/*.jsonl`` (top level
    only — ``_archive/`` is intentionally skipped)."""
    import json

    runs_dir = arena_dir / "runs" / task_set_version
    out: list[dict] = []
    if not runs_dir.is_dir():
        return out
    for p in sorted(runs_dir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --- frozen task-set drift ---------------------------------------------------


@dataclass
class FrozenDriftRow:
    arena_id: str
    version: str
    status: str
    expected: str | None      # the recorded generator_fingerprint
    actual: str | None        # what the generator produces right now
    n_tasks: int
    #: OK | DRIFT | UNFINGERPRINTED | UNCHECKABLE | GUARD-BROKEN | ASSET-ABSENT
    #: | PARTIAL-GUARD. Gating: DRIFT, GUARD-BROKEN, PARTIAL-GUARD.
    verdict: str
    reason: str = ""          # why a split contributed nothing; "" when all did


def _guarded_statuses() -> tuple[str, ...]:
    """Task-set statuses whose generator output is fingerprinted and drift-checked.

    DERIVED from ``FROZEN_STATUSES`` rather than duplicating its literals, so the
    two cannot silently disagree -- but deliberately NOT the same constant.

    ``discovery.is_frozen()`` shares ``FROZEN_STATUSES`` and the RUNNER refuses to
    run against a frozen task set. Adding "current" there would refuse every run
    in the repo. The audit needs a wider set than the runner does, so it owns one.

    Measured 2026-08-26, and the reason this exists: frozen/superseded task sets
    were 3 of 3 fingerprinted while CURRENT ones were 0 of 22. The drift guard
    covered exactly the task sets nothing runs against any more, and none of the
    22 that every published score is computed on.
    """
    from framework.discovery import FROZEN_STATUSES
    return (*FROZEN_STATUSES, "current")


def generator_fingerprint(arena_dir: Path, task_set_version: str,
                          errors: list["SplitNote"] | None = None) -> tuple[str | None, int]:
    """sha256 over the sorted ``task_id -> input_hash`` map, and the task count.

    Returns ``(None, 0)`` when the generator cannot be run in-process or emits
    nothing -- an arena whose held-out corpus is absent, for instance. That is
    never reported as OK: a fingerprint we could not compute is not a fingerprint
    that matched. What it IS reported as depends on whether one was ever recorded
    -- see ``frozen_drift``.

    Pass a list as ``errors`` to collect one line per split explaining why it
    contributed nothing.
    """
    import hashlib

    hashes = current_input_hashes(arena_dir, task_set_version, errors=errors)
    if not hashes:
        return None, 0
    payload = "\n".join(f"{tid}\t{h}" for tid, h in sorted(hashes.items()))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(hashes)


def frozen_drift(arena_dir: Path) -> list[FrozenDriftRow]:
    """Check every GUARDED task-set version of one arena against its fingerprint.

    A frozen version is a promise that the generator still produces exactly the
    tasks the published scores were computed on. Nothing else can catch a break:
    pinning a constant and merely REORDERING two ``rng`` calls each rewrote a
    frozen task set in this repo before, and only diffing the oldest gold caught
    it -- after 7 of 36, then 13 of 36, tasks had already stopped matching.
    """
    import yaml

    from framework.discovery import FROZEN_STATUSES, version_ladder

    try:
        manifest = yaml.safe_load((arena_dir / "arena.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    arena_id = manifest.get("arena_id", arena_dir.name)

    rows: list[FrozenDriftRow] = []
    for entry in version_ladder(manifest):
        if entry.get("status") not in _guarded_statuses():
            continue
        version = entry["version"]
        expected = entry.get("generator_fingerprint")
        errors: list[SplitNote] = []
        actual, n = generator_fingerprint(arena_dir, version, errors=errors)
        blocked = blocking_kinds(errors)
        if actual is not None and expected and expected == actual and blocked:
            # THE RUBBER STAMP, found by two of three consult seats 2026-08-26 and
            # reproduced: 36 tasks with .private_seed, 18 without. A fingerprint
            # BASELINED while a split was failing records the half that worked;
            # every later audit on the same machine recomputes the same half and
            # prints "[OK] 18 tasks reproduce their fingerprint". The held-out
            # half -- the split whose integrity matters most -- is then guarded by
            # nothing, permanently, and nothing says so.
            #
            # A match is only an all-clear if the thing that matched covered the
            # whole task set.
            verdict = "PARTIAL-GUARD"
        elif actual is None and expected and "failed" not in blocked and blocked:
            # A GITIGNORED corpus or a local-only secret this machine does not
            # hold. Every fresh clone is in this state; there is nothing wrong
            # with the code and re-running will never fix it. Reported, never
            # gated -- MEASURED 2026-08-26: gating it turned a clean checkout of
            # pdf-citation-matching-v1 into a hard /ship failure.
            verdict = "ASSET-ABSENT"
        elif actual is None and expected:
            # THE GUARD ITSELF HAS STOPPED WORKING, and that is not the same fact
            # as "this was never guarded". A recorded fingerprint EXISTS only
            # because `--write-fingerprints` computed one successfully at least
            # once, so failing to recompute it is a regression in the generator,
            # the environment or the secrets -- for a task set whose published
            # scores depend on exactly this promise.
            #
            # It also closes a silent downgrade, MEASURED 2026-08-26 at the real
            # call site: the same planted wrong fingerprint reported [DRIFT] +
            # exit 1 with a working generator and [UNCHECKABLE] + exit 0 with one
            # that raised. Testing `actual is None` before comparing `expected`
            # meant any generator failure turned a real drift into a clean exit.
            verdict = "GUARD-BROKEN"
        elif actual is None:
            # Never guarded and still cannot be. Unchanged, and deliberately
            # non-gating: staying unguarded is not news, and making it fail would
            # break other sessions' /ship for a fact that was already true.
            verdict = "UNCHECKABLE"
        elif not expected:
            verdict = "UNFINGERPRINTED"
        elif expected == actual:
            verdict = "OK"
        else:
            verdict = "DRIFT"
        rows.append(FrozenDriftRow(
            arena_id=arena_id, version=version, status=entry["status"],
            expected=expected, actual=actual, n_tasks=n, verdict=verdict,
            reason="; ".join(str(e) for e in errors),
        ))
    return rows


def _insert_fingerprint_line(lines: list[str], version: str, fp: str) -> bool:
    """Splice `generator_fingerprint` into the `- version: <v>` entry, in place.

    Handles BOTH YAML styles this repo uses: block

        - version: v1
          n_tasks: 36

    and flow, which is what most arena.yaml files actually contain

        - {version: v1, n_tasks: 36, released_on: "2026-04-29"}

    Returns True only if a line was actually changed. The first version of this
    function silently matched neither flow entry, so the caller printed `[WROTE]`
    for a write that never happened -- a success message for work not done, which
    is the exact defect class the portfolio rule forbids. The read-back assertion
    in the caller is the guard that would have caught it.
    """
    import re

    flow = re.compile(r"^(\s*)-\s*\{\s*version:\s*['\"]?" + re.escape(version) + r"['\"]?\s*,")
    block = re.compile(r"^(\s*)-\s*version:\s*['\"]?" + re.escape(version) + r"['\"]?\s*$")

    for i, line in enumerate(lines):
        m = flow.match(line)
        if m:
            # Insert the key inside the flow mapping, right after `version:`.
            head, sep, tail = line.partition(",")
            lines[i] = f'{head}, generator_fingerprint: "{fp}"{sep}{tail}'
            return True
        m = block.match(line)
        if m:
            indent = m.group(1) + "  "
            lines.insert(i + 1, f'{indent}generator_fingerprint: "{fp}"\n')
            return True
    return False


@dataclass
class FingerprintWriteResult:
    """What `--write-fingerprints` actually did, INCLUDING what it declined to do.

    A refusal used to be a bare `continue`, so the CLI printed nothing and the
    operator concluded the arena had needed nothing. Silence and success must
    not look the same -- that is the defect class this whole module exists to
    stop, applied to the writer itself.
    """
    written: list[tuple[str, str]] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)


def write_generator_fingerprints(arena_dir: Path) -> FingerprintWriteResult:
    """Record the CURRENT generator output as the fingerprint of each frozen
    version that lacks one. Returns the (version, fingerprint) pairs written.

    Deliberately refuses to overwrite an existing fingerprint: that would turn
    the drift gate into a rubber stamp, which is exactly how a guard becomes
    decoration. To re-baseline on purpose, delete the line by hand and say why
    in the arena CHANGELOG.

    Raises if the edit did not survive a re-parse. A writer that reports success
    for a no-op is worse than one that fails, because the operator then believes
    a frozen version is guarded when it is not.
    """
    import yaml

    from framework.discovery import FROZEN_STATUSES, version_ladder

    path = arena_dir / "arena.yaml"
    text = path.read_text(encoding="utf-8")
    manifest = yaml.safe_load(text) or {}

    pending: list[tuple[str, str]] = []
    refused: list[tuple[str, str]] = []
    for entry in version_ladder(manifest):
        if entry.get("status") not in _guarded_statuses() or entry.get("generator_fingerprint"):
            continue
        notes: list[SplitNote] = []
        fp, _n = generator_fingerprint(arena_dir, entry["version"], errors=notes)
        if fp is None:
            continue
        blocked = blocking_kinds(notes)
        if blocked:
            # REFUSE. Rejecting only `fp is None` meant this baselined whatever
            # the splits that HAPPENED TO WORK produced. Measured 2026-08-26 on
            # zcurve-evidential-v1: without .private_seed the writer recorded an
            # 18-task revealed-only hash for a 36-task set, and the next audit
            # called it OK. That is precisely the rubber stamp this function's
            # own docstring promises it will not create -- the refusal to
            # OVERWRITE was never enough, because the first write is where the
            # damage happens.
            refused.append((entry["version"],
                            "; ".join(str(n) for n in notes if n.kind in blocked)))
            continue
        pending.append((entry["version"], fp))

    if not pending:
        return FingerprintWriteResult(refused=refused)

    # Edit the YAML as text so comments and formatting survive.
    lines = text.splitlines(keepends=True)
    written: list[tuple[str, str]] = []
    for version, fp in pending:
        if _insert_fingerprint_line(lines, version, fp):
            written.append((version, fp))
        else:
            raise RuntimeError(
                f"{path}: could not locate the task_set_versions entry for {version}; "
                f"add generator_fingerprint: \"{fp}\" by hand")

    new_text = "".join(lines)
    # READ BACK. The whole point of this function is that a frozen version ends up
    # guarded; asserting the edit parsed and landed is the only thing that proves it.
    reparsed = yaml.safe_load(new_text) or {}
    got = {v["version"]: v.get("generator_fingerprint")
           for v in (reparsed.get("task_set_versions") or [])}
    for version, fp in written:
        if got.get(version) != fp:
            raise RuntimeError(
                f"{path}: fingerprint for {version} did not survive the edit "
                f"(re-parsed as {got.get(version)!r}) — arena.yaml NOT modified")

    path.write_text(new_text, encoding="utf-8")
    return FingerprintWriteResult(written=written, refused=refused)

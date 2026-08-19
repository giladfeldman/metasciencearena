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
    player_id: str
    declared: str
    resolved: str | None
    drift: bool


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
        try:
            adapter = build_adapter(entry)
        except Exception:
            continue
        try:
            resolved = adapter.resolved_tool_version()
        except Exception:
            resolved = None
        if resolved is None:
            continue
        declared = entry.get("player_version", "")
        rnum = _numeric(resolved)
        drift = bool(rnum) and (rnum not in declared)
        rows.append(DriftRow(player_id=entry["player_id"], declared=declared,
                             resolved=resolved, drift=drift))
    return rows


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


def current_input_hashes(arena_dir: Path, task_set_version: str = "v1",
                         split: str | None = None, seed: int | None = None) -> dict[str, str]:
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
    """
    if split is None:
        merged: dict[str, str] = {}
        for one in ("revealed", "private"):
            merged.update(_input_hashes_for_split(arena_dir, task_set_version, one, seed))
        return merged
    return _input_hashes_for_split(arena_dir, task_set_version, split, seed)


def _split_seed(arena_dir: Path, task_set_version: str, split: str) -> int | None:
    """The seed for one split, or None when it cannot be resolved honestly."""
    try:
        from framework.discovery import load_arena
        from framework.parity import resolve_seed

        manifest = load_arena(arena_dir)["manifest"]
        if not manifest.get("benchmark_splits"):
            return 0
        seed, note = resolve_seed(manifest, arena_dir, task_set_version, split)
        # `note` means the dev fallback fired: the real secret is absent. Hashing
        # against a derivable seed would compare records to tasks nobody ran.
        return None if note else seed
    except Exception:
        return None


def _input_hashes_for_split(arena_dir: Path, task_set_version: str,
                            split: str, seed: int | None) -> dict[str, str]:
    import importlib.util
    import sys

    from framework.runner import _hash_input

    import inspect

    gen_path = arena_dir / "generator.py"
    if not gen_path.exists():
        return {}
    if seed is None:
        seed = _split_seed(arena_dir, task_set_version, split)
        if seed is None:
            return {}
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
        return out
    except Exception:
        return {}


def resolve_task_set_version(arena_dir: Path, explicit: str | None = None) -> str:
    """The task-set version this arena should be audited at.

    Every audit helper defaulted to the literal ``"v1"``, applied to every arena.
    That was correct only while no arena had a second task set. When
    stats-extraction-v1 moved to v2 (2026-08-09) and its v1 runs were archived, the
    freshness audit scanned an empty ``runs/v1/``, hit ``if not records: continue``,
    and the arena disappeared from the report — not FRESH, not STALE, not SKIP.
    252 unchecked records read as an all-clear.

    Resolution order: an explicit ``--task-set`` wins (an operator auditing history
    on purpose), else the newest entry in ``arena.yaml#task_set_versions``, else
    ``"v1"``. Mirrors ``build-data.mjs::resolveTaskSetVersion`` so the audit and the
    published bundle can never disagree about which version is current.
    """
    if explicit:
        return explicit
    try:
        import yaml

        manifest = yaml.safe_load((arena_dir / "arena.yaml").read_text(encoding="utf-8")) or {}
        versions = manifest.get("task_set_versions") or []
        if versions:
            newest = versions[-1]
            version = newest.get("version") if isinstance(newest, dict) else newest
            if version:
                return str(version)
    except Exception:
        pass
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

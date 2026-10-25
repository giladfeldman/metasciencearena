"""Tool-feedback report generator.

Reads run-record JSONL files for one (arena, task_set_version) and produces a
report bundle for one focal player:

    runs/<arena>/<task_set>/reports/<player_id>@<player_version>/
      tool_report.json
      tool_report.md
      findings/public/<task_id>.json     (full drilldown — input_hash + gold)
      findings/held_out/<task_id>.json   (categorical only — no content)
      reproduce/<task_id>.json           (public only — verbatim envelope)
      README.md                          (how to read this report)

Design spec: docs/superpowers/specs/2026-05-06-arena-feedback-reports-design.md
Decisions Log there is authoritative for the redaction posture.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from packaging.version import InvalidVersion, Version

from framework import adherence
from framework import pricing
from framework.discovery import load_arena
from framework.storage import read_records
from framework import hermetic
from framework import timeouts


_SEMVER_PATTERN = re.compile(r"\d+(?:\.\d+){0,3}")


def _version_sort_key(version: str) -> tuple[int, Any, str]:
    """Order player versions semver-aware.

    The first embedded `\\d+(?:\\.\\d+)*` substring is parsed as a semver
    `Version`; the remainder of the string breaks ties. Strings with no
    embedded version fall back to lexicographic sort but rank below all
    semver-bearing versions, so a one-off `"experimental"` doesn't mask
    `1.10.0`.
    """
    m = _SEMVER_PATTERN.search(version)
    if m:
        try:
            return (1, Version(m.group(0)), version)
        except InvalidVersion:
            pass
    return (0, Version("0"), version)


# ---------- I/O helpers ---------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_records_for(arena_root: Path, task_set_version: str) -> list[dict]:
    """Read every JSONL run-record file under arenas/<arena>/runs/<task_set>/.

    Skips the reports/ subdir (which would otherwise loop on regeneration).
    """
    runs_dir = arena_root / "runs" / task_set_version
    if not runs_dir.exists():
        return []
    out: list[dict] = []
    for jsonl in sorted(runs_dir.glob("*.jsonl")):
        out.extend(read_records(jsonl))
    return out


# ---------- record bucketing ---------------------------------------------


def _is_errored(rec: dict) -> bool:
    bd = rec.get("score", {}).get("breakdown") or {}
    return isinstance(bd, dict) and bool(bd.get("error"))


def _is_excluded(rec: dict) -> bool:
    return rec.get("score", {}).get("primary") is None and not _is_errored(rec)


def _is_scored(rec: dict) -> bool:
    return not _is_errored(rec) and rec.get("score", {}).get("primary") is not None


def _player_key(rec: dict) -> tuple[str, str]:
    return (rec["player_id"], rec.get("player_version", ""))


# ---------- aggregations -------------------------------------------------


def _mean_ci(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    sd = statistics.stdev(values)
    return mean, 1.96 * sd / math.sqrt(n)


def collapse_trials(records: list[dict]) -> list[float]:
    """One observation per TASK: the mean primary across that task's trials.

    `framework/runner.py` writes one record per (task, trial), and `--trials`
    defaults to 3 for any player declared `deterministic: false`. Feeding those
    raw records to `_mean_ci` treats three trials of one task as three
    independent observations: for a deterministic tool the trials are identical,
    so `sd` is unchanged while `n` triples and the interval comes out sqrt(3)
    too NARROW. See framework/tests/test_trial_collapse_ci.py.

    The per-task mean is the observation; the interval describes the spread
    BETWEEN tasks, which is the quantity a reader compares across players.

    Records with no task_id fall back to their own identity so nothing is
    silently merged — an unkeyed record must not collapse into an unrelated one.
    """
    by_task: dict[object, list[float]] = defaultdict(list)
    for i, r in enumerate(records):
        key = r.get("task_id") or ("__no_task__", i)
        by_task[key].append(float(r["score"]["primary"]))
    return [sum(v) / len(v) for v in by_task.values()]


#: Marks a 0.0 this policy assigned, as opposed to a 0.0 the scorer computed.
#: Without it the two are indistinguishable in a record, and a reader cannot tell
#: a model that answered wrongly from one that produced nothing usable.
UNSCOREABLE_ZERO_FLAG = "scored_zero_by_policy"


def apply_unscoreable_policy(all_records: list[dict]) -> tuple[list[dict], set[str]]:
    """Decide what an unscoreable output is worth, using the rest of the field.

    Owner's decision, 2026-08-19 (TODO A5(b)). An output the arena's schema
    rejects — unparseable, or missing a required field — was previously dropped
    from the mean entirely, so a model that could not follow the output contract
    was *excused* rather than penalised, and its published score described only
    the tasks it happened to format correctly.

    A blanket zero is the obvious fix and it is wrong, for a reason visible in
    the data: on ``replication-target-lookup-v1`` **all five** players fail the
    same four tasks. Zeroing there would publish "every model is bad at this"
    when the honest reading is that those four tasks are broken. The rule that
    survives both cases uses the field as its control:

      * **at least one other player scored the task** -> the failure is the
        player's. Score it **0.0** and count it in the mean.
      * **nobody scored the task** -> the task is the suspect, not the players.
        Leave every record excluded and return the task id so the report can
        say so out loud.

    Returns ``(records, tasks_no_player_could_score)``. Input records are never
    mutated: they are read from disk and shared across engines, and a mutation
    here would silently change what a later caller sees.

    Measured impact when introduced: 9 records become 0.0 across two arenas
    (power-reporting-v1, prereg-deviation-v1), moving four players' means by
    -0.019 to -0.126; 4 tasks in replication-target-lookup-v1 are flagged rather
    than zeroed.

    A TRANSPORT TIMEOUT IS NEVER SCORED (owner's decision 2026-10-07, d-545c62;
    `framework/timeouts.py`). It measures the wall we set, not the player, so it
    stays an errored record — excluded from the mean, counted in n_errored — and
    `qa_coverage_checks.py` C11 refuses to land a run file that contains one. It
    also does not make a task an orphan: "nobody could score this task" is a
    claim about the TASK, and our clock says nothing about the task. Measured when
    introduced: 0 timeout records across 270 live run files, so no published
    number moved.

    A TRANSPORT ERROR (HTTP 429 / 5xx / connection) follows the same rule
    (d-685d09, 2026-10-19; `timeouts.is_transport_error`): it is the provider's
    failure, not the player's answer. Measured when introduced: 0 live records
    matched, so no published number moved; the two runs that would have been
    zeroed (gpt-oss-120b, Flash-Lite on citation-claim-fidelity-v1) had already been
    archived off the board for that reason.
    """
    scored_tasks: set[str] = {
        r.get("task_id") for r in all_records if _is_scored(r) and r.get("task_id")
    }
    unscoreable_tasks: set[str] = {
        r.get("task_id") for r in all_records
        if r.get("task_id") and not _is_scored(r)
        and not timeouts.record_is_transport_failure(r)
    }
    orphans = unscoreable_tasks - scored_tasks

    out: list[dict] = []
    for r in all_records:
        task = r.get("task_id")
        if (_is_scored(r) or not task or task in orphans
                or timeouts.record_is_transport_failure(r)):
            out.append(r)
            continue
        # Copy deeply enough that the caller's record is untouched.
        rec = dict(r)
        score = dict(rec.get("score") or {})
        breakdown = dict(score.get("breakdown") or {})
        # The reason is RENAMED rather than kept under `error`, because
        # `_is_errored` keys on that field: leaving it would make the record
        # both "scored 0.0" and "errored", so it would be counted in n_errored
        # AND dropped from the mean — i.e. the policy would silently do nothing.
        # The flag plus the preserved reason is what lets a reader tell a
        # policy-assigned 0.0 from a 0.0 the scorer computed.
        reason = breakdown.pop("error", None)
        if reason is not None:
            breakdown["unscoreable_reason"] = reason
        breakdown[UNSCOREABLE_ZERO_FLAG] = True
        score["breakdown"] = breakdown
        score["primary"] = 0.0
        rec["score"] = score
        out.append(rec)
    return out, orphans


def _arena_enum_vocabulary(arena: dict, task_set_version: str | None = None) -> frozenset[str]:
    """Every value the arena's OUTPUT schema declares legal in any enum.

    Feeds `adherence.classify_message`, which uses it to tell an `enum_swap` (a
    value that is legal somewhere in this schema, offered in the wrong field —
    `not_a_stat`, a VERDICT, put into `kind`) from an `enum_value` (a value the
    schema never declares at all).

    Best-effort and silent on failure: an arena with no output schema, or an
    unreadable one, yields an empty vocabulary, and an empty vocabulary
    classifies every rejected enum value as the WEAKER `enum_value`. That is the
    conservative direction — `enum_swap` is the specific claim the vocabulary
    hypothesis turns on and must never be assumed into existence by a missing
    file. Report generation must not fail over a diagnostic label.
    """
    if not arena.get("output_schema_path"):
        return frozenset()
    from framework.discovery import output_schema_path_for
    path = output_schema_path_for(arena, task_set_version)
    try:
        return frozenset(adherence.enum_values_in_schema(
            json.loads(Path(path).read_text(encoding="utf-8"))))
    except (OSError, ValueError):
        return frozenset()


def _is_policy_zero(rec: dict) -> bool:
    bd = (rec.get("score") or {}).get("breakdown") or {}
    return isinstance(bd, dict) and bool(bd.get(UNSCOREABLE_ZERO_FLAG))


def _summary(focal_records: list[dict], all_records: list[dict], focal_key: tuple[str, str],
             *, enum_values: frozenset[str] | set[str] = frozenset(),
             unattributable_tasks: frozenset[str] | set[str] = frozenset()) -> dict:
    """Compute the focal player's summary block, including rank vs all players."""
    scored = [r for r in focal_records if _is_scored(r)]
    # An error on a task NO player could score is the TASK's failure, not this
    # player's. `apply_unscoreable_policy` already refuses to zero those records
    # for exactly that reason, and it leaves `breakdown.error` in place (it
    # renames it only on the records it zeroes), so this predicate used to pick
    # them back up. `structural_adherence` was taught to excuse them on
    # 2026-09-05 and this count was not, so one published report asserted both
    # things about the same record: n_errored 1 beside adherence 1.0 with
    # n_unattributed 4. A record with no task_id is part of no field and stays
    # chargeable — that is the only route left by which this can be non-zero.
    #
    # EXCUSING RE-ATTRIBUTES, it does not drop: the record moves to `excluded`
    # ("reported, never in the mean"), so the three buckets still account for
    # every task the player attempted. Without that it fell out of all three —
    # `_is_excluded` wants `primary is None` and no error, and an orphan record
    # carries `primary 0.0` with the error intact — and the live report read
    # 16 + 3 + 0 for a player with 20 tasks.
    def _excused(rec: dict) -> bool:
        return _is_errored(rec) and rec.get("task_id") in unattributable_tasks

    excluded = [r for r in focal_records if _is_excluded(r) or _excused(r)]
    errored = [r for r in focal_records if _is_errored(r) and not _excused(r)]
    # Collapse repeat trials to one observation per task BEFORE the interval:
    # three trials of one task are not three independent measurements.
    primary_values = collapse_trials(scored)
    primary_mean, primary_ci = _mean_ci(primary_values)

    # Cost is DERIVED from recorded tokens, never read off the record: `cost_usd`
    # is deliberately not written by the runner (a dollar figure is a claim about
    # a price list that changes; a token count is a measured fact that stays
    # true). Reading `r["cost_usd"]` here — as this did until 2026-08-14 — meant
    # consuming a field the producer never writes, so every published report
    # carried cost_usd_mean: null while pricing.py sat uncalled.
    usage_summary = pricing.summarise(scored)
    # record_cost_usd, not cost_usd: it applies the same three-field subscription
    # check the Node build applies, so the two engines cannot disagree about
    # whether a Claude-via-CLI run is priced.
    cost_values = [
        c for c in (pricing.record_cost_usd(r) for r in scored) if c is not None
    ]
    latency_values = [float(r["latency_ms"]) for r in scored if r.get("latency_ms") is not None]

    # RANK BY MEAN — deliberately NOT the arena's published rank, and named so.
    #
    # This is a plain ordinal: sort every player by mean, take the position. The
    # PUBLISHED rank is not that. `framework/ranking.py::rank_players` runs a
    # paired sign-flip test with Holm correction, so a player only ranks below
    # someone who beat it by more than the data's noise, and players the data
    # cannot separate SHARE a rank. The two answers routinely differ and neither
    # is wrong: measured 2026-09-12 on `power-reporting-v1`,
    # `nvidia-nemotron-lightning-power` is 13/13 by mean and 6th by the tested
    # ranking, out of the same 13 competitors.
    #
    # `scripts/lib/report.mjs` hit exactly this and fixed it by consuming the
    # published ranking instead (see `publishedRank` there); this engine cannot,
    # because the tested ranking needs the ranked/common buckets that
    # `fairness-gates.mjs::computeBuckets` derives during the JS build, and
    # re-deriving them here would be the second implementation that
    # `scripts/compute_ranking.py` exists to prevent. So the field is renamed
    # rather than faked: a number called `rank` that disagrees with the site's
    # rank is the defect, and `rank_by_mean` is not that number lying.
    #
    # This report bundle is gitignored and published nowhere
    # (`.gitignore:66`); it is a local artifact for whoever runs the CLI.
    # Pinned by leaderboard-app/scripts/tests/report-engine-parity.test.mjs.
    by_player_records: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_records:
        if _is_scored(r):
            by_player_records[_player_key(r)].append(r)
    # Same collapse for ranking, so a player cannot move up or down purely by
    # having been run with more trials than its competitors.
    by_player = {k: collapse_trials(v) for k, v in by_player_records.items()}
    aggregated = [(k, sum(v) / len(v)) for k, v in by_player.items() if v]
    aggregated.sort(key=lambda kv: kv[1], reverse=True)
    rank_by_mean = next((i + 1 for i, (k, _) in enumerate(aggregated) if k == focal_key), None)

    return {
        "primary_mean": primary_mean,
        "primary_ci_half": primary_ci,
        # TASKS measured, not records written — a reader compares this as a
        # sample size, and a 3-trial player must not advertise 3x the sample.
        "n_scored": len(primary_values),
        "n_excluded": len(excluded),
        "n_errored": len(errored),
        "rank_by_mean": rank_by_mean,
        "n_competitors": len({k for k, _ in aggregated}),
        "cost_usd_mean": (sum(cost_values) / len(cost_values)) if cost_values else None,
        "latency_ms_mean": (sum(latency_values) / len(latency_values)) if latency_values else None,
        # Tokens are reported even when cost is not: an unpriced or subscription
        # player still consumed measurable resources, and "no price" must not
        # read as "no usage". n_with_usage/n_priced say how much of the cost
        # figure is actually backed by a known price rather than quietly dropped.
        "tokens_total": usage_summary["total_tokens"],
        "tokens_prompt": usage_summary["prompt_tokens"],
        "tokens_completion": usage_summary["completion_tokens"],
        "n_with_usage": usage_summary["n_with_usage"],
        # Why usage is absent, when it is. Without it "not recorded" and
        # "consumed nothing" are indistinguishable to a report's reader — and
        # the Node engine has always published it, so omitting it here made the
        # two engines emit different shapes under the same field names.
        "usage_absent_reason": usage_summary["usage_absent_reason"],
        "n_priced": usage_summary["n_priced"],
        "price_table_checked_on": usage_summary["price_table_checked_on"],
        "price_table_stale": usage_summary["price_table_stale"],
        # HOW THIS PLAYER WAS CONTAINED (CC3 decision 2026-08-19: keep and
        # LABEL, do not re-run). A CLI record written before
        # framework/hermetic.py existed carries no containment block, and is
        # reported `uncontrolled` — the player ran inside this repo with its
        # full tool set and could read the answer key. `worst`, not the majority
        # state: a player with mixed records is not a hermetic player, and
        # reporting the majority would launder the uncontrolled half.
        # FOCAL records, not the arena's: this field describes THIS player, and
        # summarising `all_records` let one CLI competitor label every R tool and
        # HTTP player in the same arena `uncontrolled` (123 of 130 published
        # reports said so, against the manifests' correct 58).
        "containment": hermetic.summarise_containment(focal_records),
        # How many of this player's scores are 0.0 because it produced nothing
        # scoreable, rather than because the scorer computed a zero. Published so
        # a reader can tell "answered badly" from "could not follow the output
        # contract" — those are different failures and a bare mean hides it.
        "n_scored_zero_by_policy": sum(1 for r in focal_records if _is_policy_zero(r)),
        # STRUCTURAL ADHERENCE — can this player fill in the form, as opposed to
        # answer it. A NEW field beside the old, never a redefinition: nothing
        # here touches `primary_mean`, and every published number is unchanged by
        # its presence.
        #
        # `n_scored_zero_by_policy` above already counts the zeros the policy
        # assigned, but it counts every CAUSE together — a timeout and a
        # misplaced enum value look identical in it — and it is a raw count, so
        # it cannot be compared between players with different task loads. This
        # is the rate, with the cause named. See framework/adherence.py.
        #
        # FOCAL records, and post-policy: `apply_unscoreable_policy` has already
        # renamed `breakdown.error` to `breakdown.unscoreable_reason` by the time
        # this runs, which is exactly the seam `adherence.violation_reason`
        # exists to survive.
        "structural_adherence": adherence.summarise(
            focal_records, enum_values=enum_values,
            unattributable_tasks=unattributable_tasks),
    }


def _error_histogram(focal_records: list[dict], category_severity: dict[str, str]) -> list[dict]:
    """Per-category counts across the focal player's tasks.

    Counts both at the finding level (n_findings) and at the task level
    (n_tasks_affected) so tool authors can tell "many small mistakes on a few
    tasks" from "one mistake spread across many tasks."
    """
    n_findings: Counter = Counter()
    tasks_affected: dict[str, set] = defaultdict(set)
    total_findings = 0

    for r in focal_records:
        findings = (r.get("score") or {}).get("findings") or []
        seen_categories_this_task = set()
        for f in findings:
            cat = f.get("category")
            if not cat:
                continue
            count = int(f.get("count", 1))
            n_findings[cat] += count
            total_findings += count
            seen_categories_this_task.add(cat)
        for cat in seen_categories_this_task:
            tasks_affected[cat].add(r["task_id"])

    histogram = []
    for cat, n in sorted(n_findings.items(), key=lambda kv: kv[1], reverse=True):
        histogram.append({
            "category": cat,
            "n_findings": n,
            "n_tasks_affected": len(tasks_affected[cat]),
            "share_of_findings": (n / total_findings) if total_findings else 0.0,
            "severity": category_severity.get(cat, "unknown"),
        })
    return histogram


def _difficulty_breakdown(focal_records: list[dict], envelopes_by_task: dict[str, dict] | None) -> dict:
    """For each declared difficulty axis, primary_mean grouped by axis value.

    We pull the difficulty map from the envelopes if available; otherwise we
    skip (records don't carry difficulty today, so an arena that hasn't passed
    envelopes to the report engine just gets an empty breakdown).
    """
    if not envelopes_by_task:
        return {"by_axis": {}}

    by_axis: dict[str, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in focal_records:
        if not _is_scored(r):
            continue
        env = envelopes_by_task.get(r["task_id"])
        if not env:
            continue
        diff = env.get("difficulty", {})
        primary = float(r["score"]["primary"])
        for axis, value in diff.items():
            by_axis[axis][value].append(primary)

    out = {}
    for axis, buckets in by_axis.items():
        rows = []
        for value, primaries in sorted(buckets.items(), key=lambda kv: kv[0]):
            mean = sum(primaries) / len(primaries)
            rows.append({"value": value, "n_scored": len(primaries), "primary_mean": mean})
        out[axis] = rows
    return {"by_axis": out}


def _vs_competitors(
    all_records: list[dict],
    focal_key: tuple[str, str],
    top_k: int = 3,
) -> list[dict]:
    """For the top-k nearest competitors by primary_mean, find tasks where they
    won vs you and the dominant failure category for each side on those tasks."""
    by_player_task: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in all_records:
        if _is_scored(r):
            by_player_task[_player_key(r)][r["task_id"]] = r

    if focal_key not in by_player_task:
        return []
    focal_tasks = by_player_task[focal_key]

    # Mean per player.
    means = {k: (sum(float(r["score"]["primary"]) for r in tasks.values()) / len(tasks))
             for k, tasks in by_player_task.items() if tasks}
    focal_mean = means.get(focal_key, 0.0)

    # Pick competitors whose mean is closest to focal but not the focal itself.
    others = [k for k in means if k != focal_key]
    others.sort(key=lambda k: abs(means[k] - focal_mean))
    nearest = others[:top_k]

    out = []
    for k in nearest:
        their_tasks = by_player_task[k]
        common = set(focal_tasks.keys()) & set(their_tasks.keys())
        their_wins = []
        your_wins = []
        for tid in common:
            yp = float(focal_tasks[tid]["score"]["primary"])
            tp = float(their_tasks[tid]["score"]["primary"])
            if tp > yp:
                their_wins.append(tid)
            elif yp > tp:
                your_wins.append(tid)

        def _dominant_category(records, task_ids):
            cats: Counter = Counter()
            for tid in task_ids:
                rec = records.get(tid)
                if not rec:
                    continue
                for f in (rec.get("score") or {}).get("findings") or []:
                    if f.get("category"):
                        cats[f["category"]] += int(f.get("count", 1))
            return cats.most_common(1)[0][0] if cats else None

        out.append({
            "competitor_id": k[0],
            "competitor_version": k[1],
            "their_primary_mean": means[k],
            "tasks_where_they_won": len(their_wins),
            "tasks_where_you_won": len(your_wins),
            "your_dominant_failure_mode_on_their_wins": _dominant_category(focal_tasks, their_wins),
            "their_dominant_failure_mode_on_your_wins": _dominant_category(their_tasks, your_wins),
        })
    return out


def _version_diff(
    focal_records: list[dict],
    all_records: list[dict],
    focal_key: tuple[str, str],
    category_severity: dict[str, str],
) -> dict | None:
    """If a previous version of this player has scored records, compute deltas."""
    pid, pver = focal_key
    other_versions = sorted(
        {r.get("player_version", "") for r in all_records
         if r["player_id"] == pid and r.get("player_version", "") != pver
         and r.get("player_version", "")},
        key=_version_sort_key,
    )
    if not other_versions:
        return None
    # "previous" = highest semver-aware version below the focal one (or the
    # highest non-focal version if none is below — first-time submissions
    # may report against a backfill).
    focal_key_v = _version_sort_key(pver)
    below = [v for v in other_versions if _version_sort_key(v) < focal_key_v]
    prev = below[-1] if below else other_versions[-1]
    prev_records = [r for r in all_records if r["player_id"] == pid and r.get("player_version", "") == prev]

    cur_mean, _ = _mean_ci([float(r["score"]["primary"]) for r in focal_records if _is_scored(r)])
    prev_mean, _ = _mean_ci([float(r["score"]["primary"]) for r in prev_records if _is_scored(r)])

    cur_hist = {h["category"]: h["n_findings"] for h in _error_histogram(focal_records, category_severity)}
    prev_hist = {h["category"]: h["n_findings"] for h in _error_histogram(prev_records, category_severity)}
    all_cats = set(cur_hist) | set(prev_hist)
    improved = sorted(c for c in all_cats if cur_hist.get(c, 0) < prev_hist.get(c, 0))
    regressed = sorted(c for c in all_cats if cur_hist.get(c, 0) > prev_hist.get(c, 0))

    return {
        "previous_version": prev,
        "primary_delta": round(cur_mean - prev_mean, 4),
        "categories_improved": improved,
        "categories_regressed": regressed,
    }


# ---------- improvement priorities ------------------------------------------

_SEVERITY_WEIGHT = {"major": 2.0, "minor": 1.0}
_AXIS_GAP_MIN = 0.10
_HIDDEN_SHARE_GAP_MIN = 0.10
_HIDDEN_MEAN_GAP_MIN = 0.10
_MAX_FAILURE_MODES = 3
_MAX_COMPETITORS = 2


def field_wide_constant_categories(all_records: list[dict],
                                   *, min_players: int = 2) -> dict[str, int]:
    """Finding categories whose count is IDENTICAL for every player in the arena.

    Such a category cannot be a property of any tool: nothing a player did
    changed it, so it is a property of the gold. Measured 2026-08-21 on
    pdf-text-fidelity-v1: every deterministic tool had exactly 56
    `footnote_missed` findings over 40 tasks -- docpluck, GROBID, liteparse and
    pdftotext alike -- because the real-paper gold hardcoded an empty footnote
    list. The report then told each of them, individually, that reducing
    `footnote_missed` was their number-one priority. It was a constant no tool
    had ever moved.

    Returns {category: the shared count}. Requires at least `min_players`
    scored players: with one player every category is trivially constant, and
    suppressing all of them would empty the report rather than clean it.
    """
    from collections import defaultdict

    per_player: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    players: set[str] = set()
    for r in all_records:
        if not _is_scored(r):
            continue
        pid = r.get("player_id")
        if not pid:
            continue
        players.add(pid)
        for f in (r.get("score") or {}).get("findings") or []:
            cat = f.get("category")
            if cat:
                per_player[pid][cat] += 1

    if len(players) < min_players:
        return {}

    categories = {c for pid in per_player for c in per_player[pid]}
    out: dict[str, int] = {}
    for cat in sorted(categories):
        values = {per_player.get(pid, {}).get(cat, 0) for pid in players}
        if len(values) == 1:
            n = values.pop()
            if n:
                out[cat] = n
    return out


def _improvement_priorities(
    *,
    public_hist: list[dict],
    held_out_hist: list[dict],
    public_mean: float | None,
    held_out_mean: float | None,
    diff_breakdown: dict,
    competitors: list[dict],
    cat_examples: dict[str, list[str]],
    competitor_examples: dict[str, list[str]],
    constant_categories: dict[str, int] | None = None,
) -> list[dict]:
    """Auto-derived, ranked improvement priorities. Pure derivation from the
    report's own aggregates — fixed sentence frames filled with numbers, no
    hand-written advice. `hidden_gap` items NEVER carry task ids or examples
    (holdout preservation; see spec 2026-05-06 D6)."""
    items: list[dict] = []

    # 1. Failure modes — top public categories by severity-weighted share.
    #
    # A category whose count is identical for EVERY player in the arena is
    # excluded first. It is a property of the gold, not of this tool, and
    # publishing it as a priority tells the reader to go and fix something no
    # tool has ever been able to move. See field_wide_constant_categories().
    constant = constant_categories or {}
    public_hist = [c for c in public_hist if c.get("category") not in constant]
    ranked = sorted(
        public_hist,
        key=lambda c: (_SEVERITY_WEIGHT.get(c.get("severity", "minor"), 1.0)
                       * float(c.get("share_of_findings", 0.0))),
        reverse=True,
    )
    for c in ranked[:_MAX_FAILURE_MODES]:
        cat = c["category"]
        items.append({
            "kind": "failure_mode",
            "headline": (f"Reduce `{cat}` ({c.get('severity', 'minor')}): "
                         f"{int(round(c.get('share_of_findings', 0.0) * 100))}% of your findings, "
                         f"{c.get('n_tasks_affected', 0)} tasks affected."),
            "evidence": {
                "category": cat,
                "severity": c.get("severity", "minor"),
                "share_of_findings": c.get("share_of_findings", 0.0),
                "n_tasks_affected": c.get("n_tasks_affected", 0),
            },
            "public_examples": cat_examples.get(cat, [])[:3],
        })

    # 2. Weakest difficulty axis — largest within-axis primary_mean gap >= min.
    worst = None  # (gap, axis, low_bucket, high_bucket)
    for axis, buckets in ((diff_breakdown or {}).get("by_axis") or {}).items():
        scored = [b for b in buckets if b.get("primary_mean") is not None]
        if len(scored) < 2:
            continue
        lo = min(scored, key=lambda b: b["primary_mean"])
        hi = max(scored, key=lambda b: b["primary_mean"])
        gap = hi["primary_mean"] - lo["primary_mean"]
        if gap >= _AXIS_GAP_MIN and (worst is None or gap > worst[0]):
            worst = (gap, axis, lo, hi)
    if worst is not None:
        _, axis, lo, hi = worst
        items.append({
            "kind": "difficulty_axis",
            "headline": (f"You fall off on `{axis}`={lo['value']} "
                         f"({lo['primary_mean']:.2f}) vs `{axis}`={hi['value']} "
                         f"({hi['primary_mean']:.2f})."),
            "evidence": {"axis": axis, "weak_value": lo["value"],
                         "weak_mean": lo["primary_mean"], "best_value": hi["value"],
                         "best_mean": hi["primary_mean"], "gap": round(worst[0], 4)},
            "public_examples": [],
        })

    # 3. Beaten by — competitors who win more public tasks than you.
    beaten = sorted(
        [c for c in competitors
         if c.get("tasks_where_they_won", 0) > c.get("tasks_where_you_won", 0)],
        key=lambda c: c.get("their_primary_mean", 0.0), reverse=True,
    )
    for c in beaten[:_MAX_COMPETITORS]:
        cid = c["competitor_id"]
        fail = c.get("your_dominant_failure_mode_on_their_wins")
        items.append({
            "kind": "beaten_by",
            "headline": (f"`{cid}` ({c.get('their_primary_mean', 0):.2f}) beats you on "
                         f"{c.get('tasks_where_they_won', 0)} public tasks"
                         + (f"; your dominant failure there is `{fail}`." if fail else ".")),
            "evidence": {"competitor_id": cid,
                         "their_primary_mean": c.get("their_primary_mean"),
                         "tasks_where_they_won": c.get("tasks_where_they_won", 0),
                         "your_dominant_failure_mode": fail},
            "public_examples": competitor_examples.get(cid, [])[:3],
        })

    # 4. Hidden-split gap — broad signal only, NO examples / task ids.
    pub_share = {c["category"]: float(c.get("share_of_findings", 0.0)) for c in public_hist}
    for c in held_out_hist:
        cat = c["category"]
        ho_share = float(c.get("share_of_findings", 0.0))
        if ho_share - pub_share.get(cat, 0.0) >= _HIDDEN_SHARE_GAP_MIN:
            items.append({
                "kind": "hidden_gap",
                "headline": (f"On the hidden split, `{cat}` is a larger share of your "
                             f"findings ({int(round(ho_share*100))}% vs "
                             f"{int(round(pub_share.get(cat, 0.0)*100))}% public). "
                             f"Examples withheld."),
                "evidence": {"category": cat, "public_share": pub_share.get(cat, 0.0),
                             "held_out_share": ho_share},
            })
    if (held_out_mean is not None and public_mean is not None
            and public_mean - held_out_mean >= _HIDDEN_MEAN_GAP_MIN):
        items.append({
            "kind": "hidden_gap",
            "headline": (f"Your hidden-split score ({held_out_mean:.2f}) trails your public "
                         f"score ({public_mean:.2f}) by {public_mean - held_out_mean:.2f} — "
                         f"generalize beyond the public benchmark. Examples withheld."),
            "evidence": {"signal": "overall_mean", "public_mean": public_mean,
                         "held_out_mean": held_out_mean},
        })

    for i, item in enumerate(items, start=1):
        item["rank"] = i
    return items


# ---------- per-task drilldown bundles ----------------------------------


def _public_drilldown(
    focal_record: dict,
    all_records: list[dict],
    envelope: dict | None,
    ground_truth: Any,
) -> dict:
    """Public per-task drilldown: full output, full gold, all findings verbatim."""
    task_id = focal_record["task_id"]
    # Best competitor on this task (highest primary).
    best = max(
        (r for r in all_records if r["task_id"] == task_id and _is_scored(r)),
        key=lambda r: float(r["score"]["primary"]),
        default=None,
    )
    best_block = None
    if best is not None:
        best_block = {
            "player_id": best["player_id"],
            "player_version": best.get("player_version", ""),
            "primary": float(best["score"]["primary"]),
            "findings": (best.get("score") or {}).get("findings") or [],
        }

    return {
        "task_id": task_id,
        "task_set_version": focal_record["task_set_version"],
        "difficulty": (envelope or {}).get("difficulty", {}),
        "input_hash": focal_record.get("input_hash"),
        "your_output": focal_record.get("output", {}),
        "gold": ground_truth,
        "your_score": focal_record["score"],
        "your_findings": (focal_record.get("score") or {}).get("findings") or [],
        "best_player_score": best_block,
    }


def _held_out_drilldown(focal_record: dict, envelope: dict | None) -> dict:
    """Held-out per-task drilldown: difficulty + categorical findings only."""
    return {
        "task_id": focal_record["task_id"],
        "task_set_version": focal_record["task_set_version"],
        "difficulty": (envelope or {}).get("difficulty", {}),
        "your_score": {
            "primary": focal_record["score"]["primary"],
            "breakdown": (focal_record.get("score") or {}).get("breakdown") or {},
        },
        "your_findings": (focal_record.get("score") or {}).get("findings") or [],
    }


def _reproduce_envelope(envelope: dict | None) -> dict | None:
    """Verbatim envelope so a tool author can re-feed the same input."""
    return envelope


# ---------- Markdown rendering ------------------------------------------


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "(no data)\n"
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join("---" for _ in headers) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


def render_markdown(report: dict) -> str:
    s = report["summary"]
    lines: list[str] = []
    lines.append(f"# Tool Report — `{report['player']['id']}@{report['player']['version']}`")
    lines.append("")
    lines.append(f"**Arena:** `{report['arena']['id']}` (task set `{report['arena']['task_set_version']}`)")
    lines.append(f"**Generated:** {report.get('generated_at_utc', '')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    rank_str = f"{s['rank_by_mean']}/{s['n_competitors']}" if s.get("rank_by_mean") else "—"
    lat_str = f"{int(s['latency_ms_mean'])}ms" if s.get("latency_ms_mean") is not None else "—"
    cost_str = f"${s['cost_usd_mean']:.4f}" if s.get("cost_usd_mean") is not None else "—"
    lines.append(_md_table(
        ["Primary score", "Rank by mean", "n scored", "n excluded", "n errored", "Latency", "Cost"],
        [[f"{s['primary_mean']:.3f} ± {s['primary_ci_half']:.3f}",
          rank_str, s["n_scored"], s["n_excluded"], s["n_errored"], lat_str, cost_str]],
    ))

    prios = report.get("improvement_priorities") or []
    if prios:
        lines.append("")
        lines.append("## Improvement priorities")
        lines.append("")
        for p in prios:
            ex = p.get("public_examples") or []
            suffix = f"  \n   Examples: {', '.join(ex)}" if ex else ""
            lines.append(f"{p['rank']}. {p['headline']}{suffix}")
        lines.append("")

    # Constant categories get their own section rather than being silently
    # dropped. Suppressing them from the PRIORITIES is right — ranking a
    # gold-side constant tells a tool author it is their differentiator, which is
    # false. Hiding them entirely would be the opposite error: it would read as
    # "you have no problem here", when in fact the whole field has the same one.
    constants = report.get("constant_categories") or {}
    if constants:
        lines.append("## Shared by the whole field")
        lines.append("")
        lines.append(
            "Every player in this arena has the IDENTICAL count in each category "
            "below, so nothing any tool did has moved it. They are excluded from "
            "the priorities above because they cannot tell you what makes your "
            "tool different — not because they do not matter. Where the arena "
            "publishes a magnitude for one of these (for example "
            "`leak_per_1k_body`), that column is where the difference between "
            "tools actually shows up.")
        lines.append("")
        for cat, n in sorted(constants.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{cat}`: {n} (same for every player)")
        lines.append("")

    vd = report.get("version_diff")
    if vd:
        lines.append(f"## Version diff vs `{vd['previous_version']}`")
        lines.append("")
        lines.append(f"- **Primary delta:** {vd['primary_delta']:+.4f}")
        if vd.get("categories_improved"):
            lines.append(f"- **Improved on:** {', '.join('`' + c + '`' for c in vd['categories_improved'])}")
        if vd.get("categories_regressed"):
            lines.append(f"- **Regressed on:** {', '.join('`' + c + '`' for c in vd['categories_regressed'])}")
        lines.append("")

    hist = report.get("error_histogram") or []
    if hist:
        lines.append("## Error category histogram")
        lines.append("")
        rows = [[h["category"], h["severity"], h["n_findings"], h["n_tasks_affected"],
                 f"{h['share_of_findings']:.1%}"] for h in hist]
        lines.append(_md_table(
            ["Category", "Severity", "Findings", "Tasks affected", "Share of findings"],
            rows,
        ))

    diff = report.get("difficulty_breakdown", {}).get("by_axis", {})
    if diff:
        lines.append("## Difficulty breakdown")
        lines.append("")
        for axis, rows in diff.items():
            lines.append(f"### Axis: `{axis}`")
            lines.append("")
            lines.append(_md_table(
                ["Value", "n scored", "Primary mean"],
                [[r["value"], r["n_scored"], f"{r['primary_mean']:.3f}"] for r in rows],
            ))

    vs = report.get("vs_competitors") or []
    if vs:
        lines.append("## vs. Competitors (nearest by score)")
        lines.append("")
        rows = [[
            f"`{c['competitor_id']}@{c['competitor_version']}`",
            f"{c['their_primary_mean']:.3f}",
            c["tasks_where_they_won"],
            c["tasks_where_you_won"],
            f"`{c['your_dominant_failure_mode_on_their_wins']}`" if c.get("your_dominant_failure_mode_on_their_wins") else "—",
            f"`{c['their_dominant_failure_mode_on_your_wins']}`" if c.get("their_dominant_failure_mode_on_your_wins") else "—",
        ] for c in vs]
        lines.append(_md_table(
            ["Competitor", "Their mean", "They won", "You won", "Your dominant failure on their wins", "Their dominant failure on your wins"],
            rows,
        ))

    pub = report.get("public_drilldown_index") or []
    if pub:
        lines.append("## Public-task drilldown")
        lines.append("")
        lines.append("Per-task files include your output, the gold answer, and a `correct_value` for each finding so you can self-train. Reproduce envelopes are alongside.")
        lines.append("")
        def _fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        rows = [[
            t["task_id"], _fmt(t.get("your_score")), _fmt(t.get("best_score")),
            f"`{t['best_player']}`", f"`{t['findings_file']}`", f"`{t['reproduce_file']}`",
        ] for t in pub]
        lines.append(_md_table(
            ["Task", "Your score", "Best score", "Best player", "Findings", "Reproduce"],
            rows,
        ))

    ho = report.get("held_out_aggregate") or {}
    lines.append("## Held-out aggregate")
    lines.append("")
    lines.append("Per-task drilldown is **redacted by construction** for held-out tasks — see Decisions Log D1/D4 in the design spec. You see categorical counts and difficulty distribution; you do not see the input or the gold.")
    lines.append("")
    if ho.get("n_scored"):
        lines.append(f"- **n scored on held-out:** {ho['n_scored']}")
        if ho.get("primary_mean") is not None:
            lines.append(f"- **Primary mean (held-out):** {ho['primary_mean']:.3f}")
        if ho.get("category_histogram"):
            lines.append("")
            rows = [[h["category"], h["severity"], h["n_findings"], h["n_tasks_affected"]] for h in ho["category_histogram"]]
            lines.append(_md_table(
                ["Category", "Severity", "Findings", "Tasks affected"],
                rows,
            ))
    else:
        lines.append("(no held-out tasks scored in this run)")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------- README that ships with each report bundle ------------------


_BUNDLE_README = """# How to read this report

Generated by Meta Science Arena's framework report engine.

## Files

- `tool_report.json` — canonical, machine-readable. Treat this as the source of truth.
- `tool_report.md` — human-readable rendering of the JSON.
- `findings/public/<task_id>.json` — full per-task drilldown for public tasks. Includes your output, the gold answer, and `correct_value` for each finding. Use this to self-train.
- `findings/held_out/<task_id>.json` — categorical-only drilldown for held-out tasks. By design, contains no input, no gold, no `correct_value`. See the design spec for why.
- `reproduce/<task_id>.json` — verbatim task envelope for public tasks. Hand it to your adapter to reproduce the exact input the framework gave you.

## Reproducing

```bash
python -m framework reproduce \\
  --arena <arena_id> \\
  --task-set <task_set_version> \\
  --task <task_id> \\
  --player <your_player_id>
```

(Reproduce CLI lands in a follow-up; meanwhile, the envelopes under `reproduce/` are valid input to any player adapter.)

## Why no held-out drilldown?

Meta Science Arena's leaderboard credibility comes from a held-out task pool that no player ever sees in raw form. Per-task gold and inputs would defeat that guarantee, so they are redacted before the records are even written to disk. You still get categorical counts, difficulty stratification, and your aggregate score — enough to know *what kind* of failures dominate, just not *which specific* held-out tasks they happened on.
"""


# ---------- public entry point ------------------------------------------


def generate_report(
    *,
    arena_dir: Path,
    task_set_version: str,
    player_id: str,
    player_version: str,
    envelopes_by_task: dict[str, dict] | None = None,
    ground_truth_lookup=None,
    tier: str | None = None,
) -> Path:
    """Generate a full report bundle. Returns the bundle directory.

    Args:
        arena_dir: arenas/<arena_id>/
        task_set_version: e.g. "v1"
        player_id, player_version: focal player
        envelopes_by_task: optional {task_id: envelope}; if provided, public
            drilldown will include difficulty + reproduce envelopes. If None,
            we still emit the report but public drilldown skips reproduce.
        ground_truth_lookup: optional callable(task_id) -> ground_truth; if
            provided, public drilldown includes the gold inline. If None,
            public drilldown stores the player's output + score only.
        tier: for a task set with DATA TIERS (``task_sets/<v>/tiers.jsonl``,
            framework/data_tiers.py), the tier to report on. Required there: a report
            over every record would pool populations, which the board never does. A tier
            whose arena.yaml ``data_tiers.<id>.score_label`` is null (rd-zhang2024:
            constraint agreement, not accuracy) is refused; its verdicts come from the
            arena's per-tier analysis (tools/analyze_run.py --tier), never from here.
            The report covers the tier's tasks and the players that ran all of them,
            exactly the records the tier's board publishes.
    """
    arena = load_arena(arena_dir)
    arena_id = arena["arena_id"]
    manifest = arena["manifest"]
    category_severity = {c["id"]: c["severity"] for c in manifest.get("error_categories", [])}

    all_records = load_records_for(arena_dir, task_set_version)
    tier_block = None
    from framework import data_tiers
    tier_index = data_tiers.read_tier_index(arena_dir, task_set_version)
    if tier_index is None and tier is not None:
        raise ValueError(f"{arena_id}/{task_set_version} has no data tiers; drop tier={tier!r}")
    if tier_index is not None:
        plan = data_tiers.plan_data_tiers(tier_index=tier_index, declarations=manifest.get("data_tiers"),
                                          records=all_records)
        if plan["unknown_task_ids"] or plan["undeclared_tiers"]:
            raise ValueError(f"{arena_id}/{task_set_version}: tier map does not cover the run records "
                             f"(unknown tasks {plan['unknown_task_ids'][:5]}, undeclared tiers "
                             f"{plan['undeclared_tiers']})")
        names = [t["id"] for t in plan["tiers"]]
        if tier is None:
            raise ValueError(f"{arena_id}/{task_set_version} has data tiers {names}; pass tier= "
                             "(scores are per tier, never pooled)")
        entry = next((t for t in plan["tiers"] if t["id"] == tier), None)
        if entry is None:
            raise ValueError(f"{arena_id}/{task_set_version} has no data tier {tier!r} (has {names})")
        if entry["score_label"] is None:
            raise ValueError(f"data tier {tier!r} publishes no score (score_label null in arena.yaml "
                             "data_tiers); report its pre-registered bars from the arena's analysis")
        complete = {p["player_id"] for p in entry["players"] if p["complete"]}
        if player_id not in complete:
            raise ValueError(f"{player_id} did not run every task of data tier {tier!r}; it is not on that board")
        in_tier = set(entry["task_ids"])
        all_records = [r for r in all_records if r["task_id"] in in_tier and r["player_id"] in complete]
        row = next(p for p in entry["players"] if p["player_id"] == player_id)
        tier_block = {"id": tier, "label": entry["label"], "score_label": entry["score_label"],
                      "n_tasks": entry["n_tasks"], "mean": row["mean"], "n_scored": row["n_scored"]}
    # What an unscoreable output is worth (TODO A5(b), decided 2026-08-19).
    # Applied ONCE here, over the whole arena, because the rule needs the rest of
    # the field: a failure only counts as the player's when somebody else managed
    # that task. Must run before any per-player filtering.
    all_records, unscoreable_tasks = apply_unscoreable_policy(all_records)
    focal_records = [r for r in all_records
                     if r["player_id"] == player_id and r.get("player_version", "") == player_version]
    if not focal_records:
        raise ValueError(f"No records found for {player_id}@{player_version} in {arena_id}/{task_set_version}")

    focal_key = (player_id, player_version)
    public_records = [r for r in focal_records if r.get("task_visibility", "held_out") == "public"]
    held_out_records = [r for r in focal_records if r.get("task_visibility", "held_out") == "held_out"]

    summary = _summary(focal_records, all_records, focal_key,
                       enum_values=_arena_enum_vocabulary(arena, task_set_version),
                       # The ORPHAN set from the policy above: tasks NO player
                       # could score. A rejection there is the task's fault, and
                       # the policy already refuses to charge it to the score.
                       unattributable_tasks=unscoreable_tasks)
    error_hist = _error_histogram(focal_records, category_severity)
    diff_breakdown = _difficulty_breakdown(focal_records, envelopes_by_task)
    competitors = _vs_competitors(all_records, focal_key)
    version_diff = _version_diff(focal_records, all_records, focal_key, category_severity)

    # Held-out aggregate (no per-task content).
    ho_scored = [r for r in held_out_records if _is_scored(r)]
    ho_primaries = [float(r["score"]["primary"]) for r in ho_scored]
    ho_mean = (sum(ho_primaries) / len(ho_primaries)) if ho_primaries else None
    ho_hist = _error_histogram(held_out_records, category_severity)

    public_hist = _error_histogram(public_records, category_severity)
    pub_scored = [r for r in public_records if _is_scored(r)]
    pub_primaries = [float(r["score"]["primary"]) for r in pub_scored]
    public_mean = (sum(pub_primaries) / len(pub_primaries)) if pub_primaries else None

    # Example public task ids per finding category (worst-scoring first).
    cat_examples: dict[str, list[str]] = {}
    for r in sorted(pub_scored, key=lambda r: float(r["score"].get("primary", 1.0))):
        for f in (r["score"].get("findings") or []):
            cat = f.get("category")
            if not cat:
                continue
            ex = cat_examples.setdefault(cat, [])
            if r["task_id"] not in ex and len(ex) < 3:
                ex.append(r["task_id"])

    # Example public task ids where a given competitor beat the focal player.
    competitor_examples: dict[str, list[str]] = {}

    # Build bundle directory.
    bundle = arena_dir / "runs" / task_set_version / "reports" / f"{player_id}@{player_version}"
    if tier_block is not None:
        bundle = bundle / "tiers" / tier
    (bundle / "findings" / "public").mkdir(parents=True, exist_ok=True)
    (bundle / "findings" / "held_out").mkdir(parents=True, exist_ok=True)
    (bundle / "reproduce").mkdir(parents=True, exist_ok=True)

    # Per-task drilldown files.
    public_index = []
    for r in public_records:
        tid = r["task_id"]
        env = (envelopes_by_task or {}).get(tid)
        gt = ground_truth_lookup(tid) if ground_truth_lookup else None
        d = _public_drilldown(r, all_records, env, gt)
        findings_file = f"findings/public/{tid}.json"
        reproduce_file = f"reproduce/{tid}.json"
        (bundle / findings_file).write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
        if env is not None:
            (bundle / reproduce_file).write_text(json.dumps(_reproduce_envelope(env), indent=2, sort_keys=True), encoding="utf-8")

        best = d.get("best_player_score")
        public_index.append({
            "task_id": tid,
            "your_score": float(r["score"]["primary"]) if _is_scored(r) else None,
            "best_score": (best or {}).get("primary"),
            "best_player": f"{(best or {}).get('player_id', '')}@{(best or {}).get('player_version', '')}" if best else None,
            "findings_file": findings_file,
            "reproduce_file": reproduce_file if env is not None else None,
        })

    for r in held_out_records:
        tid = r["task_id"]
        env = (envelopes_by_task or {}).get(tid)
        d = _held_out_drilldown(r, env)
        (bundle / "findings" / "held_out" / f"{tid}.json").write_text(
            json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")

    for entry in sorted(public_index,
                        key=lambda e: ((e["your_score"] if e["your_score"] is not None else 1.0)
                                       - (e["best_score"] or 0.0))):
        bp = entry.get("best_player") or ""
        cid = bp.split("@", 1)[0]
        if not cid or cid == player_id:
            continue
        ys, bs = entry.get("your_score"), entry.get("best_score")
        if ys is None or bs is None or bs <= ys:
            continue
        ex = competitor_examples.setdefault(cid, [])
        if entry["task_id"] not in ex and len(ex) < 3:
            ex.append(entry["task_id"])

    constant_categories = field_wide_constant_categories(all_records)
    priorities = _improvement_priorities(
        public_hist=public_hist, held_out_hist=ho_hist,
        public_mean=public_mean, held_out_mean=ho_mean,
        diff_breakdown=diff_breakdown, competitors=competitors,
        cat_examples=cat_examples, competitor_examples=competitor_examples,
        constant_categories=constant_categories)

    report = {
        "report_id": str(uuid.uuid4()),
        "generated_at_utc": _utc_now_iso(),
        "arena": {"id": arena_id, "task_set_version": task_set_version},
        "player": {"id": player_id, "version": player_version},
        # Present only for a data-tiered task set: which tier this report covers, and the
        # tier's own metric name and mean (framework/data_tiers.py, the board's number).
        **({"data_tier": tier_block} if tier_block is not None else {}),
        "summary": summary,
        "improvement_priorities": priorities,
        # Categories excluded from the priorities above because every player in
        # the arena has the identical count. Published, not hidden: the findings
        # are real and still appear in the histogram — they just cannot be acted
        # on by this tool, and saying so is more useful than silently dropping
        # them or than ranking them first.
        "constant_categories": constant_categories,
        "version_diff": version_diff,
        "error_histogram": error_hist,
        "difficulty_breakdown": diff_breakdown,
        "vs_competitors": competitors,
        "public_drilldown_index": public_index,
        "held_out_aggregate": {
            "n_scored": len(ho_scored),
            "primary_mean": ho_mean,
            "category_histogram": ho_hist,
        },
        # Tasks NO player in this arena could produce a scoreable output for.
        # Deliberately not zeroed: when the whole field fails the same item, the
        # honest reading is that the item is broken, and publishing zeros there
        # would say "every model is bad at this" instead. Surfaced so the
        # suspicion lands on the task, where it belongs.
        "tasks_no_player_could_score": sorted(unscoreable_tasks),
    }

    (bundle / "tool_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (bundle / "tool_report.md").write_text(render_markdown(report), encoding="utf-8")
    (bundle / "README.md").write_text(_BUNDLE_README, encoding="utf-8")
    return bundle

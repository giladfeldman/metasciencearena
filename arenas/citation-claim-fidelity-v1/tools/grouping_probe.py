"""Grouping probe: does judging all of a source's sentences in one call change accuracy?

For every task with >= MIN_GROUP sentences, each sentence is asked again ALONE
(the same prompt, model and settings, one sentence), through the runner image
like any arena call. The single-sentence verdicts are compared, item by item,
with the grouped verdicts from the main run (tools/analyze_run.py --out file).

Output: {"mean_difference", "ci95_lower", "ci95_upper", "n", "n_tasks"} where the
difference is grouped-correct minus single-correct per item, and the CI is a
percentile bootstrap over TASKS (items in one task share a call and an abstract).
That dict is the `grouping` input of the pass bars.

Probe answers are real model output about held-out text, so they go to article
custody (ground_truth/sciencearena-raw/citation-claim-fidelity-v1/), with the
env and call_config hashes of every call.

    python .../grouping_probe.py --player <id> --analysis <analyze_run --out json> --out <custody json>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1]
REPO = ARENA.parents[1]
sys.path.insert(0, str(REPO))
from framework import container, run_env  # noqa: E402
from framework.registry import load_registry  # noqa: E402

MIN_GROUP = 3
BOOT = 2000


def _load(name):
    spec = importlib.util.spec_from_file_location(f"ccf_gp_{name}", ARENA / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scorer, generator, analyze = _load("scorer"), _load("generator"), _load("tools/analyze_run")


def paired_ci(diffs_by_task: dict[str, list[float]], seed: int = 20261014) -> tuple[float, float, float]:
    tasks = list(diffs_by_task.values())
    flat = [d for t in tasks for d in t]
    mean = sum(flat) / len(flat)
    rng = random.Random(seed)
    stats = []
    for _ in range(BOOT):
        pick = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        n = sum(len(t) for t in pick)
        stats.append(sum(sum(t) for t in pick) / n)
    stats.sort()
    return mean, stats[int(0.025 * BOOT)], stats[min(BOOT - 1, int(0.975 * BOOT))]


def conservative_ci(diffs_by_task: dict[str, list[float]], cluster_of_task: dict[str, str]) -> tuple[float, float, float]:
    """The wider of the task-level and the abstract-level bootstrap.

    Tasks rotate sentences of ONE abstract, so tasks are not independent clusters
    (consult 2026-10-15, Sol #8: two abstracts x four tasks gave lower bound -0.083
    by task and -0.167 by abstract). Decided before any grouping data existed, and
    the same rule as the false-alarm bars' max(Wilson, abstract bootstrap) bound.
    """
    by_abstract: dict[str, list[float]] = defaultdict(list)
    for t, d in diffs_by_task.items():
        by_abstract[cluster_of_task[t]].extend(d)
    mean, lo_t, hi_t = paired_ci(diffs_by_task)
    _, lo_a, hi_a = paired_ci(dict(by_abstract))
    return mean, min(lo_t, lo_a), max(hi_t, hi_a)


def load_prior(path: Path, *, player: str, task_set: str, run_file_sha256: str | None,
               max_tasks: int | None) -> dict[tuple[str, int], dict]:
    """Answered single-sentence results of an earlier probe bound to the SAME run file."""
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    s = doc.get("summary", {})
    if (s.get("player"), s.get("task_set"), s.get("run_file_sha256"), s.get("max_tasks")) != \
            (player, task_set, run_file_sha256, max_tasks):
        raise SystemExit("--resume: the existing probe belongs to another player, task set, run file "
                         "or --max-tasks; move it aside first")
    return {(r["task_id"], r["id"]): r for r in doc.get("items", []) if r.get("error") is None}


def probe_complete(summary: dict) -> bool:
    return (summary.get("n_errored") == 0 and summary.get("n_missing_grouped") == 0
            and summary.get("n_expected") is not None and summary.get("n") == summary.get("n_expected"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True)
    ap.add_argument("--analysis", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-tasks", type=int)
    ap.add_argument("--task-set", default="v1", choices=list(generator.KNOWN_TASK_SET_VERSIONS))
    ap.add_argument("--resume", action="store_true",
                    help="reuse the answered pairs of an existing --out probe of the same player, "
                         "task set and run file; call only the errored or missing ones")
    a = ap.parse_args()
    custody = Path(os.environ["ARTICLE_REPOSITORY"]).resolve()
    if custody not in a.out.resolve().parents:
        raise SystemExit("--out must be inside article custody")
    entry = next(e for e in load_registry(REPO / "players" / "registry.yaml") if e["player_id"] == a.player)
    if container.placement(entry) != "container":
        raise SystemExit(f"{a.player} cannot run in the runner image")
    analysis = json.loads(a.analysis.read_text(encoding="utf-8"))
    ameta = analysis["summary"]["meta"]
    if ameta.get("player_id") != a.player or ameta.get("task_set") != a.task_set:
        raise SystemExit(f"--analysis is for {ameta.get('player_id')}/{ameta.get('task_set')}, "
                         f"not {a.player}/{a.task_set}")
    grouped = defaultdict(dict)
    for it in analysis["items"]:
        grouped[it["task_id"]][it["id"]] = it
    n_missing_grouped = 0
    runner = container.ensure_image()
    egress = container.ensure_egress_image()
    env_sha = run_env.store_env(container.container_env(entry, runner, egress, None))
    adapter = container.ContainerizedAdapter(entry, runner)
    adapter.prepare()
    seed = analyze.private_seed(a.task_set)
    results, diffs, cluster_of_task = [], defaultdict(list), {}
    prior = load_prior(a.out, player=a.player, task_set=a.task_set,
                       run_file_sha256=ameta.get("run_file_sha256"), max_tasks=a.max_tasks) if a.resume else {}
    tasks = [e for e in generator.generate(a.task_set, seed, split="private") if e["input"]["group_size"] >= MIN_GROUP]
    for env in tasks[: a.max_tasks]:
        gt = generator.ground_truth(env["task_id"])
        sentences = env["input"]["prompt_fields"]["sentences"].split("\n")
        for k, g in enumerate(gt["items"]):
            g_it = grouped.get(env["task_id"], {}).get(g["id"])
            if g_it is None:
                n_missing_grouped += 1
                continue
            before = prior.get((env["task_id"], g["id"]))
            if before is not None:
                # --resume: this single-sentence call already ANSWERED in an earlier
                # attempt bound to the same run file; it is never re-sampled.
                v, err = before["single_verdict"], None
                results.append(dict(before, grouped_verdict=g_it["verdict"]))
            else:
                single = json.loads(json.dumps(env))
                line = sentences[k].split(". ", 1)[1]
                single["input"]["group_size"] = 1
                single["input"]["prompt_fields"]["sentences"] = "1. " + line
                answered_unparsed = False
                try:
                    out = adapter.play_task(single, timeout_s=300)
                    err = None
                except Exception as exc:  # recorded, never scored as wrong
                    out, err = None, f"{type(exc).__name__}: {exc}"[:300]
                    # An answer the adapter could not parse is the model's answer: no
                    # verdict, as in the main run (analyze_run) and the citeclaim library.
                    # Only a call with no provider response is an error to retry.
                    if adapter.last_raw_response() is not None:
                        err, answered_unparsed = None, True
                body = adapter.last_request_body()
                cc = run_env.store_call_config(run_env.build_call_config(
                    entry=entry, body=body, prompt_template_sha256=run_env.file_sha256(REPO / entry["prompt_template_path"]),
                    timeout_s=300, group_size=1)) if body else None
                v = scorer.guarded_verdicts(out, 1, gt["abstract"])[0] if out is not None else None
                results.append({"task_id": env["task_id"], "id": g["id"], "type": g["type"], "gold": g["gold"],
                                "single_verdict": v, "grouped_verdict": g_it["verdict"], "error": err,
                                "answered_unparsed": answered_unparsed,
                                "env_sha": env_sha, "call_config_sha256": cc, "raw_response": adapter.last_raw_response()})
            single_ok = scorer.is_correct(g["gold"], v)
            if err is None:
                diffs[env["task_id"]].append(float(g_it["correct"]) - float(single_ok))
                cluster_of_task[env["task_id"]] = g_it["cluster"]
    n = sum(len(v) for v in diffs.values())
    # Identity and completeness (consult 2026-10-15): analyze_run accepts the probe for
    # the grouping bar only when every expected pair was compared, for this player,
    # task set and run file (analyze_run.grouping_for_bars).
    summary = {"player": a.player, "task_set": a.task_set, "run_file_sha256": ameta.get("run_file_sha256"),
               "max_tasks": a.max_tasks, "n_expected": sum(len(e["input"]["prompt_fields"]["sentences"].split("\n"))
                                                           for e in tasks),
               "n": n, "n_tasks": len(diffs), "n_errored": sum(1 for r in results if r["error"]),
               "n_missing_grouped": n_missing_grouped}
    if n:
        mean, lo, hi = conservative_ci(diffs, cluster_of_task)
        summary |= {"mean_difference": mean, "ci95_lower": lo, "ci95_upper": hi}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "items": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    # Non-zero when the probe cannot serve the grouping bar (consult 2026-10-15, Sol #7:
    # an errored probe exited 0 and an unattended plan moved on).
    return 0 if probe_complete(summary) else 3


if __name__ == "__main__":
    sys.exit(main())

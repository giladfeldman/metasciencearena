"""Build the gold for publication-bias-v1 task set v2 by running R once per dataset.

    .venv/Scripts/python.exe arenas/publication-bias-v1/tools/build_gold_v2.py \
        --task-set v2 --split revealed --workers 8

Every scored statistic in v2 is computed by tools/pubbias_v2_stats.R (metafor and
RoBMA). This tool generates the datasets with the arena generator, runs
that script on each one in parallel worker processes, and writes

    task_sets/<v>/_r_gold.json              (revealed split — committed)
    task_sets/<v>/_held_out/_r_gold.json    (private split — gitignored)

Each entry binds its statistics to the dataset they were computed on
(`input_sha256`), so the generator refuses a gold file that has drifted from the
datasets it generates.

`--stability-seeds 1,2,3` runs RoBMA again under each extra seed and writes the
results to --stability-out instead of the gold file. That is how the RoBMA
tolerance in scorer.py was sized: it is MCMC, so its gold is one draw, and a
tolerance tighter than its own seed-to-seed spread would score the sampler, not
the player.

The private seed is read from task_sets/<v>/.private_seed and never printed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]
R_SCRIPT = ARENA_DIR / "tools" / "pubbias_v2_stats.R"
GOLD_ROBMA_SEED = 20260930
DEFAULT_RSCRIPT = r"C:\Program Files\R\R-4.4.0\bin\Rscript.exe"


def _generator():
    spec = importlib.util.spec_from_file_location("_pubbias_generator_build", ARENA_DIR / "generator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _private_seed(task_set_version: str) -> int:
    env = os.environ.get("SCIENCEARENA_PRIVATE_SEED")
    if env:
        return int(env)
    path = ARENA_DIR / "task_sets" / task_set_version / ".private_seed"
    if not path.is_file():
        raise SystemExit(f"private seed missing: {path}")
    return int(path.read_text(encoding="utf-8").strip())


def _r_versions(rscript: str) -> dict:
    expr = ("cat(jsonlite::toJSON(list(R=as.character(getRversion()),"
            "metafor=as.character(packageVersion('metafor')),"
            
            "RoBMA=as.character(packageVersion('RoBMA')),"
            "JAGS=as.character(rjags::jags.version())), auto_unbox=TRUE))")
    out = subprocess.run([rscript, "-e", expr], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _run_r(rscript: str, envelope: dict, robma_seed: int, timeout_s: int) -> dict:
    env = dict(os.environ, PUBBIAS_ROBMA_SEED=str(robma_seed))
    proc = subprocess.run([rscript, str(R_SCRIPT)], input=json.dumps(envelope),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"{envelope['task_id']}: Rscript exit {proc.returncode}: {proc.stderr[-800:]}")
    if proc.stdout is None or not proc.stdout.strip():
        raise RuntimeError(f"{envelope['task_id']}: Rscript wrote nothing to stdout")
    stats = json.loads(proc.stdout.strip().splitlines()[-1])
    _assert_complete(envelope["task_id"], stats)
    return stats


# The exact statistics v2 scores. Must match scorer.V2_STATISTICS.
EXPECTED = {
    "egger": ["p"], "begg": ["tau", "p"], "trimfill": ["k0", "estimate"],
    "pet": ["estimate"], "peese": ["estimate"], "three_psm": ["estimate", "lrt_p"],
    "robma": ["bf_bias", "mu"],
}


def _assert_complete(task_id: str, stats: dict) -> None:
    for method, fields in EXPECTED.items():
        for f in fields:
            v = (stats.get(method) or {}).get(f)
            if not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
                raise RuntimeError(f"{task_id}: {method}.{f} is not a finite number: {v!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-set", default="v2")
    ap.add_argument("--split", choices=["revealed", "private"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rscript", default=os.environ.get("RSCRIPT", DEFAULT_RSCRIPT))
    ap.add_argument("--timeout", type=int, default=4 * 3600, help="per-dataset seconds")
    ap.add_argument("--stability-seeds", default="", help="comma list; writes --stability-out, not gold")
    ap.add_argument("--stability-out", type=Path)
    ap.add_argument("--only", default="", help="comma list of task_ids; MERGES into the existing output")
    args = ap.parse_args(argv)

    gen = _generator()
    seed = 0 if args.split == "revealed" else _private_seed(args.task_set)
    envelopes = list(gen._generate_v2(args.task_set, seed, args.split, require_gold=False))
    if args.only:
        keep = set(args.only.split(","))
        envelopes = [e for e in envelopes if e["task_id"] in keep]
    if not envelopes:
        raise SystemExit("no datasets generated — refusing to write an empty gold file")

    stability = [int(s) for s in args.stability_seeds.split(",") if s.strip()]
    jobs = [(e, s) for e in envelopes for s in (stability or [GOLD_ROBMA_SEED])]
    versions = _r_versions(args.rscript)
    print(f"[build_gold_v2] {args.task_set} {args.split}: {len(envelopes)} datasets, "
          f"{len(jobs)} R fits, {args.workers} workers; {versions}", flush=True)

    results: dict[tuple[str, int], dict] = {}
    t0 = _dt.datetime.now()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_run_r, args.rscript, e, s, args.timeout): (e["task_id"], s) for e, s in jobs}
        for fut in as_completed(futs):
            key = futs[fut]
            results[key] = fut.result()  # a failed fit raises here: no partial gold file
            mins = (_dt.datetime.now() - t0).total_seconds() / 60
            print(f"  done {key[0]} seed={key[1]} ({len(results)}/{len(jobs)}, {mins:.1f} min)", flush=True)

    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    if stability:
        out = {"_meta": {"task_set": args.task_set, "split": args.split, "seeds": stability,
                         "r_versions": versions, "computed_at": stamp},
               "fits": {f"{tid}|{s}": st for (tid, s), st in sorted(results.items())}}
        if args.split == "private":
            # Stability output for private datasets would carry held-out numbers.
            raise SystemExit("stability runs are for the revealed split only")
        if args.only and args.stability_out.is_file():
            # Partial rebuild: replace only the refits of the named datasets.
            prev = json.loads(args.stability_out.read_text(encoding="utf-8"))
            keep = {k: v for k, v in prev["fits"].items() if k.split("|")[0] not in set(args.only.split(","))}
            out["fits"] = dict(sorted({**keep, **out["fits"]}.items()))
            out["_meta"]["partial_rebuilds"] = prev["_meta"].get("partial_rebuilds", []) + [
                {"task_ids": sorted(args.only.split(",")), "computed_at": stamp}]
            out["_meta"]["computed_at"] = prev["_meta"]["computed_at"]
        args.stability_out.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
        print(f"[build_gold_v2] wrote {args.stability_out}")
        return 0

    gold = {"_meta": {
        "task_set": args.task_set, "split": args.split, "robma_seed": GOLD_ROBMA_SEED,
        "r_script": "arenas/publication-bias-v1/tools/pubbias_v2_stats.R",
        "r_versions": versions, "computed_at": stamp,
    }}
    for e in envelopes:
        gold[e["task_id"]] = {
            "input_sha256": gen.v2_input_sha256(e["input"]["studies"]),
            "gold_stats": results[(e["task_id"], GOLD_ROBMA_SEED)],
        }
    path = gen._v2_gold_path(args.task_set, args.split)
    if args.only:
        # Partial rebuild: merge into the existing file, never shrink it.
        if not path.is_file():
            raise SystemExit(f"--only merges into an existing gold file; {path} does not exist")
        prev = json.loads(path.read_text(encoding="utf-8"))
        if prev["_meta"].get("robma_seed") != GOLD_ROBMA_SEED or prev["_meta"].get("r_versions") != versions:
            raise SystemExit("refusing to merge: the existing gold was built with a different RoBMA seed or R stack")
        merged = {**prev, **{k: v for k, v in gold.items() if k != "_meta"}}
        merged["_meta"]["partial_rebuilds"] = prev["_meta"].get("partial_rebuilds", []) + [
            {"task_ids": sorted(e["task_id"] for e in envelopes), "computed_at": stamp}]
        gold = merged
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gold, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[build_gold_v2] wrote {len(envelopes)} gold entries -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

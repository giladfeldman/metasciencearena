"""Run every local tool player on the PDF-universe task sets, both splits, in parallel streams.

    .venv/Scripts/python.exe -m framework.pdfforge.tools.run_universe_players [--streams A,B,C] [--dry-run]

Each job is `python -m framework run` (the normal runner: envelopes, scoring, provenance,
held-out redaction). Public tasks and held-out tasks go to separate files, as every
earlier PDF-arena run did (`--split revealed` / `--split private --held-out-only`).
Streams run in parallel; jobs inside a stream run one after another. A job's exit code
and log are recorded in tmp/pu_runs/status.json; a failed job never stops the others and
is never reported as done.

Launch detached (it takes a day or more on CPU):
    Start-Process -WindowStyle Hidden <venv python> '-m framework.pdfforge.tools.run_universe_players'
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "tmp" / "pu_runs"
TAG = "pu1"
STATUS_FILE = OUT / "status.json"

STATS_PIPES = ["statcheck-pypdf", "statcheck-pdftotext", "statcheck-grobid", "statcheck-docpluck", "statcheck-pdf",
               "escimate-pdftotext", "escimate-docpluck", "escimate-grobid", "metacheck-dev-grobid"]
STREAMS = {
    "A": [  # fast tools
        ("stats-from-pdf-v1", "v1", STATS_PIPES),
        ("pdf-text-fidelity-v1", "v3", ["pdftotext-raw", "grobid-text-only-0-9-1", "grobid-text-only", "docpluck-academic",
                                        "docpluck-standard", "liteparse-default", "liteparse-no-ocr"]),
        ("pdf-table-extraction-v1", "v3", ["grobid-tables-0-9-1", "grobid-tables", "docpluck-tables", "liteparse-tables-heuristic"]),
        ("pdf-section-structure-v1", "v3", ["grobid-sections-0-9-1", "grobid-sections", "docpluck-sections",
                                            "liteparse-sections-heuristic"]),
        ("pdf-reference-parsing-v1", "v3", ["grobid-references", "anystyle-references", "cermine-references"]),
        ("pdf-citation-matching-v1", "v3", ["grobid-citations"]),
    ],
    "B": [  # Docling (slow)
        ("stats-from-pdf-v1", "v1", ["escimate-docling"]),
        ("pdf-text-fidelity-v1", "v3", ["docling-text-2-135"]),
        ("pdf-table-extraction-v1", "v3", ["docling-tables-2-135", "docling-tables"]),
        ("pdf-section-structure-v1", "v3", ["docling-sections-2-135"]),
    ],
    "C": [  # Marker and MinerU (slowest)
        ("pdf-text-fidelity-v1", "v3", ["marker-text", "mineru-flash-text"]),
        ("pdf-table-extraction-v1", "v3", ["marker-tables", "mineru-flash-tables"]),
        ("pdf-section-structure-v1", "v3", ["marker-sections", "mineru-flash-sections"]),
    ],
    "D": [  # metacheck at its development version on the five text arenas metacheck 0.1.0 plays
        ("significance-language-v1", None, ["metacheck-dev"]),
        ("reporting-completeness-v1", None, ["metacheck-dev"]),
        ("power-reporting-v1", None, ["metacheck-dev"]),
        ("reference-integrity-v1", None, ["metacheck-dev"]),
        ("open-practices-repro-v1", None, ["metacheck-dev"]),
    ],
}


def _jobs(stream: str):
    for arena, version, players in STREAMS[stream]:
        for player in players:
            if version is None:
                # text arenas with real revealed/private splits, at their CURRENT task set
                import yaml
                from framework.discovery import resolve_task_set_version
                man = yaml.safe_load((REPO / "arenas" / arena / "arena.yaml").read_text(encoding="utf-8"))
                version = resolve_task_set_version(man)
                splits = (("revealed", [], "mcdev"), ("private", [], "mcdev"))
                ts = ["--task-set", version]
            else:
                splits = (("revealed", [], TAG), ("private", ["--held-out-only"], f"{TAG}-heldout"))
                ts = ["--task-set", version]
            for split, extra, tag in splits:
                yield {"arena": arena, "version": version, "player": player, "split": split,
                       "argv": [sys.executable, "-m", "framework", "run", "--arena", arena, *ts,
                                "--players", player, "--split", split, *extra, "--tag", tag]}


def _task_errors(job: dict) -> tuple[int, int]:
    """(records, records that errored) in the job's run file. A run whose every task failed
    still exits 0 (each failure is a recorded, scored zero), so rc alone is not a verdict."""
    tag = job["argv"][job["argv"].index("--tag") + 1]
    f = REPO / "arenas" / job["arena"] / "runs" / job["version"] / f"{job['player']}__{job['split']}__{tag}.jsonl"
    n = bad = 0
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            b = (r.get("score") or {}).get("breakdown")
            bad += 1 if r.get("error") or (isinstance(b, dict) and b.get("error")) else 0
    return n, bad


def _run_stream(stream: str, status: dict, dry: bool, only: set | None = None, rerun: bool = False) -> None:
    for job in _jobs(stream):
        key = f"{job['arena']}|{job['version']}|{job['player']}|{job['split']}"
        if only is not None and job["player"] not in only:
            continue
        if status.get(key, {}).get("rc") == 0:
            continue  # resumable: a finished job is not re-run
        if rerun:
            # a player fixed after its run: replace the whole run file, never append to it
            job["argv"] = [*job["argv"], "--overwrite"]
        log = OUT / "logs" / (key.replace("|", "__") + ".log")
        if dry:
            print(stream, " ".join(job["argv"][2:]))
            continue
        t0 = time.time()
        with log.open("w", encoding="utf-8") as f:
            rc = subprocess.run(job["argv"], cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
        n, bad = _task_errors(job)
        status[key] = {"rc": rc, "seconds": round(time.time() - t0), "log": str(log), "stream": stream,
                       "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "records": n, "task_errors": bad}
        STATUS_FILE.write_text(json.dumps(status, indent=1, sort_keys=True), encoding="utf-8")


def _status_path(arg: str) -> Path:
    """A bare name lives in the runs folder; a path is taken from the repo root as given.
    A folder that does not exist fails NOW, not after the first job has run for hours."""
    p = Path(arg)
    sf = OUT / p if p.parent == Path(".") else (p if p.is_absolute() else REPO / p)
    if not sf.parent.is_dir():
        raise SystemExit(f"--status-file folder does not exist: {sf.parent}")
    return sf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="A,B,C,D")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status-file", default="status.json",
                    help="separate file per concurrently running driver (they must not share one)")
    ap.add_argument("--players", default="",
                    help="comma list: only these players' jobs (with --rerun, after a player fix)")
    ap.add_argument("--rerun", action="store_true",
                    help="re-run with --overwrite; use a fresh --status-file so earlier rc 0 jobs are not skipped")
    a = ap.parse_args(argv)
    only = {x for x in a.players.split(",") if x} or None
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    sf = _status_path(a.status_file)
    global STATUS_FILE
    STATUS_FILE = sf
    status = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
    streams = [s for s in a.streams.split(",") if s]
    with cf.ThreadPoolExecutor(len(streams)) as ex:
        list(ex.map(lambda s: _run_stream(s, status, a.dry_run, only, a.rerun), streams))
    bad = {k: v for k, v in status.items() if v.get("rc") != 0}
    print(f"done: {len(status)} jobs recorded, {len(bad)} failed: {sorted(bad)}")
    errs = {k: f"{v['task_errors']}/{v['records']}" for k, v in status.items() if v.get("task_errors")}
    print(f"jobs with task errors (read each log; an all-error job is a failure): {errs}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

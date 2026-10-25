"""Differential check: our answer-key verdicts against real statcheck on the same printed text.

    .venv/Scripts/python.exe -m framework.pdfforge.tools.statcheck_differential --n 3000

Simulates records of every checkable family, prints them clean or with each planted error,
runs statcheck::statcheck on each printed result (one text each), and compares statcheck's
`error` / `decision_error` with our verdict wherever ours takes a side (ambiguous records
are reported separately). statcheck is NOT the oracle -- it is one of the graded tools --
so a disagreement is investigated, not auto-resolved. Writes a JSON summary to stdout.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys

from framework.player_adapter import resolve_rscript_binary
from framework.pdfforge import statgen as g
from framework.pdfforge.printing import print_record

SIMS = [lambda s: g.simulate_independent_t(s, 40, 42, random.Random(s).uniform(-0.4, 0.7)),
        lambda s: (lambda n: g.simulate_independent_t(s, n, n, 0.3, welch=True))(30 + s % 60),
        lambda s: g.simulate_paired_t(s, 35, random.Random(s).uniform(-0.3, 0.6)),
        lambda s: g.simulate_oneway_F(s, 3, 30, random.Random(s).uniform(0, 0.4)),
        lambda s: g.simulate_interaction_F(s, 25, random.Random(s).uniform(-0.3, 0.5)),
        lambda s: g.simulate_chi2(s, 200, 2, 2, random.Random(s).uniform(0, 0.5)),
        lambda s: g.simulate_chi2(s, 300, 2, 3, random.Random(s).uniform(0, 0.4)),
        lambda s: g.simulate_r(s, 80, random.Random(s).uniform(-0.3, 0.35)),
        lambda s: g.simulate_z_proportions(s, 120, 130, .45, random.Random(s).uniform(.3, .45))]
PLANTS = [None, None, None, "R01", "R02", "R03", "R04", "R05a", "R05c"]

R = r"""
suppressPackageStartupMessages(library(statcheck)); suppressPackageStartupMessages(library(jsonlite))
x <- fromJSON(readLines("stdin", warn = FALSE))
sink(nullfile()); res <- statcheck(x, messages = FALSE); sink()
cat(toJSON(if (is.null(res)) list() else res[, c("source", "error", "decision_error", "raw")], na = "null"))
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args(argv)
    recs = []
    for k in range(a.n):
        sim = SIMS[k % len(SIMS)]
        plant = PLANTS[(k // len(SIMS)) % len(PLANTS)]
        r = print_record(sim(10_000 + k), plant, random.Random(k))
        txt = r.printed["text"]
        if r.printed.get("note") == "one-tailed":
            txt += " (one-tailed)"
        recs.append((r, txt))
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        rf = Path(d) / "diff.R"
        rf.write_text(R, encoding="utf-8")
        proc = subprocess.run([resolve_rscript_binary(None), str(rf)], input=json.dumps([t for _, t in recs]),
                              capture_output=True, text=True, encoding="utf-8", timeout=1800)
    if proc.returncode != 0:
        print(proc.stderr[-1500:], file=sys.stderr)
        return 2
    rows = json.loads(proc.stdout or "[]")
    by_src = {}
    for row in rows:
        by_src.setdefault(int(str(row["source"])), row)
    tally = collections.Counter()
    examples = collections.defaultdict(list)
    for i, (r, txt) in enumerate(recs, 1):
        v = r.verdict
        if v.get("consistent") is None and not v.get("ambiguous"):
            tally["ours_uncheckable"] += 1
            continue
        sc = by_src.get(i)
        if sc is None:
            tally["statcheck_did_not_extract"] += 1
            examples["not_extracted"].append(txt)
            continue
        s_err, s_de = bool(sc["error"]), bool(sc["decision_error"])
        if v.get("ambiguous"):
            tally[f"ambiguous: statcheck error={s_err} de={s_de}"] += 1
            continue
        ours = (not v["consistent"], bool(v["decision_error"]))
        key = "agree" if ours == (s_err, s_de) else f"DISAGREE ours(err={ours[0]},de={ours[1]}) statcheck(err={s_err},de={s_de})"
        tally[key] += 1
        if key != "agree":
            examples[key].append(f"[{r.planted or 'clean'}] {txt}")
    out = {"n": len(recs), "tally": dict(tally.most_common()),
           "examples": {k: v[:6] for k, v in examples.items()}}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

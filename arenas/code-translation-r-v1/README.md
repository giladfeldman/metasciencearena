# code-translation-r-v1 — SPSS / Stata syntax → R

**Task.** Given an analysis script in SPSS syntax (`.sps`) or a Stata do-file
(`.do`), plus the codebook of the dataset it runs against, emit R that
reproduces the same analysis.

**Scoring: executable equivalence.** The player's R is *executed* against a
fixed CSV and the statistics it produces are compared to gold by name. It is not
diffed against a reference translation. A script that reads plausibly but
computes a different quantity scores near zero — which is the entire point, as
the dominant real-world failure mode here is a translation that runs cleanly and
is silently wrong.

**Permissive about form, strict about value.** The arena asks whether a
translation is *accurate*, not whether the translator adopted our conventions.
`harvest.R` runs the player's code verbatim and then recovers the requested
statistics from whatever it left behind — a jmv results object, an `lm`/`htest`,
an ANOVA or coefficient table, plain named variables, or an explicit JSON block.
The dataset is bound under every common frame name (`data`, `df`, `x`, `dat`) in
both original and upper case, and the converters' own runtimes (`stata2r`,
`dplyr`, …) are attached, so a tool is never penalised for its idiom.

This matters more than it sounds. An earlier version *required* every player to
print a JSON object, and scored `spss2rmarkdown` 0.00 on six analyses whose
numbers were exactly right. That measured conformance to the harness rather than
translation quality, and it made deterministic tools structurally unable to
compete with LLMs — who win such a comparison merely by following instructions
better. Three tests now lock the property in
(`test_jmv_idiom_scores_on_accuracy_not_on_print_contract` and friends).

```
composite = execution_rate × statistic_accuracy
```

Multiplying rather than averaging means unrunnable code scores 0 outright: a
translator that emits confident nonsense cannot bank partial credit for it.

## Why this arena exists

Migrating legacy SPSS/Stata analyses to R is a routine step in reproduction and
reanalysis. The languages disagree on defaults, and every disagreement is a
silent-failure opportunity — no error is raised, the number just changes:

| Tier | Analysis | The trap |
|---|---|---|
| T1 | Descriptives | *(clean control — no trap)* |
| T2 | Independent t-test | SPSS/Stata headline the **pooled** test; R's `t.test()` defaults to **Welch** |
| T3 | OLS regression | Listwise deletion across model variables |
| T4 | Recode / transform | A declared **user-missing code** (99) stays a literal number in R; `MEAN.3` requires ≥3 valid values, `rowMeans(na.rm=TRUE)` does not |
| T5 | Factorial ANOVA | SPSS/Stata report **Type III** SS; `anova(lm())` gives **Type I**. Type III *also* needs sum-to-zero contrasts — setting the type alone is still wrong |
| T6 | Filter → model pipeline | `SELECT IF`/`keep` drop cases **permanently**, so statement order changes N |
| T7 | Correlations, two ways | **The pair is the trap.** SPSS `CORRELATIONS` defaults to **pairwise** deletion while `REGRESSION` defaults to **listwise** — one uniform convention gets exactly one of them wrong. Measurable here: pairwise N=171 vs listwise N=166 |
| T8 | Frequency weighting | `WEIGHT BY` is **stateful** and changes N as well as the mean. Base R has no weighting state — `mean()` silently ignores weights, so dropping the statement looks entirely reasonable |
| T9 | `SPLIT FILE` | Stateful **per-group execution** with no R equivalent. The correct translation restructures the program into a grouped operation — a different *shape* of code, not a different argument |

T7–T9 were added on 2026-08-04 because T1–T6 had **saturated**: claude-sonnet-5
scored a perfect 1.000 on all 12 tasks, so the arena had stopped discriminating
at the top of the model range. A benchmark everyone passes measures nothing.

Each analysis is authored in **both** languages, so the `source_language` axis
compares languages at matched difficulty rather than confounding the two.

## Validity is asserted, not assumed

`tests/test_scorer.py` enforces the two properties that make the arena
meaningful:

1. **Oracle check** — the hand-verified reference translation scores exactly
   1.00 on all six analyses.
2. **Trap discrimination** — five naive translations, each carrying one
   cross-language default over unchanged, score strictly lower *while still
   executing cleanly*. Measured on the real data, a naive Welch t-test gets 3 of
   4 statistics wrong (p = .0062 vs .0019), and both Type I SS **and**
   `type=3`-without-`contr.sum` get 2 of 4 wrong.

If trap discrimination ever stops holding, the arena has stopped measuring what
it claims, and the suite fails loudly.

## The two splits

**Revealed** (18 tasks) is the curated matrix above, fully symmetric across
players — that symmetry is what `framework audit` gates.

**Private** is a *genuinely independent* holdout: real third-party scripts under
`task_sets/v1/_held_out/` (gitignored), each a directory carrying its own
`source.sps`/`source.do`, `data.csv`, `meta.yaml`, and executed `gold.json`. When
that directory is empty the private split falls back to the curated matrix under
the secret seed, and the arena says so rather than implying independence it does
not have.

Because a real corpus arrives with whatever constructs its authors happened to
use, it **cannot** mirror the revealed tier × language grid — and hand-picking
real scripts to fit the template would stop them being a holdout. The manifest
therefore declares `parity.independent_holdout: true`, and `check_parity` skips
cell/category matching with an explicit note. Declaring parity we cannot honour
would be worse than declaring none.

## Gold

Gold is an **executed result**, not an asserted one: `tools/build_gold.py` runs
the hand-verified reference R against the fixed dataset and whatever it prints
becomes the answer key. Gold and players therefore go through exactly the same
mechanism.

Requires local R with `jsonlite` and `car`. R is needed to *build gold and to
score*, not to author a translation.

## Layout

```
source_scripts/
  catalog.yaml          6 analyses: tier, dataset, required statistics, traps
  datasets.yaml         column spec for the fixed dataset
  data/wellbeing.csv    generated by tools/make_dataset.py (deterministic)
  spss/*.sps            SPSS source scripts
  stata/*.do            Stata source scripts (same six analyses)
  reference_r/*.R       hand-verified translations — these PRODUCE gold
  gold/*.json           executed gold statistics
fixtures/tool_outputs/  vendored converter output, version-pinned
r_runner.py             sandboxed, timeout-bounded R execution
scorer.py               executable-equivalence scoring
```

## Regenerating

```bash
python arenas/code-translation-r-v1/tools/make_dataset.py         # fixed dataset
python arenas/code-translation-r-v1/tools/build_gold.py           # execute reference -> gold
python arenas/code-translation-r-v1/tools/dump_ground_truth.py    # tracked GT (redacted)
```

`dump_ground_truth.py` routes every entry through
`framework.holdout.redact_ground_truth_entry`, so held-out tasks never carry
`gold_statistics` into a tracked file.

## Players

The non-LLM field is genuinely tiny. A survey on 2026-08-03 (CRAN, CRAN archive,
GitHub, r-universe, rdrr.io) found only **three** real syntax translators in
existence; none has ever been on CRAN, and one has been dormant since 2021. That
asymmetry *is* the finding — the honest claim is "there is almost no tooling",
not "LLMs beat the tools".

| Player | Kind | Source | Status |
|---|---|---|---|
| [`spss2rmarkdown`](https://github.com/giladfeldman/spss2rmarkdown) | tool | SPSS | 60+ commands; emits jmv **report** code |
| [`SPSStoR`](https://github.com/lebebr01/SPSStoR) | tool | SPSS | regex-based; dormant since 2021 |
| [`skranz/stata2r`](https://github.com/skranz/stata2r) | tool | Stata | real parser, actively maintained; **refuses regressions by design** |
| `claude-*-xlat` | ai-model | both | via the `claude` CLI (Claude Max) |

Because all three converters are R packages (two GitHub-only), their output is
captured **once** and committed as version-pinned fixtures rather than invoked
live, so the arena runs on a machine without R. Capture scripts live in
`tools/capture_*.R`. Fixture code is recorded **verbatim** and never patched to
satisfy the arena's contract — patching would score our edits, not the tool.

A task with no fixture is scored as a non-execution, which is exactly what the
real tool would do.

### What each tool actually does

Every number below is the tool's real behaviour under the accuracy-based scorer,
not a harness artefact.

**`spss2rmarkdown`** is the strongest converter in the arena. It scores **1.00**
on descriptives and **1.00** on the t-test — including the T2 pooled-vs-Welch
trap, which it gets right by requesting Student's test explicitly — and **0.80**
on regression. Its two failures are genuine defects it reports about itself:
it truncates `condition` to a non-existent `COND` in the ANOVA call, and for the
recode task it emits `# Error converting: RECODE`, skips `MISSING VALUES`, and
writes `MEAN.3(...)` — an SPSS function that does not exist in R.

**`SPSStoR`** shows its five dormant years. It produces good t-test code (with
`var.equal = TRUE`, so it also understands the pooled default), but returns an
**empty translation** for DESCRIPTIVES and REGRESSION despite the README listing
them as supported, and dies with `could not find function "select_to_r"` /
`"variablelabels_to_r"` — its dispatcher calls handlers the package never
defines.

**`skranz/stata2r`** is the only actively-maintained one, and it behaves exactly
as its README promises: it correctly translates `replace`, `recode`, and `keep`
into calls on its own runtime, and emits `NA` for every estimation command. It is
scoped to data manipulation by design — "will not be usable to convert complete
Stata analyses to R" — so it transforms the data faithfully and then computes
none of the statistics the arena asks for.

Three distinct projects are called "stata2r" — `skranz/stata2r` (the real
translator), `seanmcraig/stata2r` (a 6-command toy), and `stata2r.github.io` (a
cheatsheet, and by far the most-cited). The registry uses fully-qualified ids.

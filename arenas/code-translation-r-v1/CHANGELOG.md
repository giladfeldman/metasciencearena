# Changelog — code-translation-r-v1

## 0.3.0 — 2026-08-04

**The ladder had saturated.** claude-sonnet-5 scored a perfect 1.000 on all 12
tasks at 0.2.0, so the arena had stopped discriminating at the top of the model
range — a benchmark everyone passes measures nothing. Three tiers added, chosen
because the correct R is structurally *different* from the source, not merely a
different argument:

- **T7 `correlations_pairwise`** — SPSS `CORRELATIONS` defaults to PAIRWISE
  deletion while `REGRESSION` defaults to LISTWISE. The pair is the trap: one
  uniform convention gets exactly one of them wrong.
- **T8 `weighted_descriptives`** — `WEIGHT BY` is stateful and changes N as well
  as the mean; base R has no weighting state, so `mean()` silently ignores it.
- **T9 `split_file_groups`** — `SPLIT FILE` runs every following procedure per
  group. No R equivalent; the translation must be restructured.

Dataset gained a `w` weight column (correlated with score, so weighting moves
the mean by ~5 points) and `score` is now occasionally missing. **That last
change was load-bearing:** with `score` complete, "complete on age+hours" and
"complete on all three" were the same 166 rows, so the first T7 gold came out
*identical* for pairwise and listwise (r=0.06733, N=165 both) — the trap tested
nothing. `make_dataset.py` now asserts `pairwise N > listwise N` at generation
time so this cannot regress silently. Verified: reference 1.000 on all three new
tiers; naive translations score 0.500 / 0.000 / 0.000 **while executing cleanly**.

**Real independent holdout wired (X3).** The manifest had declared
`real_holdout_dir` since 0.1.0 but nothing read it — a promise the code did not
keep. `generator.py` now ingests real third-party cases from
`task_sets/v1/_held_out/` (each a directory with its own source script, data,
meta and executed gold), falling back to the curated matrix under the secret
seed when the directory is empty. First case: a paired-samples t-test, a
construct absent from the revealed set — paired gives t=−7.93/df=119/p=1.3e−12
against unpaired t=−2.16/df=235/p=0.032, and the unpaired version scores 0.250
while running cleanly.

A genuinely independent corpus cannot mirror the revealed grid, so the contract
gained `parity.independent_holdout` and `check_parity` skips cell matching with
an explicit note rather than an arena faking a tolerance wide enough to pass.

**Opus 4.8 wired** (`claude-opus-4-8-xlat`). Free tiers remain unwired — the
opencode CLI is installed but no provider keys are set.

## 0.2.0 — 2026-08-04

**Scoring now measures translation accuracy, not harness conformance.**

0.1.0 required every player to print a JSON object of named statistics. That was
wrong: it scored `spss2rmarkdown` 0.00 on six analyses whose numbers were exactly
right, because it reports through jmv result objects rather than `cat(toJSON())`.
The arena was measuring whether a translator adopted our print convention, which
also made deterministic tools structurally unable to compete with LLMs — who win
such a comparison merely by following instructions better.

- New `harvest.R`: runs the player's code verbatim, then recovers the requested
  statistics from whatever it produced — jmv results (including variant-suffixed
  columns like `stat[stud]`, and `models[[i]]$coef`), `lm`/`htest` objects,
  ANOVA/coefficient tables, plain named variables, or an explicit JSON block.
- Values of unassigned top-level expressions are captured, since converters
  idiomatically end with a bare `jmv::descriptives(...)` call.
- The dataset is bound as `data`/`df`/`x`/`dat` in both original and upper case
  (SPSS is case-insensitive and reports upper-cased; SPSStoR documents `x`).
- Converter runtimes (`stata2r`, `restorepoint`, `repboxUtils`, `dplyr`) are
  attached, so translated code that targets them is not failed for a missing
  function the harness withheld.
- Three regression tests lock the property in, incl.
  `test_jmv_idiom_scores_on_accuracy_not_on_print_contract`.

**All three converters captured and competing.** SPSStoR 0.3.0 and
skranz/stata2r 0.1.0 fixtures added (`tools/capture_spsstor.R`,
`tools/capture_stata2r.R`).

Results over 12 tasks (6 analyses × 2 languages), on each tool's own language:

| Player | SPSS | Stata | overall |
|---|---|---|---|
| claude-sonnet-5-xlat | 1.000 | 1.000 | **1.000** |
| claude-haiku-4-5-xlat | 0.792 | 0.903 | 0.847 |
| spss2rmarkdown | **0.592** | — | 0.296 |
| SPSStoR | 0.042 | — | 0.021 |
| stata2r-skranz | — | 0.000 | 0.000 |

`spss2rmarkdown` scores 1.00 on descriptives and on the t-test — passing the T2
pooled-vs-Welch trap — and 0.80 on regression. Its failures are self-reported
defects (`condition` truncated to `COND`; `# Error converting: RECODE` and a
literal `MEAN.3()`). SPSStoR returns empty translations for DESCRIPTIVES and
REGRESSION and calls handlers it never defines (`select_to_r`) — five years
dormant. stata2r translates `replace`/`recode`/`keep` correctly and emits `NA`
for every estimation command, exactly its declared data-manipulation scope.

## 0.1.0 — 2026-08-03

Initial release.

- 12 tasks: 6 analyses × 2 source languages (SPSS `.sps`, Stata `.do`), each
  analysis authored in both languages so `source_language` is crossed with
  `tier` rather than confounded with it.
- Scoring is **executable equivalence**: the player's R is run against a fixed
  180-row CSV and the statistics it prints are compared to executed gold.
- Gold is produced by executing hand-verified reference translations
  (`tools/build_gold.py`), so gold and players share one mechanism.
- Trap ladder T2–T6: pooled-vs-Welch, listwise deletion, user-missing code +
  minimum-valid-count, Type III SS + sum-to-zero contrasts, and filter order.
- Validity tests: the reference translation scores 1.00 on all six analyses, and
  five naive translations are each verified to score lower **while executing
  cleanly**.
- Players: `spss2rmarkdown`, `SPSStoR`, `skranz/stata2r` (vendored version-pinned
  fixtures) plus Claude via the `claude` CLI.
- `spss2rmarkdown` fixtures captured at 0.1.0. It scores 0.00 — a scope mismatch
  (it emits jmv report code, not an executable statistics script), recorded
  verbatim rather than patched.

### Known gaps

- `SPSStoR` and `skranz/stata2r` fixtures are not yet captured, so both are
  currently non-executing placeholders. `skranz/stata2r` refuses regressions by
  design, so it can only ever cover the data-preparation tiers.
- No real-world held-out corpus yet: the private split re-enumerates the same
  curated scripts under a secret seed. Real legacy `.sps`/`.do` scripts from
  published reanalyses would make the held-out split genuinely independent.
- SAS is out of scope: the survey found no viable non-LLM SAS→R translator, so
  a SAS split would be LLM-vs-LLM only.

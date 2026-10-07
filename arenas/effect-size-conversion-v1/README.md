# effect-size-conversion-v1

**Challenge:** Convert a single effect size from one metric to another across the
closed set **{d, r, OR, eta2, f}**. One conversion per task: input
`{value, from, to, context?}`, output `{converted, confidence}`.

This arena is built on Meta Science Arena's revealed/private dual-benchmark framework
(see `contract/README.md` → "Revealed/private dual benchmark").

## Why this is a good arena

Effect-size conversion is *exactly* the kind of task a deterministic tool does
perfectly and an LLM gets *subtly* wrong: the identities are simple but the constants
are easy to misremember (the d→OR scaling is `π/√3`, not `1.81` rounded the wrong way;
d↔r is `d/√(d²+4)`, not `d/√(d²+1)`; `f = d/2`, not `d`). A computed-gold arena turns
that into a crisp, automatically-scorable comparison.

## Canonical formula set (the gold)

Every task's gold is computed in pure Python in `generator.py` with the formulas below,
matched **function-for-function** against the R package **`effectsize`** (and the
**`esc`** family) so the reference tool scores ≈1.0 (cross-validation):

| from → to | formula | effectsize fn |
|---|---|---|
| **d → r** | `r = d / √(d² + h)` | `d_to_r(d, n1, n2)` |
| **r → d** | `d = √h · r / √(1 − r²)` | `r_to_d(r, n1, n2)` |
| **d → OR** | `OR = exp(d · π / √3)` | `d_to_oddsratio(d)` |
| **OR → d** | `d = ln(OR) · √3 / π` | `oddsratio_to_d(OR)` |
| **eta² → f** | `f = √(eta² / (1 − eta²))` | `eta2_to_f(eta2)` |
| **f → eta²** | `eta² = f² / (1 + f²)` | `f_to_eta2(f)` |
| **d → f** | `f = d / 2` (two equal groups) | textbook (`esc::convert_d2f`) |
| **f → d** | `d = 2f` | textbook |

**The d↔r conversion factor `h`** depends on group sizes when supplied:

```
h = (n1 + n2 − 2) · (1/n1 + 1/n2)        when context = {n1, n2}
h = 4                                      when no group sizes are given
```

`h = 4` is the equal/large-sample limit (`n1 = n2 → ∞`), which is `effectsize`'s
default. This is the formula `effectsize::.get_rd_h(n1, n2)` uses, verified directly
against the package before this arena was committed.

All other constants were verified to match `effectsize` to full double precision (e.g.
`d_to_r(0.5) = 0.2425356 = 0.5/√(0.5²+4)`; `log(d_to_oddsratio(0.5)) = 0.9069 =
0.5·π/√3`).

## Input → output

- **Input** (`schemas/input.schema.json`):
  `{value: number, from: metric, to: metric, context?: {n1, n2}}`, where
  `metric ∈ {d, r, OR, eta2, f}`. `context` is supplied only for d↔r with group sizes.
- **Output** (`schemas/output.schema.json`):
  `{converted: number, confidence: number}` (both required). Optional `formula`/`note`
  strings are allowed but omitted when unused.

## Difficulty tiers

T1 simple d↔r · T2 **round-trip controls** (d↔OR, eta²↔f — the inverse must close) ·
T3 every pairwise no-context conversion · T4 **conversions needing context**
(d↔r with unequal n1, n2) · T5 extreme-but-valid magnitudes · T6 small-magnitude mix.
Values stay clear of the degenerate boundaries (|r| < ~0.93, eta² < ~0.85, OR within a
sane range) so rounding never flips agreement.

## Computed gold

The gold is the exact closed-form conversion above, computed in the generator and
independently **re-derived** in the self-consistency tests (an alternative Python
recomputation plus round-trip closure). No committed answer key, no external registry —
gold is regenerated from the seed.

## Scoring

`primary = 0.85 · agreement + 0.15 · calibration`, where:

- `agreement = 1.0` if `|converted − gold| ≤ tol`, decaying linearly to 0 across a
  wider band, where `tol = max(0.01, 0.01·|gold|)` (an absolute floor for tiny targets
  like eta², a relative band for large targets like OR).
- `calibration = 1 − |confidence − within_tol|`.

A correct, fully-confident player scores 1.0; a deterministic tool reproducing the
canonical formulas with confidence 1.0 therefore scores 1.0 (the cross-validation
oracle). Findings: `conversion_error` (converted value out of tolerance).

## Revealed vs private

- **Revealed** (`--split revealed`, seed 0): published gold + full per-task results in
  the Open Benchmark, so developers can reproduce and audit scoring.
- **Private** (`--split private`): secret seed (`task_sets/v1/.private_seed`) + any
  hand-curated real conversion cases under `_held_out/`; scores on the official
  leaderboard. Same tier matrix and conversion-kind coverage as revealed; only the
  concrete values differ.

## Players

A deterministic reference tool — `effectsize-convert`
(`players/adapters/effectsize_convert.R`, dispatching on `from`/`to` to the matching
`effectsize`/`esc` function) — and a Claude-via-CLI baseline
(`claude-haiku-4-5-esconvert`, Claude Code CLI = Claude Max, no API key). Registered in
`players/registry.yaml`.

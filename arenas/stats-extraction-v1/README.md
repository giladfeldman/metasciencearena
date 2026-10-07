# stats-extraction-v1

Synthetic, adversarial benchmark for stats extraction (NHST + effect sizes + CIs) across 6 difficulty tiers with deception injection. **First trial.**

See the design spec: `docs/superpowers/specs/2026-04-29-stats-extraction-arena-design.md`.

In v1, all "players" are invoked by us — this is not a public-submission tournament.

## Run

```bash
# Use the project venv — a bare `python` on the dev box resolves to a different
# interpreter with an incomplete package set. R tools also need RSCRIPT_BINARY.
RSCRIPT_BINARY="C:/Program Files/R/R-4.4.0/bin/Rscript.exe" PYTHONPATH=$(pwd) \
  .venv/Scripts/python.exe -m framework run --arena stats-extraction-v1 --task-set v1 \
  --players statcheck escimate claude-opus-4-8-stats claude-sonnet-5-stats claude-haiku-4-5-stats
.venv/Scripts/python.exe -m framework leaderboard --arena stats-extraction-v1 --task-set v1
```

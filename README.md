# metasciencearena

**Reproducible benchmark arenas for meta-science tools.**
Leaderboard: <https://metasciencearena.app>

Meta-science has plenty of tools — statcheck, GRIM, z-curve, regcheck, oddpub,
metacheck — and almost no way to say which one is better at what. This package
is the measurement layer: task generation, scoring, contamination control, and
leaderboard aggregation, published so that any score on the site can be
recomputed by someone who does not trust us.

```bash
pip install metasciencearena
```

## What is actually in here

| | |
|---|---|
| `framework.discovery` / `framework.registry` | load an arena manifest and a player registry |
| `framework.runner` | play a task set with a player, emit validated run records |
| `framework.scoring` | the scorers — the numbers on the leaderboard come from here |
| `framework.leaderboard` / `framework.report` | aggregation, per-player feedback reports |
| `framework.holdout` | **the single contamination boundary** (see below) |
| `framework.parity` / `framework.audit` | split symmetry, version drift, input freshness |
| `framework.publish` | the rule deciding what may be published — auditable, not asserted |
| `framework.contract.schemas` | JSON Schema for the manifest, task envelope, run record, findings |

## Reproducing a published score

```bash
export SCIENCEARENA_ARENAS_ROOT=/path/to/arenas
metasciencearena arenas list
metasciencearena run --arena grim-consistency-v1 --task-set v1 \
    --players scrutiny-grim --split revealed --tag repro
metasciencearena leaderboard --arena grim-consistency-v1 --task-set v1
```

Arena data is **not** bundled: the package is the scoring machinery, and the
arenas are the data it operates on. Point `SCIENCEARENA_ARENAS_ROOT` at a
checkout of the public arena set. If the root is missing or wrong the CLI
**raises** — it never prints an empty leaderboard and exits 0, because a broken
configuration that looks like a clean result is the failure mode this project
cares most about.

## Contamination control, stated precisely

Every arena has a **revealed** split (published: tasks, gold, scoring formula,
injected-mistake list) and a **held-out** split. For seed-based arenas the
held-out tasks are generated from a private seed that has never been in git; for
real-paper arenas the corpus itself is withheld.

`framework.holdout` is the one place that defines what "held-out" means and what
must be stripped, and every writer of a tracked or published artifact goes
through it. The invariant, in full:

> For any held-out task, no tracked file and no public artifact may carry its
> gold answer, its reconstructable player output, its per-task score breakdown,
> or its `input_hash` membership oracle.

**What we do not claim.** We cannot claim a model has never seen a *published*
paper, and we do not characterise any provider's retention. The benchmark is
contamination-**resistant**, not contamination-**free**. What we control is that
we hand over the task and never the answer.

## Writing an arena

An arena is a directory with `arena.yaml`, a `generator.py` yielding task
envelopes, and a `scorer.py`. Both conform to schemas shipped in this package;
`contract/README.md` in the repository is the prose specification, and
`contract/arena.example.yaml` is an annotated manifest.

## Registering your own player

Adapters are discovered through an entry point, so you do not need to vendor
anything into this package:

```toml
[project.entry-points."metasciencearena.adapters"]
my_tool = "my_package.adapters"
```

Every module in `my_package.adapters` is imported at runtime and registers its
`PlayerAdapter` subclasses. A plugin that fails to import is logged and skipped
— one broken optional dependency never blocks the rest of the field.

## Citing

See `CITATION.cff`. If you cite a specific score, cite the arena id and task-set
version alongside it — task sets are versioned and broadened over time, and a
score is only meaningful against the task set it was produced on.

## Licence

MIT. The private repository holds the web application, the held-out corpora and
the private seeds; this package holds the verifiable logic.

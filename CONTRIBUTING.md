# Contributing to metasciencearena

Thank you for helping. A benchmark is only as trustworthy as its scoring, so the most valuable
contributions are **reproducible reports that a score is wrong**.

## Reporting a scoring problem

Open an issue at <https://github.com/giladfeldman/metasciencearena/issues> with:

1. the arena id and task-set version (e.g. `grim-consistency-v1` / `v1`);
2. the `task_id` and the run record (or the `output` your player produced);
3. what you believe the correct score is, and why — refer to papers by DOI;
4. `pip show metasciencearena` and your Python version.

Please do not paste article text or PDFs into issues; cite the DOI.

## Flagging a test, sending tests, proposing an arena

Signed in with ORCID at <https://metasciencearena.app/contribute> you can flag the exact part of a
test you think is wrong (every block of a test page has a Flag button), send test material in your
own format (spreadsheets, a GitHub or OSF link, or a description — we translate it), or propose a
new arena. A person reads every contribution and replies with a reason; accepted tests enter the
arena's next task-set version, labelled as contributed.

## Adding a player

You do not need to change this package: write an adapter in your own package and expose it via
the `metasciencearena.adapters` entry point (README, "Writing your own player"). If you would
like a tool added to the bundled registry, open an issue with its install instructions and a
run on one public arena.

## Code changes

- This repository is **published from the maintainer's development tree**. Pull requests are
  welcome and reviewed here, then applied upstream and mirrored back, so a merged change arrives
  in the next sync commit rather than as your original commit.
- Add a test that fails without your change and passes with it; run `pytest` (install with
  `pip install -e ".[dev]"`).
- Arena changes follow `contract/README.md`. Never change a frozen task set in place — add a new
  task-set version; `metasciencearena audit --fresh` detects records left stale by an in-place
  change.
- Document every user-visible change — a command, flag, environment variable, adapter class,
  schema field or public function — in `README.md` or `docs/reference.md`. The maintainer's
  release gate fails when any of them is missing from the docs.

## Conduct

Be precise and be kind. Disagreements about a score are settled by recomputing it.

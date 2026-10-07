"""What may be published, decided by a rule rather than a list.

THE PROBLEM THIS SOLVES
-----------------------
The repo is private and stays that way: the app, the held-out corpora and the
private seeds live here. But the funder-facing claim is that a published score
is *reproducible*, and that requires the scoring logic, the task generators and
the contract to be public. So one tree has to be split in two, repeatedly, as
arenas are added.

A hand-maintained "these files are public" list drifts the first time someone
adds an arena and forgets to update it — and it drifts in the dangerous
direction, because forgetting means a NEW arena's held-out data quietly inherits
whatever the wildcard above it allows.

So publishability is *computed*:

    An arena is publishable iff
      (a) its task set has a `.private_seed` — meaning the secret is the seed
          VALUE (gitignored, never in git), not the code; AND
      (b) nothing under it depends on the real-paper gold path
          (`framework.gold`, `framework.jats`, or the local `article-finder`
          skill), which is how gold for copyrighted / real papers is authored.

THE TWO CONDITIONS ARE NOT SYMMETRIC
------------------------------------
An earlier plan recorded that the two halves "agree exactly, 16 + 6 = 22, zero
overlap, zero gap". Measured, that is not what they do:

    seeded      16   <- THIS is the partition: 16 public, 6 private
    gold-free   20   <- 2 arenas import framework.jats
    publishable 16

Condition (a) produces the split on its own. Condition (b) is a one-way VETO: it
can only move an arena from public to private, never the reverse. Four arenas
are unseeded yet gold-free, and that is fine — not being seed-reproducible is
sufficient reason to stay private.

Keeping the veto is still the conservative direction: an arena that acquires a
seed while still authoring gold from real papers would be published by the seed
rule alone, and that mistake cannot be undone once it is on PyPI. Today no arena
is in that state, and `test_publish.py` fails the build if one ever is.

WHAT A PATH RULE CANNOT DO
--------------------------
A path allowlist blocks `_held_out/` but it cannot block an answer that was
copied into `arena.yaml`, into a scorer fixture, or into a doc comment. That is
why `scan_for_leaks` exists: it reads the actual held-out material and refuses
to mirror any file that reproduces a long verbatim run of it. See that
function's docstring for exactly what it does and does not catch.
"""
from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ArenaPublishability",
    "PRIVATE_GOLD_MARKERS",
    "PRIVATE_PACKAGES",
    "PRIVATE_TEST_FILES",
    "imports_a_private_package",
    "PUBLIC_ARENA_GLOBS",
    "PUBLIC_TOP_LEVEL",
    "classify_arenas",
    "code_only",
    "MIRROR_RENAMES",
    "held_out_shingles",
    "PUBLISHED_ARENAS_FILE",
    "public_manifest",
    "published_arena_ids",
    "scan_for_leaks",
]

#: Importing or path-inserting any of these means an arena's gold comes from
#: real papers, which are not ours to republish.
PRIVATE_GOLD_MARKERS = (
    "framework.gold",
    "framework.jats",
    "article-finder",
    "article_finder",
)

#: Repo-root paths mirrored verbatim. Anything not matched here is private by
#: DEFAULT — the list says what goes out, never what stays.
PUBLIC_TOP_LEVEL = (
    "framework/**/*.py",
    "framework/contract/schemas/*.json",
    # REQUIRED, not optional: framework/pricing.py reads this at IMPORT time, so a
    # mirror shipping the module without the table is an ImportError on `import
    # metasciencearena`. Published provider list prices; no secret.
    "framework/contract/*.json",
    # The SHARED identity fixture corpus (2026-09-17). One directory deeper than
    # the glob above, so it was not matched: `framework/**/*.py` ships
    # framework/tests/test_player_identity.py, which reads this file, and a
    # public checkout without it fails at import. Our own synthetic player
    # strings -- no article content, no held-out data, nothing private -- and
    # the file the Node suite reads too, so publishing it is what lets a reader
    # re-run the identity contract in both languages.
    "framework/contract/fixtures/**/*.json",
    # Test FIXTURES. `framework/**/*.py` never matched these, so the mirrored
    # tests referenced fake arenas and registries that were not shipped.
    # (The glob was lost from this sentence by the 2026-08-14 control-character
    # repair, leaving a comment that named no cause. Restored 2026-08-15.)
    "framework/tests/fixtures/**/*.yaml",
    "framework/tests/fixtures/**/*.yml",
    "framework/tests/fixtures/**/*.json",
    "framework/tests/fixtures/**/*.jsonl",
    "contract/README.md",
    "contract/arena.example.yaml",
    # The injected-mistake taxonomy the arenas generate from — data, not answers.
    "taxonomy/**/*.yaml",
    "taxonomy/**/*.md",
    "pyproject.toml",
    "requirements.lock",
    "LICENSE",
    "PACKAGE_README.md",
    # The public docs (2026-09-27, owner directive: the public README must cover ALL
    # functionality). The full reference is too long for the README, and the repo-root
    # `docs/` here is PRIVATE (design notes, handoffs), so the public pages live under
    # `publish_docs/` and land as `docs/` in the mirror via MIRROR_RENAMES.
    # `scripts/check_docs_coverage.py` fails when either drifts from the code.
    "publish_docs/reference.md",
    "CONTRIBUTING.md",
    "CITATION.cff",
    "DATA_HANDLING.md",
    "publish_templates/*.yml",
    # The public repo's .gitignore, managed here since 2026-10-01 so it carries
    # `players/local_paths.yaml` (test_local_paths_file_is_gitignored runs in the mirror).
    "publish_templates/public.gitignore",
    "contract/published_arenas.json",
    # THE INSTRUMENT (added 2026-08-15, user directive: "our prompts and
    # everything about the way we call the models/tools should be shared").
    #
    # A published score is a measurement, and a measurement nobody can inspect
    # the instrument for is not reproducible — it is an assertion. The prompt
    # templates, the player registry and the adapters ARE the instrument: they
    # decide what each tool is asked and how its answer is read back.
    #
    # Verified before publishing, three ways (Fable 5 / Sonnet 5 / Codex, all
    # reproduced locally): every template is input-independent (it interpolates
    # `{{INPUT_TEXT}}` / `{{PDF_PATH}}` / `{{N_PAGES}}` and names no document,
    # DOI or answer); the registry holds zero literal credentials (every one is
    # an env var NAME — `test_registry_declares_credentials_by_env_var_name`
    # pins that); and no adapter reaches the real-paper gold path.
    #
    # The 6 PRIVATE arenas' prompts ship too, and that is deliberate: those
    # arenas are private because their PDFs are not ours to republish, not
    # because the instruction is secret. Hiding the instrument for them would
    # make the private half unauditable for no gain.
    "players/__init__.py",
    "players/registry.yaml",
    "players/prompts/*.txt",
    # Superseded templates, kept so a published record's
    # `prompt_template_sha256` always resolves to readable text. See
    # `players/prompts/_archive/README.md`.
    "players/prompts/_archive/*.txt",
    "players/prompts/_archive/README.md",
    "players/adapters/**/*.py",
    "players/adapters/**/*.R",
    "players/regcheck_shim/**/*.py",
    "players/regcheck_shim/*.md",
)

#: Paths that land under a DIFFERENT name in the mirror.
#:
#: Two cases, both deliberate:
#:  * the repo README is repo-facing (private layout, app, outreach); the
#:    registry-facing one is PACKAGE_README.md and becomes the mirror's README.
#:  * workflow files live under `publish_templates/` here rather than
#:    `.github/workflows/`, so the PRIVATE repo never tries to run a PyPI
#:    publish. Trusted publishing is configured for the public repo only, so a
#:    stray tag here would fail — noisily, but for the wrong reason.
MIRROR_RENAMES = {
    "PACKAGE_README.md": "README.md",
    "publish_docs/reference.md": "docs/reference.md",
    "publish_templates/publish-pypi.yml": ".github/workflows/publish-pypi.yml",
    "publish_templates/public-ci.yml": ".github/workflows/ci.yml",
    "publish_templates/public.gitignore": ".gitignore",
}

#: Per-arena paths mirrored for a PUBLISHABLE arena. Generators and scorers,
#: not generated data: a public task set is reproducible from the arena's
#: revealed seed, so shipping the code ships the tasks without shipping a second
#: copy that can drift from — or leak beyond — what the generator emits.
PUBLIC_ARENA_GLOBS = (
    "arena.yaml",
    "README.md",
    "CHANGELOG.md",
    "difficulty.yaml",
    # Every module at the arena root, not just generator/scorer: arenas have
    # sibling helpers (r_runner.py, _normalize.py, make_dataset.py) that the
    # scorer imports. Naming only the two entry points shipped a scorer whose
    # import failed. Per-file rules still apply — a helper that reaches the gold
    # path or a private package is dropped by the checks below.
    "*.py",
    "catalogs/**/*.yaml",
    "schemas/*.json",
    # Versioned schemas (publication-bias-v1/schemas/v2/) — tests load them, so
    # the flat rule alone shipped a suite that failed at collection.
    "schemas/**/*.json",
    "tests/**/*.py",
    "tools/**/*.py",
    # R is a first-class player language here (harvest.R, tools/*.R). Shipping
    # only the .py half of an arena's harness ships half a harness.
    "*.R",
    "tools/**/*.R",
    # The CONTENT a generator renders from. Placeholder sentences and format
    # variants, not answers — but without them `generate()` raises
    # FileNotFoundError, which is what stats-extraction-v1 (the arena with the
    # most published records) did in every mirror built before 2026-08-15. A
    # published generator that cannot generate is a capability we claim and do
    # not have.
    "templates/**/*.yaml",
    # Reproduction metadata for the task set: the REVEALED seed, task count and
    # generator signature. `NEVER_PUBLISH` still blocks everything else under
    # `task_sets/` — `_held_out/`, `.private_seed`, `_ground_truth.json`.
    "task_sets/*/manifest.yaml",
    # code-translation-r-v1's revealed half: the SPSS/Stata inputs, the fixed
    # dataset, the catalog, and the executed-R gold cached under
    # `source_scripts/gold/`.
    #
    # Publishing an answer key looks like the one thing this module exists to
    # prevent, so the reasoning is recorded rather than assumed. Checked, not
    # inferred: `leaderboard-app/public/data/code-translation-r-v1/ground_truth.json`
    # ALREADY serves all 18 revealed tasks with full envelopes and answers on the
    # live site, and every other public arena's revealed gold is likewise
    # obtainable by anyone — `generator.ground_truth()` recomputes it from the
    # published seed 0. This arena simply CACHES its revealed gold in files
    # (because building it means executing R) instead of recomputing it. So this
    # discloses nothing the design has not already committed to, and withholding
    # it only meant shipping a generator that raised FileNotFoundError.
    #
    # The PRIVATE half is untouched: `task_sets/v1/_held_out/` carries its own
    # `gold.json` per case and is blocked by `NEVER_PUBLISH`.
    "source_scripts/*.yaml",
    "source_scripts/**/*.sps",
    "source_scripts/**/*.do",
    "source_scripts/**/*.R",
    "source_scripts/**/*.json",
    "source_scripts/**/*.csv",
    # Captured OUTPUTS of players that cannot be installed on demand (the three
    # SPSS/Stata->R translators, none of which was ever on CRAN). Registry
    # entries point at these directories, so a mirror without them has three
    # players that cannot run. They are a player's ANSWER, not the gold it is
    # scored against — and this arena's tasks are seed-generated and public, so
    # the answers reveal nothing the task set does not already give away.
    # Computed by glob rather than named per-arena on purpose: a hardcoded path
    # is the drifting list this module exists to avoid. `NEVER_PUBLISH` still
    # drops `_ground_truth.json`, and the verbatim scan still runs over these.
    "fixtures/**/*.json",
    # Community-contributed tasks (T5, docs/design/2026-09-arena-expansion/
    # t5-submissions-spec.md). Contributions are CC BY 4.0 and the revealed
    # split is published with its gold by design, exactly like a generated
    # revealed task. A private-split contribution ships ONLY its commitment.json
    # (sha256 of the accepted bundle); its payload lives under `private/`, which
    # is gitignored AND blocked by `NEVER_PUBLISH`, so it cannot reach the mirror
    # even if someone force-adds it.
    "contributed/*/commitment.json",
    "contributed/*/bundle.json",
    "contributed/*/provenance.jsonl",
)

#: Per-arena paths that are private no matter which arena they belong to.
#: `tools/build_gold.py` is the gold-AUTHORING path: it reaches into the local
#: `article-finder` skill to resolve real papers. Publishing it would publish
#: the route to the answer key even for an arena whose tasks are seed-generated.
#:
#: `catalogs/abs_dirs_v1_as_published.yaml` is the exact pool the frozen
#: open-practices-repro v1 was generated from; one entry is a real personal
#: directory, so it is kept private and the public catalog carries a placeholder
#: (see the note above `abs_dirs` in that arena's repo_templates.yaml). The
#: `catalogs/*.yaml` glob would otherwise publish it.
PRIVATE_ARENA_PATHS = ("tools/build_gold.py", "catalogs/abs_dirs_v1_as_published.yaml")

#: Packages that exist only in the PRIVATE repo. A mirrored file with an
#: unguarded module-level import of one of these cannot run for anybody who
#: installs the package — not a leak, a breakage, but a public repo with a red X
#: is its own kind of unbacked claim.
#:
#: EMPTY SINCE 2026-08-15, and the emptiness is the point: `players` was the only
#: entry, and publishing the instrument (see PUBLIC_TOP_LEVEL) removed it. Three
#: `framework/tests/*` files that imported `players.adapters.*` and broke the
#: public repo's CI within minutes of the first release (2026-08-07) are
#: therefore mirrored again — `--verify-tests` is what proves they now run.
#:
#: The MECHANISM stays, for the next package that is private on its own terms.
#: Because an empty set makes `imports_a_private_package` unfirable — and an
#: unfirable check is decoration — two things carry the real weight now:
#: `test_the_private_import_check_actually_detects_one` exercises the mechanism
#: against a monkeypatched set, and
#: `test_every_first_party_import_in_the_mirror_resolves_inside_the_mirror`
#: checks the actual property this rule was a proxy for.
PRIVATE_PACKAGES: frozenset[str] = frozenset()

#: Test files that are bound to PRIVATE data and cannot run in the mirror.
#:
#: Determined empirically by running the mirrored suite (not guessed), then
#: encoded here with the reason. Every one reads something the package does not
#: ship — `players/prompts/*.txt`, `players/registry.yaml`, a private arena's
#: modules, or the private side of the publish boundary itself.
#:
#: This exists because the first release turned the public repo's CI red: the
#: mirror was shipping the private repo's INTEGRATION tests, which reference
#: assets only a checkout has. A public repo with a failing badge is its own kind
#: of claim the project cannot back — it was the first thing anyone following the
#: funder link would have seen.
#:
#: `scripts/build_public_mirror.py --verify-tests` RUNS the mirrored suite, so
#: this list cannot quietly go stale.
PRIVATE_TEST_FILES = frozenset({
    # NINE ENTRIES REMOVED 2026-08-15. Eight arena `test_generator.py` files and
    # `stats-extraction-v1/tests/test_coverage.py` were excluded for one reason —
    # they read a template under `players/prompts/` — and publishing the
    # instrument removed that reason. They are back in the mirrored suite, and
    # `--verify-tests` (not this comment) is what says they pass there.
    #
    # Read gold from a FILE the mirror never ships (`_ground_truth.json`). Only
    # Codex caught these two: Fable and Sonnet both recommended publishing the
    # whole `players` package, which would have shipped them. Note the contrast
    # with `test_effectsize_convert_adapter_smoke.py` and
    # `test_pcurve_adapter_smoke.py`, which call `generator.ground_truth(...)` —
    # that is COMPUTED from the arena's public seed, so it ships fine. The
    # distinction is the source of the answer, not the word "ground_truth".
    "players/adapters/tests/test_metacheck_adapter_smoke.py",
    "players/adapters/tests/test_transparency_tools_smoke.py",
    # Read private arenas, or run the private side of the publish boundary
    # (whose answer differs by construction inside the mirror).
    "framework/tests/test_publish.py",
    # Resolve the route to the eval-only gold corpus / assert on the PRIVATE
    # repo's tree. `scan_for_leaks` flagged both on sight (2026-08-13) — correctly:
    # publishing a module that resolves gold publishes the route to the answer key.
    # They exist to police the private repo's own CI, where the corpora are absent
    # by design; the mirror has its own `--verify-tests` pass.
    "framework/tests/_corpora.py",
    "framework/tests/test_clean_checkout.py",
    # Imports `_corpora` at module scope, which aborted pytest COLLECTION in the
    # mirror — no tests ran at all (found 2026-08-15 by
    # `test_every_first_party_import_in_the_mirror_resolves_inside_the_mirror`).
    # Unlike `arenas/stats-extraction-v1/tests/test_task_set_v2.py`, which was
    # fixed with a guarded import because six of its assertions need no corpus,
    # every assertion here is about the PRIVATE repo's own posture. Of its six
    # tests: two are corpora-gated (one of them a strict xfail on a known
    # exposure), one greps tracked files for seed values and skips when the tree
    # holds none, and three need no corpus at all — the scope guard and the two
    # hermetic controls added 2026-09-11. Those three would run in the mirror,
    # but the module stays excluded because its SUBJECT is this repo's seed
    # posture and the xfail asserts a debt that exists only here.
    # `test_clean_checkout.py` runs the inverse assertions on a corpus-free
    # checkout.
    "framework/tests/test_private_seed_not_in_tracked_files.py",
    # RE-DERIVED EMPIRICALLY 2026-08-15 by running the mirrored suite (54 -> 10
    # -> 0 failures). Each of these reads an asset the mirror deliberately does
    # not ship, and none of them is a check the PUBLIC repo could run anyway:
    #   * runs/ — never mirrored, so an audit over run records has nothing to see
    "framework/tests/test_audit.py",
    #   * a PRIVATE arena's catalog (replication-target-lookup-v1)
    "framework/tests/test_fred_frida_only.py",
    #   * scripts/ and .claude/skills/ — this repo's own QA tooling
    "framework/tests/test_qa_checks_documented.py",
    #     ADDED 2026-08-29. Same reason, found by `sync_public_mirror.sh`
    #     refusing to write: this test `spec_from_file_location`s
    #     `scripts/run_health.py` at import time, so in the mirror it does not
    #     fail — it fails to COLLECT, taking the whole public suite down with one
    #     FileNotFoundError. That is the "public repo with a failing badge" this
    #     list exists to prevent, and --verify-tests is what caught it.
    "framework/tests/test_run_health.py",
    #     ADDED 2026-08-29, same run, same reason: it loads
    #     `scripts/verify_records.py`, which `scripts/` never mirrors. Its five
    #     tests were invisible until `test_run_health.py`'s COLLECTION error was
    #     removed -- a collection abort hides every failure behind it, which is
    #     how this repo's CI once stayed red for three days.
    "framework/tests/test_verify_records_archive_rule.py",
    #     ADDED 2026-09-01 with QA check C9. It loads
    #     `scripts/qa_coverage_checks.py`, reads
    #     `scripts/published_field_consumers.yaml`, and searches
    #     `leaderboard-app/{app,components,lib}` for consumers -- none of which
    #     the mirror ships. Unlike `test_run_health.py` it imports the script
    #     inside a fixture rather than at module scope, so it would not abort
    #     COLLECTION; it would simply turn every one of its ten tests red, which
    #     is the failing public badge this list exists to prevent.
    "framework/tests/test_qa_c9_published_fields.py",
    #     ADDED 2026-09-02 with the served-vs-committed drift check. It loads
    #     `scripts/check_deploy_drift.py`, which `scripts/` never mirrors. Unlike
    #     the C9 entry above it would SKIP rather than fail there (its pytestmark
    #     guards on the script's absence) -- which is precisely why it is listed:
    #     a permanently-skipping test in the public repo is not coverage, it is
    #     noise that LOOKS like coverage, and this repo has been bitten by
    #     skipif-guarded tests reading as green six times.
    "framework/tests/test_deploy_drift_check.py",
    #     ADDED 2026-09-03 by the mirror republish, which `sync_public_mirror.sh`
    #     REFUSED at --verify-tests. Both assert the PRIVATE repo's own posture.
    #   * Reads the TRACKED .gitignore and asserts `arenas_private/` and the
    #     held-out rule are in it. Its docstring calls the public checkout "the
    #     only sanctioned skip" because `.gitignore` is not mirrored -- but the
    #     public repo has its OWN tracked .gitignore (7 lines, preserved by
    #     `mirror_sync_tree.PROTECTED_FILES`), so there the skip does NOT fire
    #     and 5 of its 6 tests fail. `--verify-tests` did not see that: its temp
    #     tree has no .gitignore at all, so the tests skipped and the gate read
    #     green (46 skipped there vs 40 in the real checkout, measured
    #     2026-09-03 on the exact tree). A harness that cannot reproduce the
    #     public checkout's own files is a gap in its own right -- see TODO.md.
    "framework/tests/test_arenas_private_is_gitignored.py",
    #   * Reads `arenas/*/runs/*/*.jsonl` and REFUSES a vacuous pass on zero
    #     records. `runs/` is never mirrored, so in the public repo it fails by
    #     its own design (2 failures, measured 2026-09-03).
    "framework/tests/test_provenance_labels.py",
    #     ADDED 2026-09-04, same republish. Reads contract/public_manifest.txt
    #     (never mirrored) and walks every mirrored .py for f-string syntax that
    #     Python 3.11 rejects -- the public repo's 3.11 CI job had failed at
    #     COLLECTION since 2026-08-19 on one such line, invisible to every local
    #     run because this machine and the private CI are both newer than 3.11.
    "framework/tests/test_mirrored_python_parses_on_311.py",
    #   * a PRIVATE arena's label_map.yaml (pdf-section-structure-v1)
    "players/adapters/tests/test_liteparse_heuristics.py",
    #   * stats-extraction-v1's committed _ground_truth.json (NEVER_PUBLISH)
    "arenas/stats-extraction-v1/tests/test_coherent_labels.py",
    "framework/tests/test_parity.py",
    "framework/tests/test_registry_attribution.py",
    "framework/tests/test_retry_failed.py",
    "framework/tests/test_runner_provenance.py",
    "framework/tests/test_scoring_text.py",
    "framework/tests/test_stale_input_audit.py",
    # Asserts PRIVATE-repo layout invariants: the 19-arena module-name collision,
    # the pyproject collection config, and that every arena with a scorer has
    # tests. The mirror has 16 arenas and deliberately omits some tests, so the
    # invariant is legitimately false there.
    "framework/tests/test_test_collection_integrity.py",
    # code-translation-r-v1 builds gold by EXECUTING R over
    # `source_scripts/` — which contains `source_scripts/gold/*.json`. Whether
    # that (revealed) gold should be published is a deliberate disclosure
    # decision for the project owner, not something to infer: under-publishing
    # is reversible, over-publishing is not. Until it is decided, the arena's
    # generator and scorer ship (so the LOGIC is auditable) but these two tests,
    # which need the source scripts to run, do not. See TODO.md.
    "arenas/code-translation-r-v1/tests/test_generator.py",
    "arenas/code-translation-r-v1/tests/test_scorer.py",
    # ADDED 2026-09-16, DERIVED EMPIRICALLY by running the mirrored suite, which
    # came back `26 failed, 1328 passed, 77 skipped` and blocked the republish.
    # Nine modules, one class: each asserts something about PRIVATE
    # INFRASTRUCTURE -- this repo's Next app (`leaderboard-app/`), its QA and ops
    # scripts (`scripts/`), or its own CI workflows (`.github/`) -- none of which
    # the package ships or ever could.
    #
    # THIS IS NOT THE HARNESS BEING WRONG. `--verify-tests` has a known blind
    # spot (see `test_arenas_private_is_gitignored.py` above, where a temp tree
    # with no `.gitignore` read green while the real checkout failed 5). So the
    # real public checkout was checked directly before excluding anything:
    # `contract/public_manifest.txt` and `.github/workflows/sync-public.yml` are
    # absent from `giladfeldman/metasciencearena` too. These failures are real
    # there, not an artefact of the scratch tree.
    #
    # Most arrived with the 12 test files added since the 2026-09-04 sync and
    # were swept in by the blanket `framework/**/*.py` allowlist -- the same
    # mechanism that shipped `framework/gold/` in the first mirror build.
    #
    #   * Cross-engine parity: reads `leaderboard-app/lib/compute.ts` and
    #     `leaderboard-app/scripts/lib/report.mjs` to prove the Python and Node
    #     scorers agree. The mirror ships neither, so the comparison has one arm.
    "framework/tests/test_adherence.py",
    "framework/tests/test_orphan_tasks_are_not_charged_as_errors.py",
    #   * Load `scripts/qa_coverage_checks.py` to assert that the C6 gate and the
    #     QA check share ONE ceiling rule. `scripts/` is never mirrored.
    "framework/tests/test_both_gates_share_one_ceiling_rule.py",
    "framework/tests/test_c6_distinguishes_our_ceiling_from_the_models.py",
    "framework/tests/test_c10_ignores_redaction_and_constant_gold.py",
    #   * Loads `scripts/free_tier_pilot.py`. Same reason.
    "framework/tests/test_free_tier_pilot_stands_in_for_the_real_player.py",
    #   * Assert on THIS repo's CI: that the python job builds the leaderboard
    #     bundle before pytest reads it, and that the sync workflow protects what
    #     the sync script protects. Both read `.github/workflows/*.yml`, which
    #     arrive in the public repo only through MIRROR_RENAMES and under
    #     different names -- `sync-public.yml` has no public counterpart at all,
    #     by design: the public repo does not sync anything.
    "framework/tests/test_ci_builds_the_bundle_before_pytest.py",
    "framework/tests/test_sync_workflow_protects_what_the_script_protects.py",
    #   * Reads `contract/public_manifest.txt` (never mirrored, same as
    #     `test_mirrored_python_parses_on_311.py` above) to prove no published
    #     file names this machine's account. It is a check on the PRIVATE tree's
    #     posture before publishing; running it inside the artefact it licenses
    #     is the wrong side of the boundary.
    "framework/tests/test_public_files_name_no_operator_path.py",
    #   * Reads `arenas/*/runs/` and REFUSES a vacuous pass on zero records --
    #     the same design as `test_provenance_labels.py` above, and it fails for
    #     the same reason: `runs/` is never mirrored, so there is nothing to
    #     audit and the test says so rather than passing emptily.
    "framework/tests/test_record_tool_version_is_self_consistent.py",
    #   * ADDED 2026-09-27. Loads `scripts/check_docs_coverage.py`, which `scripts/`
    #     never mirrors; the documentation-drift gate runs HERE, before a sync.
    "framework/tests/test_docs_coverage_gate.py",
})

#: Files that contain the marker strings as VOCABULARY rather than as a
#: dependency: this module declares them, and its test exercises them. Without
#: this the rule condemns the rule — the same self-reference that made the theme
#: guard fire on its own documentation.
MARKER_VOCABULARY_FILES = frozenset({
    "framework/publish.py",
    "framework/tests/test_publish.py",
})

#: Belt-and-braces denylist applied AFTER the allowlist.
#:
#: `framework/gold/` and `framework/jats.py` are here because a blanket
#: `framework/**/*.py` allowlist swept them into the first real mirror build
#: (caught by reading the file list, 2026-08-07). `framework/gold/__init__.py`
#: describes itself as "the ONLY way an arena obtains ground truth" and warns
#: that "a player that can read this module can read the answer key" — it is the
#: gold-AUTHORING path, and publishing it publishes the route to every answer.
#: Both reach into the local `article-finder` skill.
#:
#: Note what did NOT catch this: the verbatim scanner. These files contain code
#: that *reads* gold, not gold text, so there was nothing to match. An allowlist
#: that is too broad and a content scan that only sees copied text leave exactly
#: this gap between them.
NEVER_PUBLISH = re.compile(
    r"(^|/)(_held_out|runs|node_modules|__pycache__|\.venv|\.git)(/|$)"
    r"|(^|/)\.private_seed$"
    r"|(^|/)_ground_truth\.json$"
    r"|(^|/)contributed/[^/]+/private(/|$)"
    r"|(^|/)\.env"
    r"|^framework/gold(/|$)"
    r"|^framework/jats\.py$"
    # The v2 citation-arena gold builder (2026-09-28): it parses the custody JATS
    # into held-out gold, so it is gold-AUTHORING code like framework/jats.py.
    r"|^framework/jats_citations\.py$"
    r"|^framework/tests/test_jats_citations\.py$"
)

#: Windows that DO overlap held-out material but are not answers.
#:
#: Each entry is a blake2b fingerprint, never the text: this module is itself
#: mirrored, so pasting the overlapping string here would put a fragment of
#: held-out material into the public repo — the very thing the scanner exists to
#: prevent. The reason is what a reviewer audits; the fingerprint is what the
#: code matches.
#:
#: Adding an entry is a deliberate act. It suppresses ONE exact window, so any
#: other overlap in the same file still fails.
ACKNOWLEDGED_OVERLAPS: dict[str, str] = {
    # arenas/code-translation-r-v1/tests/test_scorer.py — the arena's own
    # harvest CONTRACT (the R named-list shape a translated script must print).
    # It appears in the held-out R sources because every case must satisfy it.
    # Publishing the contract is the point of the arena; it reveals no answer.
    "d0cb6a0da66a5d6a4cba7e51": "code-translation-r-v1 harvest contract (public API, not gold)",
    # arenas/open-practices-repro-v1/catalogs/repo_templates.yaml — the stock
    # "data available from the corresponding author upon reasonable request"
    # sentence. It is a published journal cliché the arena GENERATES from, and
    # it is in the catalog precisely so tasks can be synthesised; a held-out
    # case containing it is a coincidence of the phrase being ubiquitous.
    "118dba40a0705026638354af": "open-practices-repro-v1 data-availability boilerplate (1/3)",
    "535aee089e12455d4f39fdbe": "open-practices-repro-v1 data-availability boilerplate (2/3)",
    "ac2083f169dd8ac49d82913b": "open-practices-repro-v1 data-availability boilerplate (3/3)",
}

#: Length of the verbatim run `scan_for_leaks` detects. Long enough that normal
#: shared boilerplate (licence headers, import blocks) does not collide, short
#: enough to catch a pasted paragraph of a held-out paper.
SHINGLE_CHARS = 96
SHINGLE_STRIDE = 24

_TEXT_SUFFIXES = {
    ".py", ".md", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".toml",
    ".cff", ".csv", ".tsv", ".r", ".R", ".sps", ".do", ".cfg", ".lock",
}


@dataclass
class ArenaPublishability:
    arena_id: str
    root: Path
    has_private_seed: bool
    gold_markers: list[str] = field(default_factory=list)

    @property
    def seed_says_public(self) -> bool:
        return self.has_private_seed

    @property
    def gold_says_public(self) -> bool:
        return not self.gold_markers

    @property
    def publishable(self) -> bool:
        return self.seed_says_public and self.gold_says_public

    @property
    def conditions_agree(self) -> bool:
        return self.seed_says_public == self.gold_says_public

    def why(self) -> str:
        if self.publishable:
            return "seeded, no real-paper gold dependency"
        bits = []
        if not self.has_private_seed:
            bits.append("no .private_seed (tasks are not seed-reproducible)")
        if self.gold_markers:
            bits.append(f"depends on real-paper gold via {sorted(set(self.gold_markers))}")
        return "; ".join(bits)


def code_only(src: str) -> str:
    """Python source with comments and docstrings removed.

    WHY THIS IS NEEDED
    ------------------
    Marker matching over raw text reads PROSE as if it were a dependency. Two
    arenas were misclassified as unpublishable by exactly that: the matched text
    in `prereg-deviation-v1/generator.py` is a docstring saying *"no
    article-finder registry needed"* — the guard fired on documentation that
    asserted the opposite of what it was accused of.

    String LITERALS are deliberately kept. `AF_DIR = Path.home() / ".claude" /
    "skills" / "article-finder"` is a real dependency expressed as a string, and
    stripping all strings would hide it. Only comments and standalone string
    expressions (docstrings) go.

    Over-stripping is the dangerous direction, which is what
    `test_code_only_keeps_a_real_marker_but_drops_prose` pins.

    IMPLEMENTATION NOTE — WHY SPANS, NOT A TOKEN JOIN
    -------------------------------------------------
    The first version rebuilt the source by joining surviving tokens with a
    space. That silently broke every DOTTED marker: `from framework.jats import
    parse_tables` came back as `from framework . jats import parse_tables`, so
    `"framework.jats"` no longer matched and three real-paper arenas were
    reported as having no gold dependency at all. The self-test missed it
    because its marker was a single string token, which a join preserves.

    So: erase the spans of removed tokens and leave every other byte exactly
    where it was. Nothing that survives can be reshaped by the stripper.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Unparseable source: fall back to the raw text. Failing OPEN here would
        # mean a syntactically broken file could smuggle a marker past the check.
        return src

    lines = src.splitlines(keepends=True)
    kill: list[tuple[int, int, int, int]] = []
    at_statement_start = True
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            kill.append((*tok.start, *tok.end))
            continue
        if tok.type == tokenize.STRING and at_statement_start:
            kill.append((*tok.start, *tok.end))  # docstring / bare string expression
            continue
        if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                        tokenize.DEDENT, tokenize.ENCODING):
            at_statement_start = True
            continue
        at_statement_start = False

    # Blank back-to-front so earlier spans keep their coordinates.
    for srow, scol, erow, ecol in reversed(kill):
        si, ei = srow - 1, erow - 1
        if si < 0 or ei >= len(lines):
            continue
        if si == ei:
            lines[si] = lines[si][:scol] + " " * (ecol - scol) + lines[si][ecol:]
        else:
            lines[si] = lines[si][:scol] + "\n"
            for i in range(si + 1, ei):
                lines[i] = "\n"
            lines[ei] = " " * ecol + lines[ei][ecol:]
    return "".join(lines)


def classify_arenas(arenas_root: Path) -> list[ArenaPublishability]:
    """Apply the predicate to every arena under ``arenas_root``."""
    out: list[ArenaPublishability] = []
    for arena in sorted(p for p in arenas_root.iterdir() if (p / "arena.yaml").is_file()):
        # The seed lives at task_sets/<version>/.private_seed, not the arena root.
        has_seed = any(arena.glob("task_sets/*/.private_seed"))
        markers: list[str] = []
        for py in arena.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            rel = py.relative_to(arena).as_posix()
            # A marker inside a file that is never mirrored cannot make the
            # ARENA unpublishable — the gold-authoring tool is private on its
            # own terms, and letting it condemn its whole arena would withhold
            # 21 seed-generated task families for no gain.
            if rel in PRIVATE_ARENA_PATHS:
                continue
            try:
                text = py.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            code = code_only(text)
            markers.extend(m for m in PRIVATE_GOLD_MARKERS if m in code)
        out.append(ArenaPublishability(arena.name, arena, has_seed, markers))
    return out


#: Committed list of arenas that exist in the public mirror.
#:
#: The Next.js app needs this and CANNOT compute it: publishability turns on
#: `.private_seed`, which is gitignored and therefore absent from the Vercel
#: build. Without it the site deep-linked every arena into the public repo, and
#: the six private ones 404'd the moment REPO_URL was switched on — a link the
#: site could not back, which is the exact defect this whole stream exists to
#: remove. `test_publish.py` fails if this file drifts from the computed set.
PUBLISHED_ARENAS_FILE = "contract/published_arenas.json"


def published_arena_ids(arenas_root: Path) -> list[str]:
    """Sorted ids of arenas that appear in the public mirror."""
    return sorted(a.arena_id for a in classify_arenas(arenas_root) if a.publishable)


def imports_a_private_package(path: Path) -> str | None:
    """Name of the private package a file imports at module scope, else None.

    MODULE-LEVEL AND UNGUARDED ONLY. An import inside a function, or inside a
    `try: ... except ImportError:`, is a deliberate optional dependency —
    `framework/runner.py` imports `players.adapters` exactly that way so the
    in-repo adapters load in a checkout and are simply absent from an install.
    Flagging those would exclude the runner itself.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return None
    for node in tree.body:  # tree.BODY: top level only, so no try/except or def
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        for name in names:
            if name in PRIVATE_PACKAGES:
                return name
    return None


def public_manifest(repo_root: Path) -> list[Path]:
    """Repo-relative paths that make up the public mirror, sorted."""
    picked: set[Path] = set()

    for pattern in PUBLIC_TOP_LEVEL:
        for p in repo_root.glob(pattern):
            if p.is_file():
                picked.add(p.relative_to(repo_root))

    arenas_root = repo_root / "arenas"
    if arenas_root.is_dir():
        for arena in classify_arenas(arenas_root):
            if not arena.publishable:
                continue
            for pattern in PUBLIC_ARENA_GLOBS:
                for p in arena.root.glob(pattern):
                    if not p.is_file():
                        continue
                    # PRIVATE_ARENA_PATHS was previously consulted only when
                    # CLASSIFYING an arena, never here — so `tools/build_gold.py`
                    # was excluded from condemning its arena and then cheerfully
                    # mirrored. Both call sites must honour it.
                    if p.relative_to(arena.root).as_posix() in PRIVATE_ARENA_PATHS:
                        continue
                    picked.add(p.relative_to(repo_root))

    kept = [
        p for p in picked
        if not NEVER_PUBLISH.search(p.as_posix())
        # A file that cannot import in the published package must not ship.
        and not (p.suffix == ".py" and imports_a_private_package(repo_root / p))
        and p.as_posix() not in PRIVATE_TEST_FILES
    ]
    return sorted(kept)


def _fingerprint(window: str) -> str:
    """Stable digest of one window. Stable ACROSS PROCESSES — see _shingles."""
    return hashlib.blake2b(window.encode("utf-8"), digest_size=12).hexdigest()


def _shingles(text: str, stride: int) -> set[str]:
    """Hashed fixed-length character windows of whitespace-normalised text.

    STRIDE IS NOT SYMMETRIC, AND THAT MATTERS
    -----------------------------------------
    Both sides were originally indexed at the same coarse stride, which meant a
    copied passage was only detected when it happened to land at a multiple of
    that stride. A paste one character off — e.g. wrapped in `SAMPLE = \"\"\"…` —
    produced ZERO overlap. The scanner would have reported clean on the exact
    accident it exists to catch.

    So the candidate side is indexed at stride 1 (every window) and the held-out
    side at the coarse stride. Then any verbatim run of at least
    ``SHINGLE_CHARS + SHINGLE_STRIDE - 1`` characters must contain at least one
    whole coarse window, and that window is guaranteed to be in the candidate's
    stride-1 set regardless of alignment.

    The digest is blake2b rather than the builtin `hash()`: `hash()` is salted
    per process, so an acknowledged-overlap allowlist keyed on it would silently
    stop matching on the next run — and an allowlist that stops matching turns
    into a build failure nobody can reproduce.
    """
    flat = " ".join(text.split())
    if len(flat) < SHINGLE_CHARS:
        return set()
    return {
        _fingerprint(flat[i:i + SHINGLE_CHARS])
        for i in range(0, len(flat) - SHINGLE_CHARS + 1, stride)
    }


#: Shortest verbatim run the scan is guaranteed to catch, at any alignment.
MIN_DETECTED_RUN = SHINGLE_CHARS + SHINGLE_STRIDE - 1


def held_out_shingles(repo_root: Path) -> tuple[set[str], int]:
    """Fingerprints of every held-out artifact, plus how many files fed them.

    The count is returned so callers can refuse a VACUOUS scan: matching zero
    files against an empty fingerprint set proves nothing, and would report a
    clean bill of health on a repo whose held-out corpora had simply been moved.
    """
    shingles: set[str] = set()
    n_files = 0
    for held_out_dir in repo_root.glob("arenas/*/task_sets/*/_held_out"):
        for p in held_out_dir.rglob("*"):
            if not p.is_file():
                continue
            n_files += 1
            try:
                shingles |= _shingles(p.read_text(encoding="utf-8", errors="replace"), SHINGLE_STRIDE)
            except OSError:
                continue

    # THE PAPERS THEMSELVES ARE NO LONGER ON DISK, and they used to be counted
    # above. article-finder took custody of them on 2026-09-05; each paper dir
    # keeps a `source.json` naming the exact bytes by content hash.
    #
    # Resolving them here is not tidiness. Measured the day they were deleted,
    # this set fell from ~3.42M shingles over 143 files to 437k over 107 -- 12.8%
    # retained -- and the gate still reported a clean build, because the vacuity
    # guard only refuses a count of ZERO and 107 walks straight past it. Article
    # PROSE was never at risk (`jats.xml` and `gold.json` stayed, and a pasted
    # paragraph is the realistic accident this scan backstops); what was lost was
    # a held-out paper's own bytes reaching a mirrored file. A gate that quietly
    # covers less than it did is worse than one that never covered it.
    #
    # This RAISES rather than skipping when custody cannot answer: emitting a
    # mirror against a silently smaller fingerprint set is the failure mode this
    # whole function is shaped to refuse. The population affected is unchanged --
    # a checkout without the article repository is the same checkout that has no
    # held-out corpora, and the build already refuses there.
    from framework.article_custody import SOURCE_RECORD, held_out_pdf_bytes

    for held_out_dir in repo_root.glob("arenas/*/task_sets/*/_held_out"):
        for record in held_out_dir.rglob(SOURCE_RECORD):
            data = held_out_pdf_bytes(record.parent)
            if data is None:
                continue
            n_files += 1
            shingles |= _shingles(data.decode("utf-8", errors="replace"), SHINGLE_STRIDE)
    # Seeds are short, so they are matched verbatim rather than by shingle.
    return shingles, n_files


def private_seed_values(repo_root: Path) -> set[str]:
    """Every private seed's literal content. Any occurrence anywhere is a leak."""
    values = set()
    for p in repo_root.glob("arenas/*/task_sets/*/.private_seed"):
        try:
            v = p.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if v:
            values.add(v)
    return values


#: The two halves of the leak gate, split by what they NEED to run.
#:
#: "allowlist" — the path/marker half. Needs only the repo, so it runs anywhere,
#:               including CI, and protects every PR.
#: "all"       — that half PLUS the verbatim held-out scan, which needs the
#:               private corpora and therefore only runs where they exist
#:               (a dev machine, the pre-push hook).
#:
#: Why a mode flag and not "skip the scan when the corpora are absent": absence
#: is ALSO what a misconfigured arenas root looks like, so an auto-skip would
#: reintroduce exactly the false green `test_leak_scan_is_not_vacuous` exists to
#: prevent. The caller must SAY which gate it is running; "all" still refuses to
#: certify when it finds nothing to compare against.
#: Account names too generic to identify anyone. A CI runner's home is
#: `/home/runner`; flagging "runner" would only ever produce false alarms.
GENERIC_ACCOUNT_NAMES = frozenset({
    "root", "runner", "user", "admin", "administrator", "ubuntu", "vscode",
    "circleci", "default", "public", "shared", "testuser", "student",
})


def personal_identity(repo_root: Path) -> tuple[set[str], set[str]]:
    """(account names, email addresses) that identify whoever builds the mirror.

    Read from THIS machine at run time -- the OS account and home directory name,
    and git's configured user.name / user.email -- so no real name is written in
    a published file for the check to be able to find it. On the owner's machine
    this is the owner; on anyone else's it is them, which is the right answer in
    both places.
    """
    import getpass
    import subprocess

    names: set[str] = set()
    try:
        names.add(Path.home().name)
    except RuntimeError:  # no HOME/USERPROFILE; getuser() below may still answer
        pass
    try:
        names.add(getpass.getuser())
    except Exception:
        pass
    if not names:
        # Refuse to certify rather than scan for nobody: with no account name the
        # path half of the gate would match nothing and read as a clean pass.
        raise RuntimeError(
            "personal-path gate: could not read this machine's account name "
            "(Path.home() and getpass.getuser() both failed); refusing to certify"
        )
    emails: set[str] = set()
    for key, bucket in (("user.name", names), ("user.email", emails)):
        try:
            out = subprocess.run(["git", "config", key], cwd=str(repo_root),
                                 capture_output=True, text=True, timeout=10).stdout or ""
        except Exception:
            out = ""
        value = out.strip()
        if not value:
            continue
        if key == "user.name":
            bucket.update(value.split())
        else:
            bucket.add(value)
    names = {n for n in names
             if len(n) >= 3 and n.lower() not in GENERIC_ACCOUNT_NAMES}
    return names, {e for e in emails if "@" in e}


def personal_path_hits(text: str, names: set[str], emails: set[str]) -> list[str]:
    r"""Personal identifiers in `text`: a home-directory PATH naming one of
    `names` (`C:/Users/<n>`, `C:\Users\<n>`, `/home/<n>`, `/c/Users/<n>`),
    or any of `emails` verbatim. Case-insensitive.

    Path-shaped on purpose: a bare account name is also a substring of ordinary
    words ("profiling", "filing"), and a bare-substring rule would fire on clean
    files until someone turned it off.
    """
    hits: list[str] = []
    for n in sorted(names):
        pat = re.compile(r"(?:Users|home)[\\/]+" + re.escape(n) + r"(?![A-Za-z0-9_.-])",
                         re.IGNORECASE)
        hits += sorted({m.group(0) for m in pat.finditer(text)})
    low = text.lower()
    hits += sorted(e for e in emails if e.lower() in low)
    return hits


GATE_MODES = ("all", "allowlist")


def scan_for_leaks(repo_root: Path, files: list[Path], *, gates: str = "all") -> list[str]:
    """Reasons the given files must NOT be mirrored. Empty list == clean.

    WHAT THIS CATCHES
      * any file containing a private seed value verbatim;
      * any file sharing a {MINRUN}-character verbatim run (at ANY alignment)
        with a held-out
        artifact — i.e. a paragraph of a held-out paper pasted into a scorer
        fixture, an arena.yaml, or a comment, none of which a path rule sees.

    WHAT IT DOES NOT CATCH
      * paraphrase, translation, or summarisation of held-out content;
      * a gold ANSWER that never appears in a `_held_out/` file (e.g. one only
        ever computed at run time);
      * verbatim runs shorter than {MINRUN} characters.

    It is a backstop for copy-paste, which is the realistic accident. It is not
    a proof of non-disclosure, and the docs must not claim it is.
    """
    if gates not in GATE_MODES:
        raise ValueError(f"gates must be one of {GATE_MODES}, got {gates!r}")
    reasons: list[str] = []
    seeds = private_seed_values(repo_root)
    ho_shingles, ho_files = held_out_shingles(repo_root)

    # A module that REACHES the gold path is as dangerous as gold itself, and
    # the verbatim scan below cannot see it: such a file contains code that
    # reads the answer key, not the answer key. `framework/gold/` reached the
    # first mirror build through a blanket `framework/**/*.py` allowlist, and
    # only a human reading the file list caught it. So the marker check that
    # already governs ARENAS now runs over every mirrored Python file too.
    for rel in files:
        if rel.suffix != ".py" or rel.as_posix() in MARKER_VOCABULARY_FILES:
            continue
        try:
            code = code_only((repo_root / rel).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        hits = sorted({m for m in PRIVATE_GOLD_MARKERS if m in code})
        if hits:
            reasons.append(
                f"{rel}: reaches the real-paper gold path ({', '.join(hits)}). Publishing a "
                f"module that resolves gold publishes the route to the answer key, even "
                f"though the file contains no gold text for the verbatim scan to find."
            )

    # A home-directory path or email that identifies whoever is building the
    # mirror. `repo_templates.yaml` carried the owner's real Windows profile path
    # as a "synthetic" bad-path example and this gate could not
    # see it: it compares against seeds and held-out text, and a personal path is
    # neither. Runs in both gate modes -- it needs no corpora.
    names, emails = personal_identity(repo_root)
    for rel in files:
        p = repo_root / rel
        if p.suffix not in _TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found = personal_path_hits(text, names, emails)
        if found:
            reasons.append(
                f"{rel}: names the person building this mirror ({len(found)} "
                f"identifier(s): a home-directory path or email). Replace it with an "
                f"invented placeholder of the same shape."
            )

    # The path/marker half is complete. In "allowlist" mode that is the whole
    # contract, so return WITHOUT running the non-vacuity refusal below — which
    # would otherwise fire on every machine that legitimately has no corpora
    # (CI), and which is the reason this gate was red on every push for a week.
    # The caller asked for the half that does not need the corpora; it gets
    # exactly that, and the printed mode says so.
    if gates == "allowlist":
        return reasons

    # Non-vacuity. A scan that examined nothing must not report "clean" —
    # that is the exact shape of false confidence this project keeps re-learning.
    if not seeds and not ho_shingles:
        reasons.append(
            "REFUSING TO CERTIFY: found no private seeds and no held-out material to "
            "scan against. Either the arenas root is wrong or the corpora moved — "
            "a scan with nothing to compare cannot clear anything."
        )
        return reasons

    for rel in files:
        p = repo_root / rel
        if p.suffix not in _TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            reasons.append(f"{rel}: unreadable ({exc}) — cannot certify, so not mirrored")
            continue
        for seed in seeds:
            if seed in text:
                reasons.append(f"{rel}: contains a .private_seed value verbatim")
        if ho_shingles:
            # Stride 1 on the candidate: see _shingles for why symmetry here
            # silently defeats the whole check.
            overlap = (_shingles(text, 1) & ho_shingles) - set(ACKNOWLEDGED_OVERLAPS)
            if overlap:
                reasons.append(
                    f"{rel}: shares {len(overlap)} verbatim {SHINGLE_CHARS}-char window(s) "
                    f"with held-out material. If a window is shared BOILERPLATE rather "
                    f"than an answer, add its fingerprint to ACKNOWLEDGED_OVERLAPS with "
                    f"a reason: {sorted(overlap)[:3]}"
                )
    return reasons


scan_for_leaks.__doc__ = (scan_for_leaks.__doc__ or "").replace("{MINRUN}", str(MIN_DETECTED_RUN))

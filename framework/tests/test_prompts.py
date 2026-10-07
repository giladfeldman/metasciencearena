"""The instrument is published, so the instrument has to be pinned.

Prompt templates, `players/registry.yaml` and the adapters became public on
2026-08-15 (user directive: "our prompts and everything about the way we call the
models/tools should be shared"). Publishing them turns three things that used to
be private conventions into public claims:

  * that no credential is written down anywhere in the registry;
  * that a published score's `prompt_template_sha256` names text a reader can
    actually go and read;
  * that every player in one arena was asked the SAME question, so the ranking
    between them means something.

Each is asserted here against the real repo, not against a fixture.
"""
from __future__ import annotations

import hashlib

from framework.runner import normalise_newlines
import json
import re
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO / "players" / "prompts"
ARCHIVE_DIR = PROMPT_DIR / "_archive"
REGISTRY = REPO / "players" / "registry.yaml"

#: A YAML key whose NAME looks like it holds a credential.
_CREDENTIAL_KEY = re.compile(r"(key|token|secret|password|passwd|credential)", re.I)
#: What an env-var NAME looks like. A real credential does not.
_ENV_VAR_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _registry() -> list[dict]:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def _walk(node, path=""):
    """Yield (dotted_path, key, value) for every mapping entry, recursively."""
    if isinstance(node, dict):
        for k, v in node.items():
            here = f"{path}.{k}" if path else str(k)
            yield here, str(k), v
            yield from _walk(v, here)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def test_registry_declares_credentials_by_env_var_name_never_by_value():
    """A published registry must name where a secret LIVES, not what it is.

    `players/registry.yaml` is now mirrored to a public repo and shipped in the
    wheel. The convention has always been `*_key_env: PROVIDER_API_KEY` — the
    NAME of an environment variable, resolved at run time — but nothing enforced
    it, and the cost of one `api_key: sk-...` slipping in is a live credential in
    public git history, which is not something a later commit can take back.
    """
    offenders: list[str] = []
    for dotted, key, value in _walk(_registry()):
        if not _CREDENTIAL_KEY.search(key) or not isinstance(value, str):
            continue
        if not key.endswith("_env"):
            offenders.append(
                f"{dotted}: field name does not end in `_env`, so it reads as a "
                f"literal credential rather than a variable name"
            )
        elif not _ENV_VAR_NAME.match(value):
            offenders.append(
                f"{dotted} = {value[:12]!r}...: not an environment variable NAME "
                f"(expected ^[A-Z][A-Z0-9_]*$) — this looks like an actual secret"
            )
    assert offenders == [], (
        "players/registry.yaml is PUBLIC. These entries look like they carry a "
        "credential value rather than an env var name:\n  " + "\n  ".join(offenders)
    )


def test_the_credential_check_actually_fires(tmp_path, monkeypatch):
    """Prove the scan catches a planted secret rather than trusting that it would."""
    planted = [{"player_id": "x", "openai_api_key": "sk-live-abcdef0123456789"}]
    monkeypatch.setattr(
        "framework.tests.test_prompts._registry", lambda: planted
    )
    with pytest.raises(AssertionError, match="reads as a literal credential"):
        test_registry_declares_credentials_by_env_var_name_never_by_value()


#: True when this checkout ships run records at all.
#:
#: `runs/` is matched by `publish.NEVER_PUBLISH`, so the public mirror never has
#: any — and four checks below are about what the RECORDS say, which is a
#: question the mirror structurally cannot ask. Skipping on that is not the
#: auto-skip-on-absence this project keeps rejecting: the condition is "no
#: arena has a runs/ directory", a fact about the TREE, and it is deliberately
#: NOT "no record carried a hash", which is the vacuity each test still fails on
#: where runs/ does exist.
_HAS_RUNS = any(p.is_dir() for p in REPO.glob("arenas/*/runs"))
_NO_RUNS_REASON = (
    "this checkout ships no arenas/*/runs/ at all (the public mirror never does — "
    "framework.publish.NEVER_PUBLISH excludes them), so there are no published "
    "records to check. Where runs/ exists, these tests still fail on an empty scan."
)
requires_run_records = pytest.mark.skipif(not _HAS_RUNS, reason=_NO_RUNS_REASON)


def _template_hash_variants(raw: bytes) -> set[str]:
    """Every sha16 this template could legitimately be recorded under.

    Three forms, because the working tree differs by platform and BOTH the
    producer and this check read the working tree:
      * LF-normalised  - what `runner._prompt_template_sha` stamps from 2026-08-19
      * CRLF           - what it stamped on Windows before that (autocrlf)
      * raw as on disk - belt and braces for a mixed-ending file

    The CRLF form must be COMPUTED, not merely read: on a Linux checkout the
    file is LF, so the legacy hash cannot be reproduced from the bytes present.
    A shim that only hashed what it found passed on Windows and failed in CI -
    which is the same platform-dependence it exists to absorb.

    Delete the legacy forms once no published record predates the fix.
    """
    lf = normalise_newlines(raw)
    crlf = lf.replace(b"\n", b"\r\n")
    return {hashlib.sha256(b).hexdigest()[:16] for b in (lf, crlf, raw)}


def _prompt_hashes() -> dict[str, Path]:
    """sha256[:16] -> file, over live AND archived templates.

    BOTH the normalised hash and the raw-bytes hash are indexed, and the second
    one is a compatibility shim with a stated end condition.

    Until 2026-08-19 `runner._prompt_template_sha` hashed raw working-tree bytes,
    so every record produced on Windows (autocrlf) stamped the CRLF hash while
    the committed blob — what CI, the public mirror and every other reader checks
    out — is LF. Measured on `power_reporting.txt`: records claim
    `c02b588da306ed86`; the file in git hashes to `3d89484f4959b236`. This test
    is exactly where that surfaced: CI reported 11 arenas' published hashes as
    naming templates that do not exist, while the same test passed on the machine
    that wrote them.

    New records stamp the normalised hash. The raw entry exists only so
    already-published records still resolve; it can be dropped once no run record
    predates the fix.
    """
    out: dict[str, Path] = {}
    for p in list(PROMPT_DIR.glob("*.txt")) + list(ARCHIVE_DIR.glob("*.txt")):
        for sha in _template_hash_variants(p.read_bytes()):
            out.setdefault(sha, p)
    return out


def _record_prompt_hashes() -> dict[tuple[str, str], dict[str, set[str]]]:
    """(arena, task_set) -> {prompt_sha: {player_id, ...}} over published runs.

    `_archive/` and `_pilot_archive/` run directories are skipped: they hold
    superseded runs kept as evidence, and `build-data.mjs` never reads them.
    """
    out: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for path in REPO.glob("arenas/*/runs/*/*.jsonl"):
        if any(part.startswith("_") for part in path.parts):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sha = (rec.get("provenance") or {}).get("prompt_template_sha256")
            if sha:
                key = (rec.get("arena_id", "?"), rec.get("task_set_version", "?"))
                out[key][sha].add(rec.get("player_id", "?"))
    return out


@requires_run_records
def test_every_published_prompt_hash_resolves_to_a_prompt_file():
    """Provenance that points at nothing is worse than no provenance.

    Every record stamps `provenance.prompt_template_sha256`. The field is only
    worth recording if a reader can fetch the text it names, so a template is
    never edited in place and discarded — the previous bytes go to
    `players/prompts/_archive/<name>.<sha16>.txt`. This is what makes that a rule
    instead of an intention.
    """
    known = _prompt_hashes()
    # Counted as FILES, not as distinct hashes: `codex.txt` and `gemini.txt` are
    # byte-identical, so a hash count reads 24 for 25 files and a floor of 25
    # fails for a reason that has nothing to do with coverage.
    n_files = len(list(PROMPT_DIR.glob("*.txt")))
    assert n_files >= 25, (
        f"only {n_files} prompt templates found — the glob is wrong, and this "
        f"check would pass vacuously"
    )
    seen = _record_prompt_hashes()
    assert seen, "no published record carries a prompt hash — check would be vacuous"

    missing: list[str] = []
    for (arena, version), by_sha in sorted(seen.items()):
        for sha, players in sorted(by_sha.items()):
            if sha not in known:
                missing.append(
                    f"{arena}/{version}: {sha} (players: {', '.join(sorted(players))})"
                )
    assert missing == [], (
        "these published records name a prompt template that no longer exists in "
        "players/prompts/ or players/prompts/_archive/:\n  " + "\n  ".join(missing)
        + "\n\nA template was edited in place. Restore the previous bytes to "
          "_archive/<name>.<sha16>.txt — deleting the text a published score was "
          "measured with makes that score unauditable."
    )


def _group_players_by_template(by_sha: dict, known: dict) -> dict:
    """Collapse recorded hashes onto the TEMPLATE FILE each one identifies.

    A record's hash may be either digest variant (see `_prompt_hashes`), and both
    resolve to the same Path. Grouping on the raw string instead would report one
    unedited template as two instruments for any arena holding records from both
    sides of the 2026-08-19 normalisation fix.
    """
    grouped: dict = {}
    for sha, players in by_sha.items():
        key = known.get(sha)
        key = key if key is not None else Path(sha)
        grouped.setdefault(key, set()).update(players)
    return grouped


@requires_run_records
def test_one_arena_and_task_set_never_mixes_prompt_templates():
    """Players in one arena must have been asked the same question.

    The leaderboard ranks players against each other inside an arena. If two of
    them were sent different instructions, the ranking is measuring the prompt as
    much as the player — and nothing on the page would show it.

    So changing a template is not a local edit: it obliges a re-run of every
    player in that arena that uses it. This is the check that makes the
    obligation visible instead of leaving it to memory. It went red on purpose
    when `prereg_deviation.txt` was edited on 2026-08-15, and green again only
    once all four affected players had been re-run.

    Players carrying NO hash are ignored: `regcheck-*` is a third-party tool with
    its own internal prompts, which is a legitimate difference in kind rather
    than an uncontrolled variable.
    """
    known = _prompt_hashes()
    mixed: list[str] = []
    for (arena, version), by_sha in sorted(_record_prompt_hashes().items()):
        # Group by FILE, not by hash string: one template can legitimately appear
        # under two digests across the 2026-08-19 normalisation fix.
        by_file = _group_players_by_template(by_sha, known)
        if len(by_file) <= 1:
            continue
        detail = "; ".join(
            f"{path.name} <- {', '.join(sorted(players))}"
            for path, players in sorted(by_file.items(), key=lambda kv: kv[0].name)
        )
        mixed.append(f"{arena}/{version}: {len(by_file)} templates — {detail}")
    assert mixed == [], (
        "these arenas rank players that were sent DIFFERENT prompt templates, so "
        "the comparison between them is confounded by the instrument:\n  "
        + "\n  ".join(mixed)
        + "\n\nRe-run the players still on the old template, or archive the run "
          "under runs/<v>/_archive/ if it is superseded."
    )


@requires_run_records
def test_published_records_were_measured_with_the_CURRENT_template():
    """The leaderboard must show measurements taken with the live instrument.

    `test_one_arena_and_task_set_never_mixes_prompt_templates` catches players
    being compared under different prompts. It cannot catch the case where the
    template was edited and NOBODY was re-run: every record then agrees with
    every other, and all of them disagree with the file on disk. The board keeps
    publishing numbers produced by an instrument the repo no longer contains.

    That is the state this repo was briefly in on 2026-08-15 between editing
    `prereg_deviation.txt` and finishing the re-run — deliberately, and this test
    is what said so. It stayed red until all four affected players were re-run.

    SCOPE: each arena's CURRENT task set. A superseded or frozen task set was
    measured with the prompt of its own era by definition -- publication-bias-v1
    v1 asked for a verdict, v2 asks for statistics, so v1's prompt is archived,
    not live -- and the runner refuses to add records to it. Its records must
    still name a template that EXISTS, which
    `test_every_published_prompt_hash_resolves_to_a_prompt_file` enforces; they
    cannot name the live one. Scoping here, not exempting: a current task set
    measured under an edited template still turns this red.
    """
    from framework.discovery import resolve_task_set_version

    def _current(arena: str) -> str | None:
        manifest = REPO / "arenas" / arena / "arena.yaml"
        if not manifest.is_file():
            return None
        return resolve_task_set_version(yaml.safe_load(manifest.read_text(encoding="utf-8")) or {})

    live = {p: hashlib.sha256(normalise_newlines(p.read_bytes())).hexdigest()[:16]
            for p in PROMPT_DIR.glob("*.txt")}
    assert live, "no live templates found — check would be vacuous"
    # Accept the pre-2026-08-19 raw-bytes hash too; see _prompt_hashes().
    live_hashes = set()
    for f in PROMPT_DIR.glob("*.txt"):
        live_hashes |= _template_hash_variants(f.read_bytes())

    stale: list[str] = []
    checked = 0
    for (arena, version), by_sha in sorted(_record_prompt_hashes().items()):
        if version != _current(arena):
            continue
        checked += 1
        for sha, players in sorted(by_sha.items()):
            if sha not in live_hashes:
                stale.append(
                    f"{arena}/{version}: {sha} (players: {', '.join(sorted(players))})"
                )
    assert checked >= 10, f"only {checked} current task sets carry prompt hashes — scoping broke the check"
    assert stale == [], (
        "these published records were measured with a template that has since "
        "been edited, so the leaderboard is showing scores from an instrument "
        "this repo no longer has:\n  " + "\n  ".join(stale)
        + "\n\nRe-run the affected players against the current template, or move "
          "the superseded run to runs/<v>/_archive/ if it is no longer published."
    )


def _record_field(field: str) -> dict[tuple[str, str], dict[object, set[str]]]:
    """(arena, task_set) -> {provenance[field]: {player_id, ...}} over published runs."""
    out: dict[tuple[str, str], dict[object, set[str]]] = defaultdict(lambda: defaultdict(set))
    for path in REPO.glob("arenas/*/runs/*/*.jsonl"):
        if any(part.startswith("_") for part in path.parts):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = (rec.get("provenance") or {}).get(field)
            if value is not None:
                key = (rec.get("arena_id", "?"), rec.get("task_set_version", "?"))
                out[key][value].add(rec.get("player_id", "?"))
    return out


@requires_run_records
def test_one_arena_and_task_set_never_mixes_trial_counts():
    """The prompt is not the only part of the instrument that must be uniform.

    `framework run` defaults to `--trials 3`, and every published run in this
    repo was made with `--trials 1`. A re-run script that simply omits the flag
    therefore measures a DIFFERENT thing — a best-of-3 style average against
    everyone else's single shot — and the only visible symptom is a record count
    that looks like a duplication bug.

    That is exactly what happened on 2026-08-15: a prereg-deviation-v1 re-run
    silently switched two of six players to three trials. Nothing failed; the
    file just had 94 records for 46 tasks. Caught by reading the data, which is
    not a control. This is the control.
    """
    mixed: list[str] = []
    for (arena, version), by_trials in sorted(_record_field("trials").items()):
        if len(by_trials) <= 1:
            continue
        detail = "; ".join(
            f"trials={t}: {', '.join(sorted(players))}"
            for t, players in sorted(by_trials.items(), key=lambda kv: str(kv[0]))
        )
        mixed.append(f"{arena}/{version} — {detail}")
    assert mixed == [], (
        "these arenas compare players that were each given a DIFFERENT number of "
        "attempts per task, so the ranking partly measures the run configuration "
        "rather than the players:\n  " + "\n  ".join(mixed)
        + "\n\nRe-run the odd ones out with the same `--trials` as the rest "
          "(every published run in this repo uses `--trials 1`)."
    )


def test_every_prompt_template_interpolates_its_input():
    """One prompt serves both halves of the benchmark; only the INPUT differs.

    This is the property that makes publishing the instrument possible at all.
    The public and private splits are the same task asked of different material,
    so the template must receive its material through an interpolation point
    (`{{INPUT_TEXT}}`, `{{PDF_PATH}}`, `{{N_PAGES}}`) rather than carry any of it
    inline. A template with no placeholder is either dead or has the input baked
    in, and a baked-in input means the private half is a DIFFERENT task while the
    published prompt says otherwise.

    Checked against the registry rather than the directory, so a template no
    player uses cannot satisfy it by existing.

    NOT checked here: whether a DOI or filename appears anywhere in the text. An
    earlier draft did that and flagged four *illustrative* examples — the format
    specimens in `prereg_extraction.txt` and `reference_integrity.txt` that show
    the model what a DOI looks like. Those are instructions, not inputs, and a
    rule that cannot tell them apart would push us to make the prompts worse.
    """
    placeholder = re.compile(r"\{\{[A-Z0-9_]+\}\}")
    referenced = sorted({
        e["prompt_template_path"] for e in _registry() if e.get("prompt_template_path")
    })
    assert len(referenced) >= 20, (
        f"only {len(referenced)} templates are referenced by the registry — the "
        f"key name changed and this check would pass vacuously"
    )

    offenders: list[str] = []
    for rel in referenced:
        p = REPO / rel
        if not p.is_file():
            offenders.append(f"{rel}: referenced by the registry but does not exist")
            continue
        if not placeholder.search(p.read_text(encoding="utf-8")):
            offenders.append(
                f"{rel}: contains no {{{{PLACEHOLDER}}}} — nothing marks where the "
                f"task input goes, so either the input is baked in or the template "
                f"is not the one actually sent"
            )
    assert offenders == [], "\n  ".join(["published prompt templates:"] + offenders)


def test_two_digest_variants_of_one_template_are_not_a_mix():
    """REGRESSION (2026-08-20). Both digests of one file are one instrument.

    `runner._prompt_template_sha` normalised line endings on 2026-08-19, so an
    arena holding records from both sides carries two hashes for an UNEDITED
    template. Grouping on the hash string reported that as a confounded ranking
    and told the operator to re-run or archive sound runs.

    Uses the real grim-consistency-v1 values. Both digests are stamped on
    published records of this arena — `72e1ad7a...` on the pre-fix runs
    (`nvidia-nemotron-lightning-grim__revealed__lightning`) and `2b783548...` on
    the post-fix ones (`local-gpt-oss-20b-q8-grim__revealed__local-pilot`) — and
    they resolve to one unedited file, which is the whole point.

    The CRLF form is COMPUTED, not read off disk. `.gitattributes` pins
    `players/prompts/** text eol=lf`, so a Linux checkout holds no CRLF at all;
    hashing the raw bytes there produced two IDENTICAL digests and this test's
    own anti-vacuity guard fired, red on CI for a file that is exactly as
    intended. `_template_hash_variants` documents the same trap one screen up.
    """
    import hashlib
    from framework.runner import normalise_newlines

    path = PROMPT_DIR / "grim_consistency.txt"
    lf = normalise_newlines(path.read_bytes())
    crlf = lf.replace(b"\n", b"\r\n")
    legacy = hashlib.sha256(crlf).hexdigest()[:16]   # what Windows stamped pre-fix
    current = hashlib.sha256(lf).hexdigest()[:16]    # what the runner stamps now
    assert legacy != current, (
        f"{path.name} contains no newline, so its two digest variants collapse "
        f"into one and this test would pass vacuously; pick a template that has "
        f"line breaks"
    )

    known = _prompt_hashes()
    assert known.get(legacy) == path and known.get(current) == path, \
        "_prompt_hashes must index BOTH digest variants onto the same file"

    by_sha = {legacy: {"player-run-before-the-fix"},
              current: {"player-run-after-the-fix"}}
    grouped = _group_players_by_template(by_sha, known)
    assert len(grouped) == 1, f"one template reported as {len(grouped)} instruments"
    assert grouped[path] == {"player-run-before-the-fix", "player-run-after-the-fix"}


def test_genuinely_different_templates_are_still_a_mix():
    """The guard must not have been loosened into uselessness."""
    known = _prompt_hashes()
    a = PROMPT_DIR / "grim_consistency.txt"
    b = PROMPT_DIR / "power_reporting.txt"
    sha_a = next(k for k, v in known.items() if v == a)
    sha_b = next(k for k, v in known.items() if v == b)
    grouped = _group_players_by_template({sha_a: {"p1"}, sha_b: {"p2"}}, known)
    assert len(grouped) == 2, "two different templates must still count as a mix"


def test_the_run_default_trials_matches_what_every_published_record_uses():
    """WATCHED FAIL FIRST on 2026-08-28, when the default was 3.

    3,585 of the 3,605 published records carry provenance.trials = 1 and not one
    carries any other value, so a default of 3 matched nothing ever published.
    It is a trap and not a preference: a `deterministic: true` player collapses
    trials to 1 by itself, so the default only reaches NON-deterministic players
    -- exactly the LLMs -- and a mixed run silently pairs trials=1 tools against
    trials=3 models. `test_one_arena_and_task_set_never_mixes_trial_counts` then
    fails AFTER the quota has been spent, which is how this was found.
    """
    import argparse
    from framework import cli

    parser_holder = {}
    real_parser = argparse.ArgumentParser

    p = cli.build_parser() if hasattr(cli, "build_parser") else None
    if p is None:
        # The parser is built inside main(); read the literal instead of
        # reconstructing it, so this test cannot drift from the source.
        src = (Path(cli.__file__)).read_text(encoding="utf-8")
        assert '"--trials", type=int, default=1' in src, (
            "framework run's --trials default is not 1; every published record uses 1")
        assert '"--trials", type=int, default=3' not in src
        return
    ns = p.parse_args(["run", "--arena", "a", "--task-set", "v1", "--players", "p"])
    assert ns.trials == 1, f"--trials defaults to {ns.trials}, but every published record uses 1"

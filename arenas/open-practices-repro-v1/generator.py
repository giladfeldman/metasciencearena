"""Generator for open-practices-repro-v1 (metacheck-style repo/code arena).

Builds MOCKED repository snapshots (no live OSF/GitHub fetch) with KNOWN
reproducibility defects injected per file (or at the repo-url level), mirroring
metacheck's repo_check / code_check modules. All-procedural and deterministic
from (task_set_version, seed).

Dual-benchmark (revealed/private): both splits run the IDENTICAL tier matrix and
assign injected-issue KINDS deterministically (index-driven cycling through ALL
kinds, seed-independent), so every split covers the full array of injected
defects equally — this is what makes framework/parity.py pass. Only the concrete
file contents / paths / urls are seed-driven, so revealed vs private content
still differ.

Gold is regenerated from the seed and served from the in-process cache (no
external registry): the secret is the private seed, not a stored answer key. The
revealed seed is committed in arena.yaml#benchmark_splits.

INPUT  : {repo_url, files:[{name, type, content}]}
OUTPUT : {records:[{target, issue_kind, flagged, confidence}]}  (player)
GOLD   : {records:[{target, issue_kind|null}], mistake_kinds:[...]}
  target = a file name, OR the repo_url string for the broken_link defect.

Beyond the four code/repo defects, the arena also injects three OPEN-PRACTICES
REPORTING defects that live in a data/code/materials AVAILABILITY STATEMENT file
(a doc), each with a matched CLEAN look-alike statement the player must NOT flag:
  dead_data_link          : statement claims open data at a DEAD/placeholder URL
                            (clean: a genuine resolvable OSF/Zenodo/figshare DOI)
  available_upon_request  : statement offers data "upon reasonable request"
                            (clean: data openly available at a real public link)
  materials_claim_no_link : statement claims materials exist but gives NO link and
                            none are present (clean: a real materials repo + the
                            materials/ file actually present in files[])
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import yaml

ARENA_DIR = Path(__file__).resolve().parent
CATALOGS_DIR = ARENA_DIR / "catalogs"

_GROUND_TRUTH_CACHE: dict[str, dict] = {}

# The closed, ordered set of injected issue kinds (= mistake_kinds for parity).
# absolute_path / uncommented_code / missing_file_load are FILE-level (code) defects;
# broken_link is a REPO-level defect whose target is the repo_url;
# dead_data_link / available_upon_request / materials_claim_no_link are
# OPEN-PRACTICES-REPORTING defects whose target is an availability-statement doc.
# NEW kinds are APPENDED so existing task ids (or-t3-0..3 etc.) keep their meaning.
CODE_ISSUE_KINDS = ["absolute_path", "uncommented_code", "missing_file_load"]
STATEMENT_ISSUE_KINDS = ["dead_data_link", "available_upon_request", "materials_claim_no_link"]
ISSUE_KINDS = CODE_ISSUE_KINDS + ["broken_link"] + STATEMENT_ISSUE_KINDS

# Which availability-statement file each statement kind renders into. Two kinds
# share the DATA_AVAILABILITY.md slot (they never co-occur in one task), and the
# materials kind uses its own MATERIALS.md slot, so no target-name collisions.
_STATEMENT_FILE = {
    "dead_data_link": "DATA_AVAILABILITY.md",
    "available_upon_request": "DATA_AVAILABILITY.md",
    "materials_claim_no_link": "MATERIALS.md",
}


def _load_catalog() -> dict:
    with (CATALOGS_DIR / "repo_templates.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed_int(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _render_script(script: dict, mode: str, rng: random.Random) -> str:
    """Render a script file's content in the given mode.

    mode: clean | no_comment | abs_path | missing | trap
      trap = a CLEAN file (relative path, commented, file present) whose comments
             deliberately MENTION an absolute path / a renamed file so it LOOKS
             suspicious. A good player must NOT flag it (T2 false-alarm trap).
    """
    abs_dir = rng.choice(_CATALOG["abs_dirs"])
    fields = {
        "data_file": script["data_file"],
        "missing_file": script["missing_file"],
        "abs_dir": abs_dir,
    }
    if mode == "trap":
        # Looks scary (mentions an absolute path + a rename in a COMMENT) but the
        # actual code uses a relative path to a file that IS present.
        body = (
            f"# {script['name']} — cleaned for sharing\n"
            f"# Originally this lived at {abs_dir}/, now made portable.\n"
            f"# Path is relative to the repo root; see README for the data layout.\n"
            f"data <- read.csv(\"{script['data_file']}\")\n"
            if script["reader"] == "read.csv"
            else
            f"# {script['name']} — cleaned for sharing\n"
            f"import pandas as pd\n"
            f"# Originally read from {abs_dir}/, rewritten to a relative path for the repo.\n"
            f"df = pd.read_csv(\"{script['data_file']}\")\n"
        )
        return body
    return script[mode].format(**fields)


def _script_target(script: dict, mode: str) -> tuple[dict, dict]:
    """Return (file_entry, gold_record) for a script rendered in `mode`."""
    content = _render_script(script, mode, _RNG)
    file_entry = {"name": script["name"], "type": script["type"], "content": content}
    issue = {
        "abs_path": "absolute_path",
        "no_comment": "uncommented_code",
        "missing": "missing_file_load",
    }.get(mode)  # clean / trap -> None
    gold = {"target": script["name"], "issue_kind": issue}
    return file_entry, gold


# Module-level handles wired up inside generate() so helpers stay simple.
_CATALOG: dict = {}
_RNG: random.Random = random.Random(0)


def _data_file_entry(script: dict) -> dict:
    """The (present) data file a clean script loads — itself never defective."""
    return {"name": script["data_file"], "type": "data", "content": "id,condition,rt\n1,a,431\n2,b,502\n"}


def _statement_fields(rng: random.Random) -> dict:
    """Deterministic placeholder fills for availability-statement templates."""
    return {
        "good_doi": rng.choice(_CATALOG["good_dois"]),
        "good_repo": rng.choice(_CATALOG["good_repos"]),
        "dead_url": rng.choice(_CATALOG["dead_links"]),
    }


def _statement_target(kind: str, mode: str, rng: random.Random) -> tuple[dict, dict, list[dict]]:
    """Render an availability-statement file in the given mode.

    kind : a STATEMENT_ISSUE_KINDS member (dead_data_link / available_upon_request
           / materials_claim_no_link).
    mode : 'bad'   -> the defective statement (gold issue_kind = kind)
           'clean' -> the matched honest look-alike (gold issue_kind = None)

    Returns (statement_file, gold_record, extra_files). For the materials kind the
    CLEAN control additionally yields the present materials/ file (extra_files);
    the BAD materials statement yields none (the claim has no link AND no file).
    """
    spec = _CATALOG["availability_statements"][kind]
    fname = spec["file"]
    fields = _statement_fields(rng)
    content = spec[mode].format(**fields)
    file_entry = {"name": fname, "type": "doc", "content": content}
    gold = {"target": fname, "issue_kind": (kind if mode == "bad" else None)}
    extra: list[dict] = []
    if kind == "materials_claim_no_link" and mode == "clean":
        mat = rng.choice(_CATALOG["materials_files"])
        extra.append({"name": mat["name"], "type": "data", "content": mat["content"]})
    return file_entry, gold, extra


def _clean_statements(rng: random.Random) -> tuple[list[dict], list[dict]]:
    """A CLEAN data-availability + a CLEAN materials statement (+ present file).

    Every base repo carries these so the open-practices look-alikes appear as
    clean negatives in the clean/trap/subtle tiers — a good player must not flag
    a genuine DOI / a real materials repo with the materials present.
    """
    files, gold = [], []
    sfe, sg, sx = _statement_target("dead_data_link", "clean", rng)   # working-DOI data statement
    files.append(sfe)
    files.extend(sx)
    gold.append(sg)
    mfe, mg, mx = _statement_target("materials_claim_no_link", "clean", rng)  # real materials repo + file
    files.append(mfe)
    files.extend(mx)
    gold.append(mg)
    return files, gold


def _assemble(task_id, tier, repo_url, files, gold_records, split, visibility) -> tuple[dict, dict]:
    targets = [f["name"] for f in files] + [repo_url]
    # mistake_kinds = the distinct injected defects present (deterministic order).
    present = [r["issue_kind"] for r in gold_records if r["issue_kind"] is not None]
    seen, kinds = set(), []
    for k in ISSUE_KINDS:
        if k in present and k not in seen:
            seen.add(k)
            kinds.append(k)
    n_issues = sum(1 for r in gold_records if r["issue_kind"] is not None)
    envelope = {
        "task_id": task_id,
        "arena_id": "open-practices-repro-v1",
        "task_set_version": "v1",
        "split": split,
        "visibility": visibility,
        "difficulty": {"tier": tier, "n_issues": n_issues},
        "input": {
            "repo_url": repo_url,
            "files": files,
            "targets": targets,
        },
    }
    gold = {"records": gold_records, "mistake_kinds": kinds}
    return envelope, gold


def generate(task_set_version: str, seed: int, split: str = "revealed"):
    global _CATALOG, _RNG
    visibility = "public" if split == "revealed" else "held_out"
    _CATALOG = _load_catalog()
    scripts = _CATALOG["scripts"]

    def emit(tier, idx, build):
        """build(rng) -> (repo_url, files, gold_records). Deterministic per cell."""
        global _RNG
        tid = f"or-t{tier}-{idx}-s{seed}"
        _RNG = random.Random(_seed_int(task_set_version, seed, tier, idx))
        repo_url, files, gold_records = build(_RNG)
        env, gt = _assemble(tid, tier, repo_url, files, gold_records, split, visibility)
        _GROUND_TRUTH_CACHE[tid] = gt
        return env

    def good_url(rng):
        return rng.choice(_CATALOG["repo_urls"]["good"])

    def broken_url(rng):
        return rng.choice(_CATALOG["repo_urls"]["broken"])

    def clean_repo(rng, script_mode="clean"):
        """All scripts in `script_mode` (clean/trap), good url, data files present.

        Also carries a CLEAN data-availability + a CLEAN materials statement (the
        confusable look-alikes for the open-practices defect kinds).
        """
        url = good_url(rng)
        files, gold = [], []
        for s in scripts:
            fe, g = _script_target(s, script_mode)
            files.append(fe)
            files.append(_data_file_entry(s))
            gold.append(g)
        # data files & a README are clean candidate targets too.
        files.append({"name": "README.md", "type": "doc",
                      "content": "# Study 1\nData and analysis code. Paths are relative to the repo root.\n"})
        gold.append({"target": "README.md", "issue_kind": None})
        for s in scripts:
            gold.append({"target": s["data_file"], "issue_kind": None})
        sfiles, sgold = _clean_statements(rng)
        files.extend(sfiles)
        gold.extend(sgold)
        gold.append({"target": url, "issue_kind": None})
        return url, files, gold

    def repo_with(rng, script_modes, url, data_stmt, materials_stmt, readme="# Study 1\nAnalysis code.\n"):
        """Build files+gold from explicit per-script + per-statement modes.

        script_modes : list aligned to `scripts` (clean/trap/abs_path/no_comment/missing).
        data_stmt    : ('dead_data_link'|'available_upon_request', 'bad') or
                       ('dead_data_link', 'clean') for the DATA_AVAILABILITY.md slot.
        materials_stmt: ('materials_claim_no_link', 'bad'|'clean') for MATERIALS.md.
        """
        files, gold = [], []
        for s, mode in zip(scripts, script_modes):
            fe, g = _script_target(s, mode)
            files.append(fe)
            files.append(_data_file_entry(s))
            gold.append(g)
        files.append({"name": "README.md", "type": "doc", "content": readme})
        gold.append({"target": "README.md", "issue_kind": None})
        for s in scripts:
            gold.append({"target": s["data_file"], "issue_kind": None})
        dk, dmode = data_stmt
        dfe, dg, dx = _statement_target(dk, dmode, rng)
        files.append(dfe)
        files.extend(dx)
        gold.append(dg)
        mk, mmode = materials_stmt
        mfe, mg, mx = _statement_target(mk, mmode, rng)
        files.append(mfe)
        files.extend(mx)
        gold.append(mg)
        gold.append({"target": url, "issue_kind": "broken_link" if url in _CATALOG["repo_urls"]["broken"] else None})
        return url, files, gold

    # ---- T1: clean/simple — everything reproducible, no defects. -------------
    for k in range(2):
        def build(rng, _k=k):
            return clean_repo(rng, "clean")
        yield emit(1, k, build)

    # ---- T2: false-alarm trap — files that LOOK suspicious but are CLEAN. ----
    # Comments mention absolute paths / renames; code is relative + present.
    for k in range(2):
        def build(rng, _k=k):
            return clean_repo(rng, "trap")
        yield emit(2, k, build)

    # ---- T3: single injected mistake — cycle through ALL kinds. -------------
    # One task per issue kind so the revealed (seed 0) set covers every kind.
    # Code/broken kinds keep both statements CLEAN; a statement kind flips only
    # its own statement to 'bad' (everything else clean) => exactly one defect.
    _code_mode = {"absolute_path": "abs_path", "uncommented_code": "no_comment",
                  "missing_file_load": "missing"}
    for i, kind in enumerate(ISSUE_KINDS):
        def build(rng, _kind=kind):
            url = broken_url(rng) if _kind == "broken_link" else good_url(rng)
            # Default: every script clean, both statements clean.
            script_modes = ["clean"] * len(scripts)
            data_stmt = ("dead_data_link", "clean")
            materials_stmt = ("materials_claim_no_link", "clean")
            if _kind in _code_mode:
                defective = rng.randrange(len(scripts))
                script_modes[defective] = _code_mode[_kind]
            elif _kind in ("dead_data_link", "available_upon_request"):
                data_stmt = (_kind, "bad")
            elif _kind == "materials_claim_no_link":
                materials_stmt = (_kind, "bad")
            return repo_with(rng, script_modes, url, data_stmt, materials_stmt)
        yield emit(3, i, build)

    # ---- T4: subtle — a single defect that hides amid trap-styled others. ----
    # Cycles the two file-level kinds AND each open-practices statement kind. The
    # genuine defect sits next to its confusable CLEAN look-alike: trap-styled
    # (suspicious-looking but clean) code, plus a CLEAN counterpart statement (a
    # genuine DOI / a real materials repo) the player must NOT flag.
    t4_kinds = ["absolute_path", "missing_file_load",
                "dead_data_link", "available_upon_request", "materials_claim_no_link"]
    for i, kind in enumerate(t4_kinds):
        def build(rng, _kind=kind):
            url = good_url(rng)
            script_modes = ["trap"] * len(scripts)  # suspicious-looking but clean
            data_stmt = ("dead_data_link", "clean")
            materials_stmt = ("materials_claim_no_link", "clean")
            if _kind in ("absolute_path", "missing_file_load"):
                defective = rng.randrange(len(scripts))
                script_modes[defective] = {"absolute_path": "abs_path",
                                           "missing_file_load": "missing"}[_kind]
            elif _kind in ("dead_data_link", "available_upon_request"):
                data_stmt = (_kind, "bad")
            elif _kind == "materials_claim_no_link":
                materials_stmt = (_kind, "bad")
            return repo_with(rng, script_modes, url, data_stmt, materials_stmt, readme="# Study 1\n")
        yield emit(4, i, build)

    # ---- T5: multiple co-occurring defects across files + repo + statement. --
    # Two file-level code defects + a broken repo_url + ONE open-practices
    # statement defect (deterministic by _k, so both splits inject the same kinds).
    for k in range(2):
        def build(rng, _k=k):
            file_modes = ["abs_path", "no_comment", "missing"]
            script_modes = [file_modes[(j + _k) % len(file_modes)] for j in range(len(scripts))]
            url = broken_url(rng)  # plus a broken link
            if _k == 0:
                data_stmt = ("dead_data_link", "bad")          # dead data link
                materials_stmt = ("materials_claim_no_link", "clean")
            else:
                data_stmt = ("available_upon_request", "bad")  # upon-request dodge
                materials_stmt = ("materials_claim_no_link", "bad")  # materials w/ no link
            return repo_with(rng, script_modes, url, data_stmt, materials_stmt, readme="# Study 1\n")
        yield emit(5, k, build)

    # ---- T6: full composition — code + statement defects, trap doc, broken url.
    # Compose every flavour: two code defects, a broken repo_url, an open-practices
    # statement defect, AND a clean-but-suspicious trap doc that must NOT be flagged.
    for k in range(2):
        def build(rng, _k=k):
            url = broken_url(rng)
            code_modes = ["abs_path", "missing"] if _k == 0 else ["no_comment", "abs_path"]
            script_modes = [code_modes[j % len(code_modes)] for j in range(len(scripts))]
            if _k == 0:
                data_stmt = ("dead_data_link", "bad")            # dead data link
                materials_stmt = ("materials_claim_no_link", "clean")
            else:
                data_stmt = ("available_upon_request", "bad")    # upon-request dodge
                materials_stmt = ("materials_claim_no_link", "bad")  # materials, no link
            url2, files, gold = repo_with(rng, script_modes, url, data_stmt, materials_stmt,
                                          readme="# Study 1\n")
            # one extra clean-but-suspicious trap doc (must NOT be flagged)
            files.append({"name": "notes.txt", "type": "doc",
                          "content": "Reproduces on a clean machine; see README.\n"})
            gold.append({"target": "notes.txt", "issue_kind": None})
            return url2, files, gold
        yield emit(6, k, build)


def ground_truth(task_id: str) -> dict:
    """Return gold for a task. Regenerated from seed via the in-process cache.

    The runner always calls generate() before ground_truth(); this arena needs
    no external gold registry because the secret is the private seed, not a
    stored answer key. Raises KeyError if the task_id was never generated.
    """
    if task_id not in _GROUND_TRUTH_CACHE:
        raise KeyError(
            f"No cached gold for {task_id!r}; call generate() for the matching "
            "split/seed before ground_truth()."
        )
    return _GROUND_TRUTH_CACHE[task_id]

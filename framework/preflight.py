"""The gate before a costly arena run: preflight, pretest, per-record evidence contract.

Owner request 2026-11-01: a check before each run, a small-batch pretest before costly runs, and a
cross-model consult (Opus/Sonnet/Sol) on what is about to be spent. Design and consult round 1:
``docs/design/2026-11-run-provenance-completeness.md`` sections 3.2 and 8.C. It lives in the runner
library (``run_tournament``), not the command-line handler, because pilots call ``run_tournament``
directly.

1. **Preflight** (no model calls). A run that is larger than the pretest and includes a player that
   calls a model is refused unless the files that shape the run are committed (tracked changes AND
   untracked files under ``framework/``, ``players/``, ``scripts/``, ``tools/`` and the arena, its
   ``runs/`` excepted) and every model-calling player's configuration digest has a stored "go"
   review in ``run_env/preflight_reviews/``.
2. **Pretest**: a run of at most ``PRETEST_N`` tasks needs neither; it is how the records a review
   reads are produced. Inside every run the first ``PRETEST_N`` tasks act as the pretest: a player
   whose every pretest record failed stops the run.
3. **Evidence contract** on every record of a model-calling player: an environment, a call
   configuration, and for an agentic CLI the recorder's block with the effort acknowledged in the
   CLI's own log and only the requested model billed. A full run stops at the first violation (the
   record is written first, so nothing paid for is lost); a pretest-sized run reports them all at
   the end.

The configuration digest covers what the consult seats said a review must be keyed on: the registry
entry, the prompt template, the CLI binary and version, the run environment, the arena, task-set version
and scorer, and the code: every repository file the adapter, the recorder, the scorer and the arena's
generator import, followed through their imports (``code_closure``; static, so an import inside a
function counts and a module loaded by path does not), plus the version of each outside package they
import and, for one not installed as a package (powerbench on PYTHONPATH), the hash of its files. Not
covered, and recorded per call instead: the CLI executable's bytes beyond its version, and account-side
state (the provider's model catalog and instructions hashes, an advisor setting). Any change re-requires
a review; one review file may cover many digests, so a queue of runs needs one consult, not one per run.
(Go review 2026-11-01: the recorder hash had been None in every digest, and the base class, the
scorer's shared modules and powerbench were not covered.)
"""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

from framework import hermetic, publication_contract, run_env

#: Tasks per player that form the pretest (design 3.2: "3 tasks per player first").
PRETEST_N = 3

#: Adapter classes that call a model. A tool or a local parser is not gated.
API_ADAPTER_CLASSES = frozenset({"OpenAIChatCompletionsAdapter", "AnthropicApiAdapter",
                                 "GeminiGenerateContentAdapter", "LocalGgufAdapter",
                                 "PowerReviewFreeAdapter"})
MODEL_ADAPTER_CLASSES = frozenset(hermetic.CLI_ADAPTER_CLASSES) | API_ADAPTER_CLASSES | {"RegcheckShimAdapter"}

#: Paths whose uncommitted state makes a record unattributable to a commit.
RUN_SHAPING_DIRS = ("framework", "players", "scripts", "tools")

REPO_ROOT = Path(__file__).resolve().parents[1]


class PreflightRefused(RuntimeError):
    """A costly run would start without what the gate requires. Nothing has been spent."""


class RecordContractViolation(RuntimeError):
    """A record of a model-calling player lacks its evidence. The record was written first."""


def _cls(entry: dict) -> str:
    return (entry.get("adapter_class") or "").rsplit(".", 1)[-1]


def spends(entry: dict) -> bool:
    """Does this player call a model (and so cost quota or produce model-dependent output)?"""
    return _cls(entry) in MODEL_ADAPTER_CLASSES


def is_costly(max_tasks: int | None, only_tasks: set[str] | None, trials: int = 1) -> bool:
    """Larger than the pretest: more than PRETEST_N calls per player (tasks x trials).

    A retry of named tasks is sized by their number. Trials count (review 2026-11-01: 3 tasks at
    30 trials was 90 paid calls labelled a pretest).
    """
    n = len(only_tasks) if only_tasks is not None else max_tasks
    if n is None:
        return True
    return n * max(1, int(trials or 1)) > PRETEST_N


# --------------------------------------------------------------------------- the tree
def dirty_run_files(arena_dir: Path, repo_root: Path = REPO_ROOT) -> list[str]:
    """Uncommitted tracked changes and untracked files under the paths that shape a run.

    Untracked files count here (unlike the record's ``-dirty`` marker): an untracked prompt or
    adapter is exactly what a review could not have seen. The arena's ``runs/`` is excluded, since
    a tournament writes its own run file there.
    """
    arena_rel = Path(arena_dir).resolve().relative_to(repo_root.resolve()).as_posix()
    paths = [*RUN_SHAPING_DIRS, arena_rel]
    proc = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
                          capture_output=True, text=True, cwd=repo_root, encoding="utf-8",
                          errors="replace", timeout=60)
    if proc.returncode != 0:
        return [f"<git status failed: {proc.stderr.strip()[:200]}>"]
    out = []
    for line in proc.stdout.splitlines():
        path = line[3:].strip().strip('"')
        if path.startswith(f"{arena_rel}/runs/") or "/__pycache__/" in f"/{path}" or path.endswith(".pyc"):
            continue
        out.append(line.strip())
    return out


# --------------------------------------------------------------------------- digests
def _sha_file(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def _module_file(obj) -> Path | None:
    """The source file of a class, function, method or module; for any other object, of its class.

    It used to take ``type(obj)`` for everything but a class, so for a function it asked for the file
    of the built-in function type and returned None (go review 2026-11-01).
    """
    target = obj if (inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj)
                     or inspect.ismodule(obj)) else type(obj)
    try:
        return Path(inspect.getfile(target))
    except (TypeError, OSError):
        return None


#: Top-level packages of this repository whose imports the digest follows.
LOCAL_PACKAGES = ("framework", "players", "arenas")


def _local_module_file(name: str, root: Path) -> Path | None:
    parts = name.split(".")
    if not parts or parts[0] not in LOCAL_PACKAGES:
        return None
    base = root.joinpath(*parts)
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def code_closure(roots: list[Path], *, root: Path = REPO_ROOT) -> tuple[dict[str, str], set[str]]:
    """Every repository file reachable from ``roots`` through import statements, with its hash, and the
    top-level names of the outside packages those files import.

    Static: an import inside a function counts (a lazy import still runs when the player plays); a
    module loaded by file path does not, which is why the scorer and the generator are passed as roots.
    Importing ``a.b`` runs ``a/__init__.py`` too, so each parent package's ``__init__`` is included.
    """
    root = root.resolve()
    files: dict[str, str] = {}
    external: set[str] = set()
    todo = [Path(r) for r in roots if r]
    while todo:
        f = todo.pop().resolve()
        try:
            rel = f.relative_to(root)
        except ValueError:
            continue
        if rel.as_posix() in files or not f.is_file():
            continue
        files[rel.as_posix()] = _sha_file(f)
        for i in range(1, len(rel.parts) - 1):
            init = root.joinpath(*rel.parts[:i], "__init__.py")
            if init.is_file():
                todo.append(init)
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        pkg = list(rel.with_suffix("").parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level:
                    base = pkg[: max(len(pkg) - (node.level - 1), 0)]
                    mod = ".".join([*base, *([mod] if mod else [])])
                names = [mod, *(f"{mod}.{a.name}" for a in node.names)] if mod else []
            else:
                continue
            for n in names:
                hit = _local_module_file(n, root)
                if hit:
                    todo.append(hit)
                elif n and n.split(".")[0] not in LOCAL_PACKAGES:
                    top = n.split(".")[0]
                    if top not in sys.stdlib_module_names and top != "__future__":
                        external.add(top)
    return dict(sorted(files.items())), external


def external_packages(names: set[str]) -> dict:
    """Each outside package's installed version; for one that is not installed as a package (a folder
    on PYTHONPATH, such as powerbench) and so has no version to pin, the hash of its Python files."""
    try:
        dists = importlib.metadata.packages_distributions()
    except Exception:  # noqa: BLE001 -- a broken metadata store must not hide the rest
        dists = {}
    out: dict = {}
    for name in sorted(names):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            out[name] = {"found": False}
            continue
        version = None
        for dist in dists.get(name, []):
            try:
                version = importlib.metadata.version(dist)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        locations = list(spec.submodule_search_locations or [])
        origin = Path(spec.origin) if spec.origin and spec.origin not in ("built-in", "frozen") else None
        where = [Path(x) for x in locations] or ([origin.parent] if origin else [])
        entry: dict = {"version": version}
        if not any("site-packages" in w.parts for w in where):
            if locations:
                h = hashlib.sha256()
                pkg_dir = Path(locations[0])
                for f in sorted(pkg_dir.rglob("*.py")):
                    h.update((f.relative_to(pkg_dir).as_posix() + "\t" + str(_sha_file(f)) + "\n").encode())
                entry["files_sha256"] = h.hexdigest()
            elif origin:
                entry["files_sha256"] = _sha_file(origin)
        out[name] = entry
    return out


def config_components(entry: dict, adapter, *, arena_id: str, task_set_version: str, env_sha: str,
                      scorer_path: Path | None) -> dict:
    """Everything a review of this player-in-this-arena is a review of."""
    from framework import model_call
    cli = None
    cmd = entry.get("cli_command") or []
    if cmd:
        binary = hermetic.normalise_binary(cmd[0])
        cli = {"binary": binary, "version": hermetic.cli_version(binary)}
    elif _cls(entry) == "RegcheckShimAdapter" and entry.get("regcheck_client") == "openai":
        cli = {"binary": "codex", "version": hermetic.cli_version("codex")}
    clean_entry = {k: v for k, v in sorted(entry.items()) if not k.startswith("_")}
    # A containerised player is wrapped by framework.container.ContainerizedAdapter; the code that
    # answers is the registry's adapter class inside the image, so hash that (review 2026-11-01).
    from framework import player_adapter
    played = player_adapter._ADAPTER_CLASSES.get(entry.get("adapter_class") or "") or type(adapter)
    roots = [_module_file(played), _module_file(type(adapter)), _module_file(model_call.prepare), scorer_path]
    if scorer_path is not None and (Path(scorer_path).parent / "generator.py").is_file():
        roots.append(Path(scorer_path).parent / "generator.py")
    code, external = code_closure([r for r in roots if r is not None])
    return {
        "digest_schema_version": 2,
        "arena_id": arena_id,
        "task_set_version": task_set_version,
        "scorer_sha256": _sha_file(scorer_path),
        "registry_entry": clean_entry,
        "adapter_class": played.__name__,
        "adapter_code_sha256": _sha_file(_module_file(played)),
        "wrapper_class": type(adapter).__name__ if type(adapter) is not played else None,
        "recorder_code_sha256": _sha_file(_module_file(model_call.prepare)),
        "hermetic_profile": hermetic.PROFILE_VERSION,
        # Resolved from the repository, not the working folder (a relative path read from elsewhere
        # hashed nothing).
        "prompt_template_sha256": _sha_file(REPO_ROOT / entry["prompt_template_path"])
        if entry.get("prompt_template_path") else None,
        "cli": cli,
        "env_sha": env_sha,
        "code_sha256": code,
        "external_packages": external_packages(external),
    }


def config_digest(components: dict) -> str:
    return run_env.canonical_sha256(components)


# --------------------------------------------------------------------------- reviews
def review_dir() -> Path:
    return run_env.store_root() / "preflight_reviews"


def find_review(digest: str) -> dict | None:
    """The newest stored review whose verdict is "go" and which covers ``digest``."""
    d = review_dir()
    if not d.is_dir():
        return None
    best = None
    for p in sorted(d.glob("*.json")):
        try:
            rev = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if rev.get("verdict") == "go" and digest in (rev.get("digests") or {}):
            if best is None or str(rev.get("reviewed_utc")) > str(best.get("reviewed_utc")):
                best = {**rev, "_path": p.as_posix()}
    return best


def write_review(*, digests: dict[str, dict], verdict: str, seats: list[str], consult_dir: str,
                 pretest_run_files: list[str], notes: str = "") -> Path:
    """Store one review covering one or more configuration digests (``{digest: label}``)."""
    if verdict not in ("go", "no-go"):
        raise ValueError("verdict must be go or no-go")
    if verdict == "go" and len(seats) < 2:
        raise ValueError("a go verdict needs the running session plus at least one other model's seat")
    if verdict == "go":
        # Tie the verdict to a consult that ran (review 2026-11-01: a list of seat names proved nothing).
        rj = Path(consult_dir) / "ROUND.json"
        try:
            rnd = json.loads(rj.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"a go verdict needs the consult round's ROUND.json ({rj}): {e}") from e
        answered = [s.get("seat") for s in rnd.get("seats") or [] if s.get("verdict") == "OK"]
        if not rnd.get("complete") or not answered:
            raise ValueError(f"{rj}: the round is incomplete or no seat answered OK; not a review")
    rev = {"review_schema_version": 1, "reviewed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "verdict": verdict, "seats": seats, "consult_dir": consult_dir,
           "pretest_run_files": pretest_run_files, "digests": digests, "notes": notes}
    d = review_dir()
    d.mkdir(parents=True, exist_ok=True)
    name = run_env.canonical_sha256(rev)[:16]
    path = d / f"{rev['reviewed_utc'][:10]}-{name}.json"
    path.write_text(json.dumps(rev, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- the gate
def check(*, arena: dict, arena_dir: Path, task_set_version: str, adapters: list, env_shas: dict,
          scorer_path: Path | None, max_tasks: int | None, only_tasks: set[str] | None,
          trials: int = 1) -> dict:
    """Refuse a costly run that lacks a clean tree or a review. Returns what it checked."""
    spenders = [(e, a) for e, a in adapters if spends(e)]
    digests = {}
    for e, a in spenders:
        comps = config_components(e, a, arena_id=arena["arena_id"], task_set_version=task_set_version,
                                  env_sha=env_shas[e["player_id"]], scorer_path=scorer_path)
        # Content-addressed, so a reviewer can read exactly what a digest stands for.
        digests[e["player_id"]] = run_env._store("preflight_configs", comps)
    report = {"costly": is_costly(max_tasks, only_tasks, trials), "digests": digests, "reviews": {}}
    if not spenders or not report["costly"]:
        return report
    problems = []
    dirty = dirty_run_files(arena_dir)
    if dirty:
        problems.append("uncommitted or untracked files shape this run (commit them first):\n    "
                        + "\n    ".join(dirty[:20]) + ("\n    ..." if len(dirty) > 20 else ""))
    missing = []
    for pid, dg in digests.items():
        rev = find_review(dg)
        if rev is None:
            missing.append(f"{pid}  {dg}")
        else:
            report["reviews"][pid] = rev["_path"]
    if missing:
        problems.append(
            "no stored 'go' review for these configurations (player, digest):\n    " + "\n    ".join(missing)
            + f"\n  Run a pretest first (at most {PRETEST_N} tasks per player), have the pretest records reviewed"
              " by a consult (this session plus another model), then store the verdict with\n"
              "    python -m framework.preflight review --verdict go --seats <s1,s2> --consult-dir <dir>"
              " --pretest <run file> --digest <player=digest> ...")
    if problems:
        raise PreflightRefused(f"{arena['arena_id']}: refusing a costly run.\n  " + "\n  ".join(problems))
    return report


def record_problems(record: dict, entry: dict) -> list[str]:
    """What this model-calling player's record lacks. Empty: the record carries its evidence."""
    if not spends(entry):
        return []
    out = []
    if not record.get("env_sha"):
        out.append("no env_sha")
    cls = _cls(entry)
    observable = cls in API_ADAPTER_CLASSES or cls in hermetic.CLI_ADAPTER_CLASSES or (
        cls == "RegcheckShimAdapter" and entry.get("regcheck_client") == "openai")
    if observable and not record.get("call_config_sha256"):
        out.append("no call_config_sha256")
    if cls in hermetic.CLI_ADAPTER_CLASSES or (cls == "RegcheckShimAdapter" and entry.get("regcheck_client") == "openai"):
        mc = record.get("model_call")
        if not publication_contract.has_call_block(mc):
            out.append("no model_call block (or one that records no call)")
            return out
        blocks = mc.get("blocks") if isinstance(mc.get("blocks"), list) else [mc]
        for i, b in enumerate(blocks):
            where = f"call {i + 1}: " if len(blocks) > 1 else ""
            ch = b.get("checks") or {}
            if ch.get("effort_applied") not in (True, "not-applicable", "unverified"):
                out.append(f"{where}effort_applied={ch.get('effort_applied')!r} "
                           f"(asked {(b.get('launch') or {}).get('effort_requested')!r})")
            if ch.get("session_log") not in ("found", "inside-container-not-kept", "not-applicable"):
                out.append(f"{where}session_log={ch.get('session_log')!r}: the effort cannot be checked")
            if ch.get("only_requested_model_billed") is False:
                out.append(f"{where}a model other than the requested one was billed")
            # A missing check is not a pass (go review 2026-11-01: a Codex record with no evidence of
            # which model answered was accepted).
            if ch.get("session_log") == "found" and ch.get("only_requested_model_served") is not True:
                out.append(f"{where}the session log does not show that only the requested model answered "
                           f"(only_requested_model_served={ch.get('only_requested_model_served')!r})")
            allowed = (b.get("launch") or {}).get("allow_tools")
            if allowed == [] and (ch.get("tools_called") or 0) > 0:
                out.append(f"{where}called {ch['tools_called']} tool(s) though the launch allowed none")
            if ch.get("cli_version_changed"):
                out.append(f"{where}the CLI version changed after launch ({ch['cli_version_changed']}): "
                           "the reviewed configuration no longer describes this call")
    return out


# --------------------------------------------------------------------------- command line
def _main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m framework.preflight")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rv = sub.add_parser("review", help="store a consult's verdict for one or more configuration digests")
    rv.add_argument("--verdict", required=True, choices=["go", "no-go"])
    rv.add_argument("--seats", required=True, help="comma-separated, e.g. opus-session,sonnet,sol")
    rv.add_argument("--consult-dir", required=True)
    rv.add_argument("--pretest", action="append", default=[], help="pretest run file the review read")
    rv.add_argument("--digest", action="append", required=True, help="player_id=digest")
    rv.add_argument("--notes", default="")
    a = ap.parse_args(argv)
    digests = dict(d.split("=", 1)[::-1] for d in a.digest)
    path = write_review(digests=digests, verdict=a.verdict, seats=[s for s in a.seats.split(",") if s],
                        consult_dir=a.consult_dir, pretest_run_files=a.pretest, notes=a.notes)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))

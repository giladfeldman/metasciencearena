"""Every filesystem root the framework needs, resolved in exactly one place.

WHY THIS MODULE EXISTS
----------------------
Six modules each computed ``Path(__file__).resolve().parents[1]`` and hung a
root off it. That works only while the code sits in a checkout. Under a real
``pip install`` the same expression points at ``site-packages/``, so:

* ``site-packages/contract/schemas/arena_manifest.schema.json`` — **verified
  missing**, so ``load_arena`` raised ``FileNotFoundError`` on the first call;
* ``site-packages/arenas`` — absent, and ``framework arenas list`` responded by
  printing nothing and exiting 0, i.e. a broken install looked like an empty one.

An *editable* install hides all of it, because the checkout is still on the
path. Nothing fails until someone installs for real with the source tree moved
away — which is what ``scripts/check_clean_install.py`` now does on every push.

THE RULES HERE
--------------
1. **Package data travels with the package.** The contract schemas are resolved
   through ``importlib.resources``, so they are found identically from a
   checkout, a wheel, and a zipimport.
2. **Data roots are inputs, not constants.** ``arenas/`` and
   ``players/registry.yaml`` are *the caller's* data — a third party installing
   this package has their own. They come from an environment variable or an
   explicit argument, with the in-repo location as a convenience default.
3. **A missing root raises.** It never returns an empty list, an empty path, or
   ``None``. A discovery helper that returns empty when the root is wrong makes
   a broken run indistinguishable from a clean one — the exact failure that let
   a portfolio-wide directory move go unnoticed for five weeks.
"""
from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

__all__ = [
    "ARENAS_ROOT_ENV",
    "REGISTRY_PATH_ENV",
    "RootNotFoundError",
    "arenas_root",
    "installed_repo_root",
    "registry_path",
    "schema_path",
]

#: Point the framework at an arenas directory outside this repo.
ARENAS_ROOT_ENV = "SCIENCEARENA_ARENAS_ROOT"
#: Point the framework at a players registry outside this repo.
REGISTRY_PATH_ENV = "SCIENCEARENA_REGISTRY"


class RootNotFoundError(FileNotFoundError):
    """A required data root could not be located. Never swallowed."""


def installed_repo_root() -> Path:
    """The checkout root, when running from one.

    Under ``pip install`` this resolves to ``site-packages`` and the paths built
    from it will not exist — which is precisely why every caller below treats it
    as a *fallback* and raises when the result is absent.
    """
    return Path(__file__).resolve().parents[1]


def schema_path(name: str) -> Path:
    """Absolute path to a contract schema shipped as package data.

    ``name`` is a bare filename such as ``run_record.schema.json``.
    """
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"schema name must be a bare filename, got {name!r}")
    # `as_file` would be needed for a zipimported distribution; this project is
    # installed from a wheel onto a real filesystem, and `files()` already
    # returns a concrete path there. Resolve eagerly so the error, if any, names
    # the schema rather than surfacing later as a confusing JSON parse failure.
    path = Path(str(resources.files("framework.contract").joinpath("schemas", name)))
    if not path.is_file():
        raise RootNotFoundError(
            f"contract schema {name!r} is missing from the installed package "
            f"(looked in {path.parent}). The wheel was built without its package "
            f"data — check [tool.setuptools.package-data] in pyproject.toml."
        )
    return path


def _from_env_or_repo(env_var: str, repo_relative: str, kind: str, *, must_be_dir: bool) -> Path:
    override = os.environ.get(env_var)
    if override:
        path = Path(override).expanduser().resolve()
        ok = path.is_dir() if must_be_dir else path.is_file()
        if not ok:
            raise RootNotFoundError(
                f"{env_var}={override!r} does not point at an existing {kind}. "
                f"Refusing to continue: a wrong root silently produces an empty "
                f"run that looks like a clean one."
            )
        return path

    path = installed_repo_root() / repo_relative
    ok = path.is_dir() if must_be_dir else path.is_file()
    if not ok:
        raise RootNotFoundError(
            f"No {kind} found. Set {env_var} to your {repo_relative}, or run from "
            f"a checkout that has one (looked for {path}). This package ships the "
            f"scoring framework, not the arena data."
        )
    return path


def arenas_root() -> Path:
    """Directory holding one subdirectory per arena. Raises if absent."""
    return _from_env_or_repo(ARENAS_ROOT_ENV, "arenas", "arenas directory", must_be_dir=True)


def registry_path() -> Path:
    """Path to ``players/registry.yaml``. Raises if absent."""
    return _from_env_or_repo(
        REGISTRY_PATH_ENV, "players/registry.yaml", "players registry file", must_be_dir=False
    )

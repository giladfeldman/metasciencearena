"""The PDF-universe build cache lives OUTSIDE the project (d-ff1794).

Its pickles hold whole synthetic papers (prose + reference lists). Inside the repo, even
gitignored under tmp/, the portfolio custody scan (article-finder publication-text-scan)
counts each one as an article-text leak, so every /ship Phase 0.4 went red after any run
that rebuilt the universe.
"""
from __future__ import annotations

import importlib

from framework.pdfforge import universe as U


def test_default_cache_is_not_inside_the_repo(monkeypatch):
    monkeypatch.delenv("SCIENCEARENA_PDFFORGE_CACHE", raising=False)
    mod = importlib.reload(U)
    try:
        assert not mod.CACHE.resolve().is_relative_to(mod.REPO.resolve()), mod.CACHE
    finally:
        importlib.reload(U)


def test_env_override_names_the_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("SCIENCEARENA_PDFFORGE_CACHE", str(tmp_path / "c"))
    mod = importlib.reload(U)
    try:
        assert mod.CACHE == tmp_path / "c"
    finally:
        monkeypatch.delenv("SCIENCEARENA_PDFFORGE_CACHE")
        importlib.reload(U)

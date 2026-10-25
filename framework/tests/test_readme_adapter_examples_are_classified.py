"""A documented adapter registration must also classify the class for the runner.

The runner refuses an adapter class that is neither containerised nor a named
exemption (RunnerUnavailable, CLAUDE.md "no arena run outside the runner image").
PACKAGE_README.md's Quickstart registered MyGrimAdapter without classifying it, so
the documented example failed (sciencearena-cleanup, 2026-10-19). The docs gate
runs the Quickstart, but only when its full run is invoked; this check is static
and instant.
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
# The public mirror publishes PACKAGE_README.md as README.md (publish.MIRROR_RENAMES),
# so the same check runs against the README a public reader actually gets.
README = _ROOT / "PACKAGE_README.md" if (_ROOT / "PACKAGE_README.md").exists() else _ROOT / "README.md"


def unclassified_registrations(text: str) -> list[str]:
    bad = []
    for block in re.findall(r"```python\n(.*?)```", text, flags=re.S):
        for name in re.findall(r'register_adapter_class\(\s*"([^"]+)"', block):
            if not re.search(rf'(EXEMPT|CONTAINER)_ADAPTER_CLASSES\b[^\n]*"{re.escape(name)}"', block):
                bad.append(name)
    return bad


def test_every_documented_registration_is_classified():
    assert unclassified_registrations(README.read_text(encoding="utf-8")) == []


def test_the_check_catches_an_unclassified_registration():
    block = '```python\nregister_adapter_class("X", X)\n```\n'
    assert unclassified_registrations(block) == ["X"]

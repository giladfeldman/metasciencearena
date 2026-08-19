"""Held-out egress gate (2026-08-04).

Sending a held-out task to a third-party provider is IRREVERSIBLE. On 2026-08-04
a `--split revealed` run of a cloud LLM player transmitted 9 held-out real papers
before it was caught — `--split` selects the seed and never filtered visibility.

Two independent guards exist now. The CLI makes `--split revealed` imply
--public-only (test_revealed_split_is_public_only.py). THIS gate lives in the
runner, so it also protects every caller that bypasses the CLI: scripts,
notebooks, the retry queue, any future API.
"""
from __future__ import annotations

import pytest

from framework.runner import (
    _EGRESS_ALLOW_ENV,
    assert_heldout_egress_allowed,
    is_cloud_player,
)

CLOUD = {"player_id": "claude-sonnet-5-sections", "adapter_class": "LlmCliPdfSectionsAdapter"}
CLI_CLOUD = {"player_id": "claude-haiku-4-5-grim", "adapter_class": "SubprocessCliAdapter"}
LOCAL_R = {"player_id": "scrutiny-grim", "adapter_class": "RCliAdapter"}
LOCAL_LIB = {"player_id": "docpluck-sections", "adapter_class": "DocpluckSectionsAdapter"}


def test_llm_pdf_adapter_is_cloud():
    assert is_cloud_player(CLOUD) is True


def test_subprocess_cli_is_cloud():
    """The CLI players are `claude`/`gemini`/`codex`/`opencode` — all remote."""
    assert is_cloud_player(CLI_CLOUD) is True


def test_local_r_and_library_adapters_are_not_cloud():
    assert is_cloud_player(LOCAL_R) is False
    assert is_cloud_player(LOCAL_LIB) is False


def test_local_classification_requires_review_not_just_a_non_cloud_name():
    """REPLACES an assertion that encoded the 2026-08-12 defect.

    This test used to read: "a new REMOTE adapter should be named with a cloud
    prefix. Anything genuinely local stays local", and asserted that the invented
    class `SomeLocalLibAdapter` was LOCAL. That makes a naming convention the
    safety mechanism — the exact thing the GROBID note further down this file says
    not to trust — and `ModelProxyAdapter` is the proof it fails: a genuinely
    remote adapter that nobody named with a cloud prefix, silently exempt from the
    gate.

    The invariant now: a class is local only if it was REVIEWED onto
    `_LOCAL_ADAPTER_CLASSES`. An unrecognised name is cloud.
    """
    from framework.runner import _LOCAL_ADAPTER_CLASSES

    assert "DocpluckSectionsAdapter" in _LOCAL_ADAPTER_CLASSES
    assert is_cloud_player({"player_id": "x", "adapter_class": "DocpluckSectionsAdapter"}) is False
    assert is_cloud_player({"player_id": "x", "adapter_class": "SomeLocalLibAdapter"}) is True
    assert is_cloud_player({"player_id": "x", "adapter_class": "HttpAdapter"}) is True


def test_gate_blocks_cloud_player_on_held_out(monkeypatch):
    monkeypatch.delenv(_EGRESS_ALLOW_ENV, raising=False)
    with pytest.raises(RuntimeError) as exc:
        assert_heldout_egress_allowed([CLOUD, LOCAL_R], will_play_held_out=True)
    msg = str(exc.value)
    assert "claude-sonnet-5-sections" in msg
    assert _EGRESS_ALLOW_ENV in msg, "the error must name the exact opt-in"
    assert "DATA_HANDLING" in msg
    assert "scrutiny-grim" not in msg, "local players must not be blamed"


def test_gate_allows_public_only_run(monkeypatch):
    """public_only=True means no held-out envelope ever reaches a player."""
    monkeypatch.delenv(_EGRESS_ALLOW_ENV, raising=False)
    assert_heldout_egress_allowed([CLOUD], will_play_held_out=False)


def test_gate_allows_local_players_on_held_out(monkeypatch):
    """Local tools send nothing off-machine, so held-out is fine for them."""
    monkeypatch.delenv(_EGRESS_ALLOW_ENV, raising=False)
    assert_heldout_egress_allowed([LOCAL_R, LOCAL_LIB], will_play_held_out=True)


def test_explicit_env_opt_in_permits_egress(monkeypatch):
    monkeypatch.setenv(_EGRESS_ALLOW_ENV, "1")
    assert_heldout_egress_allowed([CLOUD], will_play_held_out=True)


def test_a_random_env_value_does_not_count_as_consent(monkeypatch):
    monkeypatch.setenv(_EGRESS_ALLOW_ENV, "maybe")
    with pytest.raises(RuntimeError):
        assert_heldout_egress_allowed([CLOUD], will_play_held_out=True)


# --- loopback HTTP players are NOT egress (2026-08-09) -----------------------
#
# `escimate` is an `HttpAdapter` pointed at `http://127.0.0.1:9422` and started by
# a local `start_command`. Nothing leaves the machine. But `HttpAdapter` matches
# `_CLOUD_ADAPTER_PREFIXES`, so the gate classed it as third-party egress and
# refused every private-split run — including the stats-extraction-v1 task-set v2
# tournament on 2026-08-09, where it blocked the ONLY non-statcheck player the
# private split has ever had.
#
# The conservative default is right and stays: an HttpAdapter with no endpoint, or
# one pointed anywhere but loopback, is still cloud. What changes is that an
# explicitly loopback endpoint is believed, because it is checkable evidence
# rather than a name on an allowlist.

LOOPBACK_HTTP = {
    "player_id": "escimate",
    "adapter_class": "HttpAdapter",
    "endpoint": "http://127.0.0.1:9422/api/v1/process-text",
}
REMOTE_HTTP = {
    "player_id": "some-saas",
    "adapter_class": "HttpAdapter",
    "endpoint": "https://api.example.com/v1/process",
}


def test_loopback_http_player_is_not_cloud():
    assert is_cloud_player(LOOPBACK_HTTP) is False


def test_remote_http_player_is_still_cloud():
    assert is_cloud_player(REMOTE_HTTP) is True


def test_http_player_without_an_endpoint_stays_cloud():
    """Unknown destination keeps the conservative default."""
    assert is_cloud_player({"player_id": "x", "adapter_class": "HttpAdapter"}) is True


def test_openai_compatible_adapter_is_cloud_without_endpoint():
    assert is_cloud_player({
        "player_id": "nvidia-nemotron-nano-9b-grim-api",
        "adapter_class": "OpenAIChatCompletionsAdapter",
    }) is True


def test_antigravity_cli_adapter_is_cloud_without_endpoint():
    assert is_cloud_player({
        "player_id": "antigravity-gemini-3-6-flash-low-grim",
        "adapter_class": "AntigravityCliAdapter",
    }) is True


import pytest  # noqa: E402


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:8070",
        "http://127.0.0.1:9422/api",
        "http://[::1]:9422/api",
        "http://127.1.2.3:80/x",
    ],
)
def test_all_loopback_forms_are_local(endpoint):
    assert is_cloud_player(
        {"player_id": "x", "adapter_class": "HttpAdapter", "endpoint": endpoint}
    ) is False


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.anthropic.com/v1",
        "http://192.168.1.10:9422",          # LAN is still off-machine
        "http://evil.com#127.0.0.1",         # fragment must not fool the parse
        "http://127.0.0.1.attacker.com/x",   # suffix trick
    ],
)
def test_non_loopback_endpoints_stay_cloud(endpoint):
    assert is_cloud_player(
        {"player_id": "x", "adapter_class": "HttpAdapter", "endpoint": endpoint}
    ) is True


def test_the_real_escimate_registry_entry_is_local():
    """Pin the actual production entry, not just a hand-written stand-in."""
    import yaml
    from pathlib import Path

    reg = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "players" / "registry.yaml").read_text(encoding="utf-8")
    )
    entry = next(p for p in reg if p["player_id"] == "escimate")
    assert is_cloud_player(entry) is False


# --- a declared endpoint OUTRANKS the adapter-name heuristic (2026-08-09) -----
#
# `is_cloud_player` only inspected `endpoint` for adapter classes already matching
# `_CLOUD_ADAPTER_PREFIXES`. Every other class returned False WITHOUT looking at
# the endpoint at all — flatly contradicting the function's own docstring
# ("Deliberately conservative: unknown adapters are treated as cloud").
#
# The GROBID adapters are the live instance: `GrobidTextAdapter`,
# `GrobidSectionsAdapter`, `GrobidTablesAdapter`, `GrobidReferencesAdapter` and
# `GrobidCitationsAdapter` all take `endpoint: str = "http://localhost:8070"` and
# match no cloud prefix. Public GROBID servers exist. A one-line `endpoint:` edit
# in registry.yaml pointing any of them at a remote host would have sent held-out
# real papers — some copyrighted APA PDFs — off-machine with no gate, no warning
# and no opt-in. Dormant today because every entry uses the localhost default.
#
# Found by a Sonnet review pass 2026-08-09, reproduced before fixing. The endpoint
# is DIRECT evidence of where bytes go; the class name is only a naming
# convention, and conventions are what this defect was hiding behind.

REMOTE_GROBID = {
    "player_id": "grobid-text-only",
    "adapter_class": "GrobidTextAdapter",
    "endpoint": "http://cloud.science-miner.com:8070",
}
LOCAL_GROBID = {
    "player_id": "grobid-text-only",
    "adapter_class": "GrobidTextAdapter",
    "endpoint": "http://localhost:8070",
}


def test_remote_endpoint_makes_any_adapter_cloud_whatever_its_class_is_called():
    assert is_cloud_player(REMOTE_GROBID) is True


def test_loopback_endpoint_keeps_a_non_prefixed_adapter_local():
    assert is_cloud_player(LOCAL_GROBID) is False


@pytest.mark.parametrize("endpoint", [
    "https://grobid.example.com",
    "http://198.51.100.7:8070",
    "http://192.168.1.10:8070",
])
def test_every_off_machine_endpoint_is_cloud_regardless_of_class(endpoint):
    assert is_cloud_player(
        {"player_id": "x", "adapter_class": "SomeLocalLookingAdapter", "endpoint": endpoint}
    ) is True


def test_endpointless_local_adapter_is_still_local():
    """No endpoint means no evidence; fall back to the class-name heuristic."""
    assert is_cloud_player({"player_id": "x", "adapter_class": "RCliAdapter"}) is False


def test_every_real_registry_entry_with_an_endpoint_is_classified_from_it():
    """Pin production: today every endpoint is loopback, so every one must be local."""
    import yaml
    from pathlib import Path

    reg = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "players" / "registry.yaml").read_text(encoding="utf-8")
    )
    checked = 0
    for entry in reg:
        ep = entry.get("endpoint")
        if not ep:
            continue
        checked += 1
        assert is_cloud_player(entry) is False, (
            f"{entry['player_id']} points at {ep}, which is NOT loopback — it would "
            "send held-out papers off-machine. Confirm that is intended."
        )
    assert checked > 0, "no registry entry declares an endpoint — check would be vacuous"


# --- an adapter that TALKS TO THE NETWORK must be cloud, whatever it is called -
#
# Found 2026-08-12 by enumerating the live adapter registry instead of reasoning
# about it. `ModelProxyAdapter` (players/adapters/model_proxy.py) POSTs every task
# to `f"{MODEL_PROXY_URL}/openapi/chat/completions"` — the Kaggle model proxy, a
# third party. It declares no `endpoint:` key (the URL arrives via env var, the
# same shape as OpenAIChatCompletionsAdapter), and its class name matched no entry
# in `_CLOUD_ADAPTER_PREFIXES`, so `is_cloud_player` returned False and the
# held-out egress gate did not fire for it at all.
#
# It is the GROBID defect re-armed, in a different adapter: dormant ONLY because
# all three `kaggle-*` registry entries are still commented out. Uncommenting them
# — the exact next step when the Kaggle Benchmarks grant lands — would have sent
# held-out real papers to a third-party proxy with no gate, no warning and no
# opt-in.
#
# Verified before fixing: of the 33 registered adapter classes, ModelProxyAdapter
# was the ONLY one performing network I/O that `is_cloud_player` called local.

MODEL_PROXY = {"player_id": "kaggle-gpt-5-grim", "adapter_class": "ModelProxyAdapter"}


def test_model_proxy_adapter_is_cloud():
    """It POSTs to the Kaggle proxy. No endpoint key, so the class name must carry it."""
    assert is_cloud_player(MODEL_PROXY) is True


def test_unknown_adapter_without_an_endpoint_is_cloud():
    """The docstring promised this; the code did the opposite.

    `is_cloud_player` said "unknown adapters are treated as cloud", but with no
    endpoint it returned `cls.startswith(_CLOUD_ADAPTER_PREFIXES)` — so an
    unrecognised class was silently LOCAL. That makes egress opt-OUT for anything
    nobody remembered to classify, which is backwards.
    """
    assert is_cloud_player(
        {"player_id": "x", "adapter_class": "SomeBrandNewProviderAdapter"}
    ) is True


def test_every_registered_adapter_class_is_explicitly_classified():
    """A new adapter must force a classification decision, not inherit a default.

    Walks the REAL adapter registry (importing the modules, so this cannot pass on
    a clone that is missing an adapter file — the gap recorded in TODO.md). Every
    registered class must be either cloud by prefix, or named in the framework's
    reviewed local allowlist.
    """
    from framework import runner as _runner  # noqa: F401  (triggers adapter autoload)
    from framework.player_adapter import _ADAPTER_CLASSES
    from framework.runner import _LOCAL_ADAPTER_CLASSES

    assert len(_ADAPTER_CLASSES) > 10, "adapters did not autoload; the check would be vacuous"

    unclassified = [
        name for name in sorted(_ADAPTER_CLASSES)
        if not is_cloud_player({"player_id": "probe", "adapter_class": name})
        and name not in _LOCAL_ADAPTER_CLASSES
    ]
    assert not unclassified, (
        "these adapter classes are treated as LOCAL but were never reviewed as local: "
        + ", ".join(unclassified)
        + ". Add each to _CLOUD_ADAPTER_PREFIXES or to _LOCAL_ADAPTER_CLASSES in "
          "framework/runner.py after confirming where its bytes actually go."
    )


def test_local_allowlist_names_only_classes_that_exist():
    """A stale allowlist entry is a licence nobody is using — and hides a rename."""
    from framework import runner as _runner  # noqa: F401
    from framework.player_adapter import _ADAPTER_CLASSES
    from framework.runner import _LOCAL_ADAPTER_CLASSES

    stale = sorted(set(_LOCAL_ADAPTER_CLASSES) - set(_ADAPTER_CLASSES))
    assert not stale, f"_LOCAL_ADAPTER_CLASSES names non-existent adapters: {stale}"

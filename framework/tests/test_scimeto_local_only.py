"""The local-only guard on OUR OWN hosted tools actually fires.

User directive 2026-08-28, verbatim scope: *"for tools we develop that are
competing on the arenas that have online versions and local server versions, we
never use the online version, we only use the local servers. These are things
like scimeto, escimate, docpluck... we run them on vercel/render/neon/supabase/
railway and these things cost us money. so, only for those."* It does NOT cover
third-party AI players -- free-tier APIs, subscription CLIs and local runs are
how the arena is meant to work and must not be blocked.

WHY THESE THREE PLAYERS AND NOT A REGISTRY-WIDE PREDICATE. `is_cloud_player()`
classifies from the endpoint DECLARED in registry.yaml. The three scimeto players
declare NO endpoint and resolve their URL from `SCIMETO_API_URL` at CALL time, so
that gate cannot see them by construction. They are simultaneously the entire
population the directive is about and the only players no existing check covers.

MEASURED 2026-08-28, the reason this is not hypothetical: `SCIMETO_API_KEY` is
set in the Windows USER environment with a `cg_live_` prefix, so it is armed and
persistent in every shell. A local dev instance would not need a live key. Only
the ABSENCE of `SCIMETO_API_URL` stood between an arena run and Vercel/Neon
billing -- and `_require_local_scimeto` existed with zero call sites for six days
(written 2026-08-22, wired 2026-08-28).

WATCHED FAIL FIRST: before wiring, `_config()` with `SCIMETO_API_URL` pointed at
the hosted instance returned that URL together with the live key and raised
nothing.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: RETIRED 2026-09-22: `scimeto_citations.py` / `scimeto_matching.py` were
#: deleted with their registry entries -- they wired a synchronous route SciMeto
#: has never served and never produced a record. See the tombstones in
#: players/registry.yaml. The guard itself is unchanged; only its population
#: shrank, so this list is now the one adapter that remains.
ADAPTERS = [
    ("scimeto_replication.py", "ScimetoReplicationApiAdapter"),
]


def _adapter(filename, clsname):
    path = REPO / "players" / "adapters" / filename
    spec = importlib.util.spec_from_file_location(f"_lo_{filename[:-3]}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    a = getattr(mod, clsname).__new__(getattr(mod, clsname))
    a.api_url_env = "SCIMETO_API_URL"
    a.api_key_env = "SCIMETO_API_KEY"
    return a


@pytest.mark.parametrize("filename,clsname", ADAPTERS)
def test_a_hosted_url_is_refused(monkeypatch, filename, clsname):
    """The defect. Before wiring, this returned the hosted URL and a live key."""
    monkeypatch.setenv("SCIMETO_API_URL", "https://scimeto.app/api/replications/lookup")
    monkeypatch.delenv("SCIENCEARENA_ALLOW_REMOTE", raising=False)
    with pytest.raises(RuntimeError, match="NOT a local service"):
        _adapter(filename, clsname)._config()


@pytest.mark.parametrize("filename,clsname", ADAPTERS)
def test_a_loopback_url_still_passes(monkeypatch, filename, clsname):
    """CONTROL. A guard that refused everything would pass the test above while
    breaking every legitimate local run, which is the failure this pairs with."""
    monkeypatch.setenv("SCIMETO_API_URL", "http://127.0.0.1:3001/api/replications/lookup")
    monkeypatch.delenv("SCIENCEARENA_ALLOW_REMOTE", raising=False)
    url, _key = _adapter(filename, clsname)._config()
    assert url.startswith("http://127.0.0.1:")


@pytest.mark.parametrize("filename,clsname", ADAPTERS)
def test_the_default_when_the_env_is_unset_is_loopback(monkeypatch, filename, clsname):
    monkeypatch.delenv("SCIMETO_API_URL", raising=False)
    monkeypatch.delenv("SCIENCEARENA_ALLOW_REMOTE", raising=False)
    url, _key = _adapter(filename, clsname)._config()
    assert url.startswith("http://127.0.0.1:3001/")


@pytest.mark.parametrize("filename,clsname", ADAPTERS)
def test_the_explicit_override_still_works(monkeypatch, filename, clsname):
    """The override must remain reachable: the directive is about ACCIDENTAL
    spend, not about making a deliberate, approved hosted run impossible."""
    monkeypatch.setenv("SCIMETO_API_URL", "https://scimeto.app/api/replications/lookup")
    monkeypatch.setenv("SCIENCEARENA_ALLOW_REMOTE", "1")
    url, _key = _adapter(filename, clsname)._config()
    assert url.startswith("https://")


def test_a_loopback_lookalike_hostname_is_not_treated_as_local(monkeypatch):
    """`_is_loopback_endpoint` matches the HOST, never a substring. A gate that
    accepted these would be worse than none, because the URL LOOKS local."""
    for hostile in ("http://127.0.0.1.attacker.example/api",
                    "https://evil.example/x#127.0.0.1",
                    "https://localhost.attacker.example/api"):
        monkeypatch.setenv("SCIMETO_API_URL", hostile)
        monkeypatch.delenv("SCIENCEARENA_ALLOW_REMOTE", raising=False)
        with pytest.raises(RuntimeError, match="NOT a local service"):
            _adapter(*ADAPTERS[0])._config()


def test_third_party_ai_players_are_not_touched_by_this_guard():
    """SCOPE CONTROL. The directive covers OUR hosted tools only. If a future
    change widened this guard to the 103 players carrying a provider key, it
    would produce 103 false positives and zero true ones -- the arena's free-tier
    and subscription players are how it is meant to run."""
    import yaml
    reg = yaml.safe_load((REPO / "players" / "registry.yaml").read_text(encoding="utf-8"))
    guarded = {p["player_id"] for p in reg if p.get("api_url_env") == "SCIMETO_API_URL"}
    # Two of the original three were retired 2026-09-22 (see players/registry.yaml).
    # The assertion stays EXACT rather than becoming a subset check: a new
    # SCIMETO_API_URL player must be a deliberate change that updates this line,
    # not something that slips past a loosened test.
    assert guarded == {"scimeto-replication-api", "scimeto-replication-api-1-74-107"}, (
        f"the guarded set changed: {sorted(guarded)}")
    third_party = [p["player_id"] for p in reg if p.get("openai_api_key_env")]
    assert len(third_party) > 50, "expected the free/subscription player population to be large"
    assert not (guarded & set(third_party)), "the guard has leaked onto third-party AI players"

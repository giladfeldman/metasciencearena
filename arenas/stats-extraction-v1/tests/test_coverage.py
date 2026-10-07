"""Coverage check: the generator emits all 6 tiers."""
import importlib.util
import sys
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parents[1]

# Load the stats-extraction generator under a unique module name so it doesn't
# collide with sibling arenas that also expose a top-level `generator` module.
_SPEC = importlib.util.spec_from_file_location(
    "_stats_extraction_generator", ARENA_DIR / "generator.py"
)
generator = importlib.util.module_from_spec(_SPEC)
sys.modules["_stats_extraction_generator"] = generator
_SPEC.loader.exec_module(generator)

def test_generator_covers_all_six_tiers():
    tiers_seen = {t["input"]["tier"] for t in generator.generate("v1", seed=0)}
    assert tiers_seen == {1, 2, 3, 4, 5, 6}


def test_generator_produces_deception_items():
    tasks = list(generator.generate("v1", seed=0))
    # No registry scaffolding: `ground_truth()` serves from the in-process
    # _GROUND_TRUTH_CACHE that `generate()` just filled. The comment here used to
    # say it "reads gold from the article-finder registry", which stopped being
    # true at the registry-free migration — a factual claim about the call graph
    # that had quietly gone stale, in a file that is now PUBLISHED.
    seen_deception = False
    for t in tasks:
        gt = generator.ground_truth(t["task_id"])
        if any(item["deception_kind"] for item in gt["items"]):
            seen_deception = True
            break
    assert seen_deception

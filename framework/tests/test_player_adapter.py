"""Tests for the player adapter ABC and stubs used by other tests."""
import pytest
from framework.player_adapter import (
    HttpAdapter,
    PlayerAdapter,
    StubFailAdapter,
    StubPassAdapter,
    _extract_json_value,
    build_adapter,
)


class TestExtractJsonValue:
    """Robust JSON recovery from chatty text-CLI stdout.

    Regression guard for the Opus-4.8 siglang run where 10/21 tasks raised
    JSONDecodeError ('Expecting value' on empty output; 'Extra data' on a
    trailing sentence) and were unfairly scored 0.0.
    """

    def test_bare_object(self):
        assert _extract_json_value('{"flags": []}') == {"flags": []}

    def test_bare_array(self):
        assert _extract_json_value('[1, 2, 3]') == [1, 2, 3]

    def test_json_fence(self):
        assert _extract_json_value('```json\n{"flags": [1]}\n```') == {"flags": [1]}

    def test_plain_fence(self):
        assert _extract_json_value('```\n{"flags": [1]}\n```') == {"flags": [1]}

    def test_leading_prose(self):
        # Opus often prefaces the JSON with a sentence.
        assert _extract_json_value('Here is my analysis:\n{"flags": ["a"]}') == {"flags": ["a"]}

    def test_trailing_prose_extra_data(self):
        # The 'Extra data: line 3 column 1' case — a sentence AFTER the JSON.
        assert _extract_json_value('{"flags": []}\n\nLet me know if you need more.') == {"flags": []}

    def test_prose_both_sides_with_fence(self):
        raw = 'Sure!\n```json\n{"flags": [{"x": 1}]}\n```\nHope that helps.'
        assert _extract_json_value(raw) == {"flags": [{"x": 1}]}

    def test_nested_braces_and_strings(self):
        raw = 'note {ignored} \n{"a": {"b": "}"}, "c": [1, 2]}'
        assert _extract_json_value(raw) == {"a": {"b": "}"}, "c": [1, 2]}

    def test_empty_output_raises(self):
        # 'Expecting value: line 1 column 1' — empty/whitespace must stay an honest error.
        with pytest.raises(ValueError):
            _extract_json_value("")
        with pytest.raises(ValueError):
            _extract_json_value("   \n  ")

    def test_pure_prose_refusal_raises(self):
        with pytest.raises(ValueError):
            _extract_json_value("I cannot help with that request.")


def test_stub_pass_returns_label_ok():
    adapter = StubPassAdapter(player_id="stub-pass", player_version="0.0.1", player_type="tool",
                              confidence_strategy="implicit-1.0", deterministic=True)
    out = adapter.play_task({"task_id": "t1", "arena_id": "x", "task_set_version": "v1", "difficulty": {}, "input": {"text": "hi"}}, timeout_s=5)
    assert out == {"label": "ok"}


def test_stub_fail_raises():
    adapter = StubFailAdapter(player_id="stub-fail", player_version="0.0.1", player_type="tool",
                              confidence_strategy="implicit-1.0", deterministic=True)
    with pytest.raises(RuntimeError):
        adapter.play_task({"task_id": "t1", "arena_id": "x", "task_set_version": "v1", "difficulty": {}, "input": {"text": "hi"}}, timeout_s=5)


def test_build_adapter_dispatches_on_class_name():
    entry = {"player_id": "stub-pass", "player_version": "0.0.1", "player_type": "tool",
             "adapter_class": "StubPassAdapter", "confidence_strategy": "implicit-1.0", "deterministic": True}
    adapter = build_adapter(entry)
    assert isinstance(adapter, StubPassAdapter)
    assert adapter.player_id == "stub-pass"


def test_build_adapter_unknown_class_raises():
    with pytest.raises(ValueError):
        build_adapter({"player_id": "x", "player_version": "1", "player_type": "tool",
                       "adapter_class": "DoesNotExist", "confidence_strategy": "native", "deterministic": True})


def test_abc_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        PlayerAdapter(player_id="x", player_version="1", player_type="tool",
                      confidence_strategy="native", deterministic=True)


def _escimate_adapter():
    return HttpAdapter(
        player_id="escimate", player_version="0.6.2", player_type="platform",
        confidence_strategy="implicit-1.0", deterministic=True,
        endpoint="http://127.0.0.1:9422/api/v1/process-text",
    )


def test_normalize_kind_uses_df_presence_to_split_nhst_from_effect_size():
    # The ambiguous symbols r/chi2 are both a test statistic and an effect size;
    # df-presence decides. A bare "r = -0.85" (no df) must be effect_size, not
    # nhst_stat (the kind_mismatch the arena report flagged).
    adapter = _escimate_adapter()
    text = ("r(28) = 0.48, p = .01. The effect was r = -0.85, 95% CI "
            "[-1.03, -0.71]. Cohen's d = 0.42. z = 1.96, p = .05.")
    resp = {"results": [
        {"raw_text": "r(28) = 0.48", "stat_value": 0.48, "test_type": "r",
         "df1": 28, "check_type": "effect_size"},
        {"raw_text": "r = -0.85", "effect_reported": -0.85, "test_type": "r",
         "check_type": "extraction_only"},
        {"raw_text": "Cohen's d = 0.42", "effect_reported": 0.42, "test_type": "",
         "check_type": "extraction_only"},
        {"raw_text": "z = 1.96", "stat_value": 1.96, "test_type": "z",
         "check_type": "effect_size"},
    ]}
    out = adapter._normalize(resp, text)
    assert [e["kind"] for e in out["extractions"]] == [
        "nhst_stat", "effect_size", "effect_size", "nhst_stat"]
    # span is anchored on the raw_text substring (offset integrity)
    first = out["extractions"][0]["span"]
    assert text[first["char_start"]:first["char_end"]] == first["text"] == "r(28) = 0.48"

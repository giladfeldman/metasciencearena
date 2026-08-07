"""Scorer tests for code-translation-r-v1.

These are the arena's validity checks, not just unit tests. They assert that:
  * the hand-verified reference translation scores a perfect 1.0 (oracle check);
  * a NAIVE translation — one that carries a cross-language default over
    unchanged — scores strictly worse even though its code runs cleanly.

If the second property ever stops holding, the arena has stopped measuring the
thing it claims to measure, and that must fail loudly.

Tests that need R are skipped when R is absent, so the suite still runs on a
machine without it — but the skip is explicit, never a silent pass.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ARENA_DIR = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


generator = _load("_xlat_generator", ARENA_DIR / "generator.py")
scorer = _load("_xlat_scorer", ARENA_DIR / "scorer.py")
r_runner = _load("_xlat_r_runner", ARENA_DIR / "r_runner.py")

REF_DIR = ARENA_DIR / "source_scripts" / "reference_r"
needs_r = pytest.mark.skipif(r_runner.find_rscript() is None,
                             reason="no local R — executable-equivalence scoring unavailable")


def _gt(analysis: str, language: str = "spss") -> dict:
    for env in generator.generate("v1", 0):
        gt = generator.ground_truth(env["task_id"])
        if gt["analysis_id"] == analysis and gt["source_language"] == language:
            return gt
    raise AssertionError(f"no task for {analysis}/{language}")


ALL_ANALYSES = [
    "descriptives", "ttest_groups", "regression_multi",
    "recode_transform", "anova_factorial", "pipeline_select_model",
    # T7-T9, added 2026-08-04 because T1-T6 saturated (sonnet-5 scored 1.000
    # on every task, so the arena stopped discriminating at the top).
    "correlations_pairwise", "weighted_descriptives", "split_file_groups",
]


@needs_r
@pytest.mark.parametrize("analysis", ALL_ANALYSES)
def test_reference_translation_scores_perfect(analysis):
    """The oracle check: gold's own producer must score 1.0."""
    code = (REF_DIR / f"{analysis}.R").read_text(encoding="utf-8")
    res = scorer.score({"r_code": code}, _gt(analysis))
    assert res["primary"] == pytest.approx(1.0), res["breakdown"]
    assert res["breakdown"]["execution_rate"] == 1.0
    assert not res["findings"], res["findings"]


@needs_r
def test_naive_welch_translation_is_penalised():
    """T2 trap: R's t.test defaults to Welch; SPSS/Stata headline the pooled test."""
    naive = '''
df <- read.csv(Sys.getenv("ARENA_DATA"), stringsAsFactors = FALSE)
tt <- t.test(score ~ group, data = df)
m <- tapply(df$score, df$group, mean, na.rm = TRUE)
out <- list(t_statistic = unname(tt$statistic), df = unname(tt$parameter),
            p_value = tt$p.value, mean_diff = unname(m[["1"]] - m[["2"]]))
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
'''
    res = scorer.score({"r_code": naive}, _gt("ttest_groups"))
    # It RUNS — that is the point. It is wrong anyway.
    assert res["breakdown"]["execution_rate"] == 1.0
    assert res["primary"] < 1.0
    assert res["breakdown"]["n_statistics_wrong"] >= 3
    assert any(f["category"] == "wrong_statistic" for f in res["findings"])


@needs_r
def test_type_one_ss_translation_is_penalised():
    """T5 trap: anova(lm(...)) is Type I; SPSS/Stata report Type III."""
    naive = '''
df <- read.csv(Sys.getenv("ARENA_DATA"))
df$group <- factor(df$group); df$condition <- factor(df$condition)
tab <- anova(lm(score ~ group * condition, data = df))
out <- list(f_group = tab["group", "F value"], f_condition = tab["condition", "F value"],
            f_interaction = tab["group:condition", "F value"], df_resid = tab["Residuals", "Df"])
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
'''
    res = scorer.score({"r_code": naive}, _gt("anova_factorial"))
    assert res["breakdown"]["execution_rate"] == 1.0
    assert res["primary"] < 1.0


@needs_r
def test_type_three_without_contrasts_is_still_penalised():
    """The subtle T5 trap: type=3 with R's default treatment contrasts is wrong.

    This is the mistake a translator is most likely to make while believing it
    has handled the SS-type issue, so it must not score as correct.
    """
    almost = '''
df <- read.csv(Sys.getenv("ARENA_DATA"))
df$group <- factor(df$group); df$condition <- factor(df$condition)
tab <- car::Anova(lm(score ~ group * condition, data = df), type = 3)
out <- list(f_group = tab["group", "F value"], f_condition = tab["condition", "F value"],
            f_interaction = tab["group:condition", "F value"], df_resid = tab["Residuals", "Df"])
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
'''
    res = scorer.score({"r_code": almost}, _gt("anova_factorial"))
    assert res["breakdown"]["execution_rate"] == 1.0
    assert res["primary"] < 1.0


@needs_r
def test_ignoring_user_missing_code_is_penalised():
    """T4 trap: leaving 99 as a literal value inflates the index."""
    naive = '''
df <- read.csv(Sys.getenv("ARENA_DATA"))
df$item1_high <- ifelse(df$item1 >= 4, 1, 0)
items <- df[, c("item1","item2","item3","item4")]
df$index <- rowMeans(items, na.rm = TRUE)
out <- list(n_recoded_high = sum(df$item1_high == 1, na.rm = TRUE),
            mean_index = mean(df$index, na.rm = TRUE),
            n_missing_after = sum(is.na(df$item4)))
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
'''
    res = scorer.score({"r_code": naive}, _gt("recode_transform"))
    assert res["breakdown"]["execution_rate"] == 1.0
    assert res["primary"] < 1.0


@needs_r
def test_unrunnable_code_scores_zero():
    res = scorer.score({"r_code": "this is not R at all ((("}, _gt("descriptives"))
    assert res["primary"] == 0.0
    assert res["breakdown"]["execution_rate"] == 0.0
    assert res["findings"][0]["category"] == "does_not_execute"


@needs_r
def test_hallucinated_function_scores_zero():
    """A confident call to a function that does not exist must not earn credit."""
    code = '''
df <- read.csv(Sys.getenv("ARENA_DATA"))
out <- spss_descriptives_magic(df)
cat(jsonlite::toJSON(out, auto_unbox = TRUE))
'''
    res = scorer.score({"r_code": code}, _gt("descriptives"))
    assert res["primary"] == 0.0
    assert res["findings"][0]["category"] == "does_not_execute"


@needs_r
def test_partial_output_scores_partially():
    """Emitting some statistics earns proportional credit, not all-or-nothing."""
    code = '''
df <- read.csv(Sys.getenv("ARENA_DATA"))
out <- list(mean_age = mean(df$age, na.rm = TRUE), sd_age = sd(df$age, na.rm = TRUE))
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
'''
    res = scorer.score({"r_code": code}, _gt("descriptives"))
    b = res["breakdown"]
    assert 0.0 < res["primary"] < 1.0
    assert b["n_statistics_matched"] == 2
    assert b["n_statistics_missing"] == 4
    assert any(f["category"] == "missing_statistic" for f in res["findings"])


@needs_r
def test_jmv_idiom_scores_on_accuracy_not_on_print_contract():
    """A jmv translation must score on its NUMBERS, not on how it reports them.

    This is the arena's central fairness property. An earlier scorer required
    every player to print a JSON block, which scored real converters 0.00 for
    emitting perfectly correct analyses in their own idiom — measuring
    conformance to our harness rather than translation quality, and making
    deterministic tools structurally unable to compete with LLMs (who simply
    follow instructions better).

    jmv assigns nothing, prints no JSON, and upper-cases variable names. It must
    still score 1.0 when the numbers are right.
    """
    code = '''
jmv::descriptives(data = data, vars = c('AGE', 'SCORE'),
                  mean = TRUE, sd = TRUE, missing = TRUE)
'''
    res = scorer.score({"r_code": code}, _gt("descriptives"))
    b = res["breakdown"]
    assert b["execution_rate"] == 1.0, res["findings"]
    # mean/sd/n for AGE and SCORE are all six required statistics.
    assert b["n_statistics_matched"] == 6, res["findings"]
    assert res["primary"] == pytest.approx(1.0)


@needs_r
def test_jmv_ttest_pooled_variant_is_harvested():
    """jmv suffixes columns by test variant (`stat[stud]` vs `stat[welc]`).

    The pooled (Student's) variant is the one SPSS and Stata headline, so it is
    the one that must be compared to gold — matching only a bare `stat` column
    found nothing and scored a numerically perfect translation 0.
    """
    code = '''
jmv::ttestIS(data = data, vars = c("SCORE"), group = "GROUP",
             students = TRUE, welchs = TRUE, meanDiff = TRUE)
'''
    res = scorer.score({"r_code": code}, _gt("ttest_groups"))
    assert res["primary"] == pytest.approx(1.0), res["findings"]


@needs_r
def test_unassigned_final_expression_is_still_harvested():
    """Converters end with a bare call whose result is printed, never assigned.

    Discarding unassigned top-level values would score a translation 0 for a
    stylistic choice.
    """
    code = 'summary(lm(score ~ age + hours, data = data))\nlm(score ~ age + hours, data = data)\n'
    res = scorer.score({"r_code": code}, _gt("regression_multi"))
    assert res["breakdown"]["n_statistics_matched"] >= 4, res["findings"]


@needs_r
def test_uniform_deletion_rule_is_penalised():
    """T7 trap: SPSS CORRELATIONS is PAIRWISE, REGRESSION is LISTWISE.

    The PAIR is the trap — a translator that applies one missing-data convention
    uniformly gets exactly one of the two matrices wrong. Verified measurable:
    pairwise N=171 vs listwise N=166 on the fixed dataset.
    """
    naive = '''
df <- read.csv(Sys.getenv("ARENA_DATA")); v <- df[, c("age","hours","score")]
r <- cor(v, use = "complete.obs"); n <- sum(complete.cases(v))
out <- list(r_pairwise_age_hours = r["age","hours"], n_pairwise_age_hours = n,
            r_listwise_age_hours = r["age","hours"], n_listwise_age_hours = n)
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
'''
    res = scorer.score({"r_code": naive}, _gt("correlations_pairwise"))
    assert res["breakdown"]["execution_rate"] == 1.0, "must RUN — that is the point"
    assert res["primary"] < 1.0, res["breakdown"]


@needs_r
def test_dropping_weight_by_is_penalised():
    """T8 trap: SPSS WEIGHT BY is stateful; base R has no weighting state.

    mean() silently ignores weights, so a translation that drops the WEIGHT
    statement returns the unweighted mean and looks perfectly reasonable. The
    weight is correlated with score here, so the two differ by ~5 points.
    """
    naive = '''
df <- read.csv(Sys.getenv("ARENA_DATA"))
out <- list(weighted_mean_score = mean(df$score, na.rm = TRUE),
            weighted_n = sum(!is.na(df$score)))
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
'''
    res = scorer.score({"r_code": naive}, _gt("weighted_descriptives"))
    assert res["breakdown"]["execution_rate"] == 1.0
    assert res["primary"] < 1.0, res["breakdown"]


@needs_r
def test_ignoring_split_file_is_penalised():
    """T9 trap: SPLIT FILE runs every following procedure PER GROUP.

    It has no R equivalent — the correct translation restructures the program
    into a grouped operation. A translator that ignores the split reports one
    pooled mean and silently answers a different question.
    """
    naive = '''
df <- read.csv(Sys.getenv("ARENA_DATA"))
m <- mean(df$score, na.rm = TRUE); n <- sum(!is.na(df$score))
out <- list(mean_score_group1 = m, mean_score_group2 = m, n_group1 = n, n_group2 = n)
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 10))
'''
    res = scorer.score({"r_code": naive}, _gt("split_file_groups"))
    assert res["breakdown"]["execution_rate"] == 1.0
    assert res["primary"] < 1.0, res["breakdown"]


def test_bool_never_matches_a_non_bool():
    """Found by cross-model review (codex + sonnet-5), 2026-08-04.

    `_close` used to coerce through bool() whenever EITHER side was a bool, so
    `_close(2, True)` and `_close("x", True)` both returned True — a player
    emitting a non-numeric value could be credited for a statistic it never
    computed. On a scientific benchmark that publishes tool comparisons, a
    false credit is worse than a false failure.
    """
    assert scorer._close(2, True) is False
    assert scorer._close("x", True) is False
    assert scorer._close(1, True) is False
    assert scorer._close(0, False) is False
    assert scorer._close(True, True) is True
    assert scorer._close(False, False) is True


def test_counts_and_df_compare_exactly_not_within_tolerance():
    """Found by cross-model review, 2026-08-04.

    A relative tolerance is wrong for counts: at n=1,000,000 rel_tol=1e-6
    accepts an off-by-one, and "analysed one more case than SPSS did" is exactly
    the listwise/pairwise deletion error this arena exists to catch.
    """
    assert scorer.is_exact_statistic("n_analysed")
    assert scorer.is_exact_statistic("n_age")
    assert scorer.is_exact_statistic("df")
    assert scorer.is_exact_statistic("df_resid")
    assert scorer.is_exact_statistic("resid_df")
    assert not scorer.is_exact_statistic("mean_age")
    assert not scorer.is_exact_statistic("p_value")

    assert scorer._close(1_000_000, 1_000_001, exact=True) is False
    assert scorer._close(178, 178.00001, exact=True) is False
    assert scorer._close(172, 172, exact=True) is True
    # continuous statistics keep the tolerance
    assert scorer._close(52.9442111111, 52.94421111115) is True


def test_missing_gold_is_charged_to_the_arena_not_the_player():
    """Found by cross-model review, 2026-08-04.

    A required statistic absent from gold used to compare against
    `gold.get(key)` -> None and be recorded as the PLAYER's wrong answer. It is
    an arena defect: it must not count against the player, and must leave the
    denominator.
    """
    gt = dict(_gt("descriptives"))
    gt["gold_statistics"] = {k: v for k, v in gt["gold_statistics"].items()
                             if k != "sd_score"}
    code = 'cat(\'{"mean_age":1,"sd_age":1,"n_age":1,"mean_score":1,"sd_score":1,"n_score":1}\')'
    res = scorer.score({"r_code": code}, gt)
    b = res["breakdown"]
    assert b["n_statistics_no_gold"] == 1
    assert b["n_statistics_judgeable"] == len(gt["required_statistics"]) - 1
    assert "sd_score" not in [f["detail"] for f in res["findings"] if "wrong" in f["category"]]
    assert any("Gold has no value" in f["detail"] for f in res["findings"])


@needs_r
def test_an_unrelated_htest_does_not_supply_the_t_statistic():
    """Found by cross-model review, 2026-08-04 — the highest-severity finding.

    A translation legitimately runs assumption checks beside the target
    analysis; SPSStoR's REAL output calls car::leveneTest() before t.test().
    Every such check is also class "htest", and the harvester took the first one
    it walked. Shapiro's W (0.986) would have been scored as the t-statistic
    (gold -3.147) — a wrong number, silently, which is the worst possible
    failure for a benchmark.
    """
    code = '''
sh <- shapiro.test(data$score)                              # htest, NOT a t-test
tt <- t.test(score ~ group, data = data, var.equal = TRUE)  # the real target
m  <- tapply(data$score, data$group, mean, na.rm = TRUE)
md <- unname(m[["1"]] - m[["2"]])
'''
    res = scorer.score({"r_code": code}, _gt("ttest_groups"))
    b = res["breakdown"]
    # t_statistic/df/p_value must come from the t-test, so all three match gold.
    assert b["n_statistics_matched"] >= 3, res["findings"]
    assert res["primary"] >= 0.75, res["findings"]


def test_empty_output_scores_zero_without_r():
    """Guard rails that need no R at all."""
    gt = _gt("descriptives")
    for bad in ({}, {"r_code": ""}, {"r_code": None}):
        res = scorer.score(bad, gt)
        assert res["primary"] == 0.0
        assert res["findings"][0]["category"] == "does_not_execute"


def test_missing_gold_is_reported_as_error_not_player_failure():
    gt = dict(_gt("descriptives"))
    gt["gold_statistics"] = None
    res = scorer.score({"r_code": "cat('{}')"}, gt)
    assert res["breakdown"] == {"error": "gold_not_built"}

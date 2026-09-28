"""Regression tests for statistical-correctness fixes (TODO.md #37-#43, #88, #89)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.statistics import ABTestAnalyzer, power_analysis
from src.statistics.diagnostics import (
    describe_statistical_fallbacks,
    resolve_expected_treatment_ratio,
    validate_expected_treatment_ratio,
)
from src.statistics.engine_helpers import combine_total_effect_per_customer
from src.statistics.statsmodels_engine import run_two_proportion_test
from src.statistics.summary_builder import ABTestSummaryBuilder


def _analyzer_for(df: pd.DataFrame, **mapping_extra: Any) -> ABTestAnalyzer:
    analyzer = ABTestAnalyzer()
    analyzer.set_dataframe(df)
    mapping: Dict[str, Any] = {
        "group": "group",
        "effect_value": "post_effect",
        "post_effect": "post_effect",
    }
    if "pre_effect" in df.columns:
        mapping["pre_effect"] = "pre_effect"
    mapping.update(mapping_extra)
    analyzer.set_column_mapping(mapping)
    analyzer.set_group_labels("treatment", "control")
    return analyzer


def _revenue_frame(n_t: int, n_c: int, seed: int = 0) -> pd.DataFrame:
    """Zero-inflated revenue: treatment converts more AND spends more."""
    rng = np.random.default_rng(seed)
    t_conv = rng.random(n_t) < 0.40
    c_conv = rng.random(n_c) < 0.25
    t_rev = np.where(t_conv, rng.normal(100, 10, n_t), 0.0)
    c_rev = np.where(c_conv, rng.normal(100, 10, n_c), 0.0)
    return pd.DataFrame({
        "group": ["treatment"] * n_t + ["control"] * n_c,
        "post_effect": np.concatenate([t_rev, c_rev]),
    })


# ---------------------------------------------------------------------------
# #42 — total_effect must not add the proportion effect on top of the mean
# ---------------------------------------------------------------------------
class TestTotalEffectNoDoubleCount:
    def test_combiner_prefers_mean_difference(self) -> None:
        assert combine_total_effect_per_customer(
            effect_size=15.0, is_significant=True, proportion_effect_per_customer=4.0
        ) == 15.0
        assert combine_total_effect_per_customer(
            effect_size=15.0, is_significant=False, proportion_effect_per_customer=4.0
        ) == 4.0
        assert combine_total_effect_per_customer(
            effect_size=15.0, is_significant=False, proportion_effect_per_customer=0.0
        ) == 0.0

    def test_both_significant_total_equals_mean_difference(self) -> None:
        result = _analyzer_for(_revenue_frame(2000, 2000)).run_ab_test()

        assert result.is_significant
        assert result.proportion_is_significant
        assert result.proportion_effect > 0  # still reported, informationally
        assert result.total_effect_per_customer == pytest.approx(result.effect_size)
        assert result.total_effect == pytest.approx(result.effect_size * result.treatment_size)


# ---------------------------------------------------------------------------
# #89 — CUPED residuals must not leak into proportion / DiD / Bayesian paths
# ---------------------------------------------------------------------------
class TestCupedIsolation:
    def test_binary_metric_skips_cuped_and_keeps_true_proportions(self) -> None:
        rng = np.random.default_rng(1)
        n = 1500
        pre = rng.normal(10, 2, 2 * n)
        post = (rng.random(2 * n) < np.r_[np.full(n, 0.30), np.full(n, 0.20)]).astype(float)
        df = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "pre_effect": pre,
            "post_effect": post,
        })

        result = _analyzer_for(df, cuped=True).run_ab_test()

        assert result.cuped_applied is False
        assert result.treatment_proportion == pytest.approx(post[:n].mean())
        assert result.control_proportion == pytest.approx(post[n:].mean())
        assert result.treatment_proportion < 0.5  # not the ~1.0 residual artefact
        assert any("CUPED" in w for w in result.statistical_warnings)

    def test_continuous_cuped_only_adjusts_primary_estimate(self) -> None:
        rng = np.random.default_rng(2)
        n = 1500
        pre = rng.normal(50, 10, 2 * n)
        converted = rng.random(2 * n) < 0.6
        post = np.where(converted, pre + rng.normal(0, 3, 2 * n), 0.0)
        post[:n] = np.where(converted[:n], post[:n] + 2.0, 0.0)
        df = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "pre_effect": pre,
            "post_effect": post,
        })

        baseline = _analyzer_for(df).run_ab_test()
        adjusted = _analyzer_for(df, cuped=True).run_ab_test()

        assert adjusted.cuped_applied is True
        # Proportion test sees raw zeros, not residuals (which are ~never 0).
        assert adjusted.treatment_proportion == pytest.approx(baseline.treatment_proportion)
        assert adjusted.control_proportion == pytest.approx(baseline.control_proportion)
        # Post means, DiD and Bayesian stay on the metric's own scale.
        assert adjusted.treatment_post_mean == pytest.approx(baseline.treatment_post_mean)
        assert adjusted.did_effect == pytest.approx(baseline.did_effect)
        assert adjusted.bayesian_credible_interval == pytest.approx(
            baseline.bayesian_credible_interval
        )


# ---------------------------------------------------------------------------
# #40 — SRM must use the designed allocation, not a hardcoded 50/50
# ---------------------------------------------------------------------------
class TestAllocationAwareSrm:
    @staticmethod
    def _holdout_frame() -> pd.DataFrame:
        rng = np.random.default_rng(3)
        n_t, n_c = 900, 100
        return pd.DataFrame({
            "group": ["treatment"] * n_t + ["control"] * n_c,
            "post_effect": np.concatenate([rng.normal(12, 2, n_t), rng.normal(10, 2, n_c)]),
        })

    def test_default_ratio_flags_intentional_holdout(self) -> None:
        result = _analyzer_for(self._holdout_frame()).run_ab_test()
        srm = result.diagnostics["experiment_quality"]["srm"]
        assert srm["is_sample_ratio_mismatch"] is True
        assert result.is_significant is False

    def test_declared_ratio_clears_srm_and_keeps_significance(self) -> None:
        result = _analyzer_for(self._holdout_frame(), expected_treatment_ratio=0.9).run_ab_test()
        srm = result.diagnostics["experiment_quality"]["srm"]
        assert srm["expected_treatment_ratio"] == pytest.approx(0.9)
        assert srm["is_sample_ratio_mismatch"] is False
        assert result.is_significant is True

    def test_analyzer_level_default_ratio(self) -> None:
        analyzer = ABTestAnalyzer(expected_treatment_ratio=0.9)
        analyzer.set_dataframe(self._holdout_frame())
        analyzer.set_column_mapping({"group": "group", "effect_value": "post_effect"})
        analyzer.set_group_labels("treatment", "control")
        srm = analyzer.run_ab_test().diagnostics["experiment_quality"]["srm"]
        assert srm["is_sample_ratio_mismatch"] is False

    def test_group_imbalance_recommendation_respects_design(self) -> None:
        result = _analyzer_for(self._holdout_frame(), expected_treatment_ratio=0.9).run_ab_test()
        recs = ABTestSummaryBuilder()._generate_recommendations([result])
        assert not any(rec.startswith("GROUP IMBALANCE") for rec in recs)

    @pytest.mark.parametrize("bad", [0, 1, 1.5, -0.2, "abc", float("nan")])
    def test_invalid_ratio_rejected(self, bad: Any) -> None:
        with pytest.raises(ValueError):
            validate_expected_treatment_ratio(bad)

    def test_resolver_prefers_mapping(self) -> None:
        assert resolve_expected_treatment_ratio({"expected_treatment_ratio": "0.2"}, 0.5) == 0.2
        assert resolve_expected_treatment_ratio({}, 0.3) == 0.3
        assert resolve_expected_treatment_ratio(None) == 0.5


# ---------------------------------------------------------------------------
# #39 — adequacy from design sensitivity (MDE), not observed power
# ---------------------------------------------------------------------------
class TestMdeAdequacy:
    def test_large_null_experiment_is_adequate_despite_low_observed_power(self) -> None:
        rng = np.random.default_rng(4)
        n = 5000
        df = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "post_effect": rng.normal(10, 2, 2 * n),
        })
        result = _analyzer_for(df).run_ab_test()

        assert result.power < 0.8  # observed power on a ~zero effect is tiny
        assert result.achieved_mde < result.target_effect_size
        assert result.is_sample_adequate is True

    def test_tiny_experiment_with_big_observed_effect_is_not_adequate(self) -> None:
        rng = np.random.default_rng(5)
        n = 20
        df = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "post_effect": np.concatenate([rng.normal(14, 2, n), rng.normal(10, 2, n)]),
        })
        result = _analyzer_for(df).run_ab_test()

        assert result.power > 0.8  # the observed-power fallacy would call this adequate
        assert result.is_sample_adequate is False
        # Required N is planned for the target effect, not the observed one.
        assert result.required_sample_size == power_analysis.calculate_required_sample_size(
            effect_size=power_analysis.DEFAULT_TARGET_EFFECT_SIZE,
            ratio=1.0,
            power_threshold=0.8,
            significance_level=0.05,
        )

    def test_target_effect_size_override(self) -> None:
        rng = np.random.default_rng(6)
        n = 300
        df = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "post_effect": rng.normal(10, 2, 2 * n),
        })
        strict = _analyzer_for(df, target_effect_size=0.05).run_ab_test()
        lenient = _analyzer_for(df, target_effect_size=0.5).run_ab_test()
        assert strict.is_sample_adequate is False
        assert lenient.is_sample_adequate is True
        assert strict.target_effect_size == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# #43 — fallbacks become visible warnings that reach the summary
# ---------------------------------------------------------------------------
class TestFallbackWarnings:
    def test_reason_codes_are_translated(self) -> None:
        msgs = describe_statistical_fallbacks(
            "Primary effect model",
            {"reasons": ["small_sample_size", "model_fit_failed_fallback_to_ols"]},
            model_type="compare_means_fallback",
        )
        assert len(msgs) == 2
        assert all(m.startswith("Primary effect model:") for m in msgs)

    def test_power_solver_failure_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_args: Any, **_kwargs: Any) -> float:
            raise RuntimeError("solver exploded")

        monkeypatch.setattr(power_analysis.TTestIndPower, "solve_power", boom)
        sink: list[str] = []
        assert power_analysis.calculate_power(
            effect_size=0.3, n_treatment=50, n_control=50,
            significance_level=0.05, warnings_sink=sink,
        ) == 0.0
        assert power_analysis.calculate_required_sample_size(
            effect_size=0.3, ratio=1.0, power_threshold=0.8,
            significance_level=0.05, warnings_sink=sink,
        ) == power_analysis.REQUIRED_SAMPLE_SIZE_UNREACHABLE
        assert len(sink) == 2

    def test_proportion_failure_reaches_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.statistics.statsmodels_engine as engine_module

        def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("proportion test exploded")

        monkeypatch.setattr(engine_module, "test_proportions_2indep", boom)
        analyzer = _analyzer_for(_revenue_frame(500, 500))
        result = analyzer.run_ab_test()

        assert result.proportion_p_value == 1.0
        assert any(w.startswith("Proportion test:") for w in result.statistical_warnings)
        summary = analyzer.generate_summary([result])
        assert any("Proportion test" in w for w in summary.analysis_warnings)


# ---------------------------------------------------------------------------
# #37 — count-based proportion test and its guardrails
# ---------------------------------------------------------------------------
class TestCountBasedProportionTest:
    def test_counts_match_array_based_score_test(self) -> None:
        engine = ABTestAnalyzer().stats_engine
        t = np.r_[np.ones(50), np.zeros(50)]
        c = np.r_[np.ones(30), np.zeros(70)]
        from_arrays = engine.run_proportion_test(t, c)
        from_counts = run_two_proportion_test(
            treatment_conversions=50, n_treatment=100, control_conversions=30, n_control=100
        )
        assert from_counts["p_value"] == pytest.approx(from_arrays["p_value"])
        assert from_counts["z_stat"] == pytest.approx(from_arrays["z_stat"])

    def test_small_expected_cells_block_significance(self) -> None:
        # 3 vs 0 conversions: expected cell counts too small.
        result = run_two_proportion_test(
            treatment_conversions=3, n_treatment=1000, control_conversions=0, n_control=1000
        )
        assert result["diagnostics"]["blocks_significance"] is True

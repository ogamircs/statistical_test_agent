"""Direct unit tests for metric-family detection."""

from __future__ import annotations

import numpy as np
import pytest

from src.statistics.model_families import (
    infer_metric_type,
    is_binary_metric,
    is_count_metric,
    is_heavy_tail_metric,
)

VARIANCE_EPSILON = 1e-12


def test_is_binary_metric_zero_one() -> None:
    assert is_binary_metric(np.array([0, 1, 0, 1])) is True


def test_is_binary_metric_rejects_continuous() -> None:
    assert is_binary_metric(np.array([0.1, 0.5, 0.9])) is False


def test_is_count_metric_nonneg_integers() -> None:
    assert is_count_metric(np.array([0, 1, 5, 10])) is True
    assert is_count_metric(np.array([0, 1, 5, 10, 2, 3])) is True


def test_is_count_metric_rejects_negatives() -> None:
    assert is_count_metric(np.array([-1, 2, 3])) is False


def test_is_heavy_tail_metric_detects_skewed_distribution() -> None:
    rng = np.random.default_rng(0)
    heavy = rng.lognormal(mean=0.0, sigma=2.0, size=500)
    assert is_heavy_tail_metric(heavy, variance_epsilon=VARIANCE_EPSILON) is True


def test_is_heavy_tail_metric_rejects_normal() -> None:
    rng = np.random.default_rng(1)
    # Mean far from zero so the p99/p50 ratio rule cannot accidentally fire.
    normal = rng.normal(50.0, 1.0, size=500)
    assert is_heavy_tail_metric(normal, variance_epsilon=VARIANCE_EPSILON) is False


def test_is_heavy_tail_metric_too_small_returns_false() -> None:
    assert is_heavy_tail_metric(np.array([1.0, 2.0, 3.0]), variance_epsilon=VARIANCE_EPSILON) is False


def test_infer_metric_type_routes_to_binary() -> None:
    assert (
        infer_metric_type(np.array([0, 1, 0, 1]), variance_epsilon=VARIANCE_EPSILON)
        == "binary"
    )


def test_infer_metric_type_routes_to_count() -> None:
    assert (
        infer_metric_type(
            np.array([0, 1, 5, 10, 2, 3]), variance_epsilon=VARIANCE_EPSILON
        )
        == "count"
    )


def test_infer_metric_type_routes_to_heavy_tail() -> None:
    rng = np.random.default_rng(2)
    heavy = rng.lognormal(mean=0.0, sigma=2.5, size=500)
    assert infer_metric_type(heavy, variance_epsilon=VARIANCE_EPSILON) == "heavy_tail"


def test_infer_metric_type_falls_through_to_continuous() -> None:
    rng = np.random.default_rng(3)
    normal = rng.normal(50.0, 10.0, size=500)
    assert infer_metric_type(normal, variance_epsilon=VARIANCE_EPSILON) == "continuous"


def test_infer_metric_type_respects_explicit_request() -> None:
    assert (
        infer_metric_type(
            np.array([0, 1, 0, 1]),
            requested_metric_type="continuous",
            variance_epsilon=VARIANCE_EPSILON,
        )
        == "continuous"
    )


def test_deterministic_small_n_blocks_significance() -> None:
    """A tiny sample with zero variance in both arms must not read as
    significant: every other branch honors small_n (TODO.md #41)."""
    from src.statistics.model_families import estimate_treatment_effect

    result = estimate_treatment_effect(
        treatment_data=np.array([5.0, 5.0]),
        control_data=np.array([3.0, 3.0]),
        significance_level=0.05,
        min_recommended_sample_size=8,
        variance_epsilon=VARIANCE_EPSILON,
    )

    assert result["model_type"] == "deterministic"
    assert result["p_value"] == 0.0
    assert result["diagnostics"]["blocks_significance"] is True
    assert "small_sample_size" in result["diagnostics"]["reasons"]


def test_deterministic_adequate_n_does_not_block_significance() -> None:
    """With an adequate sample, a deterministic difference may stay
    significant — only the small-n case is blocked."""
    from src.statistics.model_families import estimate_treatment_effect

    result = estimate_treatment_effect(
        treatment_data=np.full(20, 5.0),
        control_data=np.full(20, 3.0),
        significance_level=0.05,
        min_recommended_sample_size=8,
        variance_epsilon=VARIANCE_EPSILON,
    )

    assert result["model_type"] == "deterministic"
    assert result["diagnostics"]["blocks_significance"] is False


def test_binary_glm_primary_ci_is_on_risk_difference_scale() -> None:
    """For a binary metric the primary CI must bracket the raw proportion
    difference, not the log-odds coefficient (TODO.md #36). The delta-method
    interval for a saturated binomial GLM equals the Wald two-proportion CI."""
    from statsmodels.stats.proportion import confint_proportions_2indep

    from src.statistics.model_families import estimate_treatment_effect

    rng = np.random.default_rng(42)
    treatment = rng.binomial(1, 0.62, 500).astype(float)
    control = rng.binomial(1, 0.50, 500).astype(float)

    result = estimate_treatment_effect(
        treatment_data=treatment,
        control_data=control,
        significance_level=0.05,
        min_recommended_sample_size=30,
        variance_epsilon=VARIANCE_EPSILON,
        metric_type="binary",
    )

    assert result["model_type"] == "glm_binomial"
    ci_low, ci_high = result["confidence_interval"]
    expected_low, expected_high = confint_proportions_2indep(
        int(treatment.sum()),
        len(treatment),
        int(control.sum()),
        len(control),
        method="wald",
        alpha=0.05,
    )
    assert ci_low == pytest.approx(expected_low, abs=1e-8)
    assert ci_high == pytest.approx(expected_high, abs=1e-8)
    assert ci_low < result["effect_size"] < ci_high


def test_binary_glm_model_ci_stays_on_log_odds_scale() -> None:
    """The model-scale interval must remain available explicitly on
    ``model_confidence_interval`` (log-odds), bracketing the log odds ratio."""
    from src.statistics.model_families import estimate_treatment_effect

    rng = np.random.default_rng(42)
    treatment = rng.binomial(1, 0.62, 500).astype(float)
    control = rng.binomial(1, 0.50, 500).astype(float)

    result = estimate_treatment_effect(
        treatment_data=treatment,
        control_data=control,
        significance_level=0.05,
        min_recommended_sample_size=30,
        variance_epsilon=VARIANCE_EPSILON,
        metric_type="binary",
    )

    model_low, model_high = result["model_confidence_interval"]
    assert model_low < result["model_effect"] < model_high
    assert result["model_effect_scale"] == "log_odds"
    # Exponentiating the model CI yields the odds-ratio interval (~e^0.5 here),
    # which must not collide with the risk-difference scale of the primary CI.
    assert model_low > 0.0
    assert result["confidence_interval"][0] < result["effect_size"]


def test_count_glm_primary_ci_brackets_raw_rate_difference() -> None:
    """For a count metric the primary CI must bracket the raw rate difference
    and agree with the model's significance call (TODO.md #36)."""
    from src.statistics.model_families import estimate_treatment_effect

    rng = np.random.default_rng(7)
    treatment = rng.poisson(3.2, 400).astype(float)
    control = rng.poisson(2.4, 400).astype(float)

    result = estimate_treatment_effect(
        treatment_data=treatment,
        control_data=control,
        significance_level=0.05,
        min_recommended_sample_size=30,
        variance_epsilon=VARIANCE_EPSILON,
        metric_type="count",
    )

    assert result["model_type"] in {"glm_poisson", "glm_negative_binomial"}
    ci_low, ci_high = result["confidence_interval"]
    assert ci_low < result["effect_size"] < ci_high
    # Clear effect: p < alpha and the effect-scale CI excludes zero, and the
    # bounds are raw-scale (the log-rate model CI is far narrower in absolute
    # terms and would not bracket the raw difference).
    assert result["p_value"] < 0.05
    assert ci_low > 0.0
    assert ci_high > result["model_confidence_interval"][1]


def test_log_transform_primary_ci_brackets_raw_effect() -> None:
    """The log-transform heavy-tail strategy must also report the primary CI
    on the raw scale (smearing-retransformed), containing the raw effect."""
    from src.statistics.model_families import estimate_treatment_effect

    rng = np.random.default_rng(11)
    treatment = rng.lognormal(2.15, 1.0, 300)
    control = rng.lognormal(1.9, 1.0, 300)

    result = estimate_treatment_effect(
        treatment_data=treatment,
        control_data=control,
        significance_level=0.05,
        min_recommended_sample_size=30,
        variance_epsilon=VARIANCE_EPSILON,
        metric_type="heavy_tail",
        heavy_tail_strategy="log_transform",
    )

    assert result["model_type"] == "ols_log1p_hc3"
    ci_low, ci_high = result["confidence_interval"]
    assert np.isfinite(ci_low) and np.isfinite(ci_high)
    assert ci_low < result["effect_size"] < ci_high
    assert "marginal_effect_ci_failed_using_model_scale" not in result["diagnostics"]["reasons"]


def test_continuous_metric_primary_ci_unchanged() -> None:
    """Continuous OLS-HC3 already reports on the mean-difference scale; the
    marginal-effect recomputation must not alter it."""
    from src.statistics.model_families import estimate_treatment_effect

    rng = np.random.default_rng(3)
    treatment = rng.normal(10.5, 2.0, 200)
    control = rng.normal(10.0, 2.0, 200)

    result = estimate_treatment_effect(
        treatment_data=treatment,
        control_data=control,
        significance_level=0.05,
        min_recommended_sample_size=30,
        variance_epsilon=VARIANCE_EPSILON,
        metric_type="continuous",
    )

    assert result["model_type"] == "ols_hc3"
    assert result["confidence_interval"] == result["model_confidence_interval"]
    assert "marginal_effect_ci_failed_using_model_scale" not in result["diagnostics"]["reasons"]

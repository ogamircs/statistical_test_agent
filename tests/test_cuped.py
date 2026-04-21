"""Tests for CUPED variance reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics.analyzer import ABTestAnalyzer
from src.statistics.covariate_resolver import apply_cuped


def test_apply_cuped_no_pre_data_returns_unchanged() -> None:
    res = apply_cuped(
        treatment_post=np.array([1.0, 2.0, 3.0]),
        control_post=np.array([1.0, 2.0, 3.0]),
        treatment_pre=None,
        control_pre=None,
    )
    assert res.applied is False
    assert res.reason == "no_pre_period_metric"


def test_apply_cuped_zero_variance_pre_returns_unchanged() -> None:
    res = apply_cuped(
        treatment_post=np.array([1.0, 2.0, 3.0]),
        control_post=np.array([1.0, 2.0, 3.0]),
        treatment_pre=np.array([5.0, 5.0, 5.0]),
        control_pre=np.array([5.0, 5.0, 5.0]),
    )
    assert res.applied is False
    assert res.reason == "zero_variance"


def test_apply_cuped_reduces_variance_with_correlated_pre() -> None:
    rng = np.random.default_rng(0)
    n = 500
    treatment_pre = rng.normal(50, 10, n)
    control_pre = rng.normal(50, 10, n)
    # Strongly correlated post values plus a small treatment effect
    treatment_post = treatment_pre * 0.9 + rng.normal(2.0, 1.0, n)
    control_post = control_pre * 0.9 + rng.normal(0.0, 1.0, n)

    res = apply_cuped(
        treatment_post=treatment_post,
        control_post=control_post,
        treatment_pre=treatment_pre,
        control_pre=control_pre,
    )

    assert res.applied is True
    assert res.variance_reduction > 0.5  # ≥50% variance removed for rho≈0.99
    assert res.theta > 0
    # Treatment-vs-control mean difference is preserved (~2.0)
    assert abs((res.treatment_adjusted.mean() - res.control_adjusted.mean()) - 2.0) < 0.5


def _df_with_pre_post(n: int = 400, *, lift: float = 0.4) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    treatment_pre = rng.normal(50, 10, n)
    control_pre = rng.normal(50, 10, n)
    treatment_post = treatment_pre * 0.95 + rng.normal(lift, 1.0, n)
    control_post = control_pre * 0.95 + rng.normal(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "experiment_group": ["treatment"] * n + ["control"] * n,
            "pre_effect": np.concatenate([treatment_pre, control_pre]),
            "post_effect": np.concatenate([treatment_post, control_post]),
        }
    )


def test_analyzer_opt_in_cuped_populates_result_fields() -> None:
    analyzer = ABTestAnalyzer()
    analyzer.set_dataframe(_df_with_pre_post())
    analyzer.set_column_mapping({
        "group": "experiment_group",
        "effect_value": "post_effect",
        "post_effect": "post_effect",
        "pre_effect": "pre_effect",
        "cuped": True,
    })
    analyzer.set_group_labels("treatment", "control")

    result = analyzer.run_ab_test()
    assert result.cuped_applied is True
    assert result.cuped_variance_reduction > 0.5
    assert result.cuped_theta > 0


def test_analyzer_default_cuped_off_keeps_legacy_behavior() -> None:
    analyzer = ABTestAnalyzer()
    analyzer.set_dataframe(_df_with_pre_post())
    analyzer.set_column_mapping({
        "group": "experiment_group",
        "effect_value": "post_effect",
        "post_effect": "post_effect",
        "pre_effect": "pre_effect",
    })
    analyzer.set_group_labels("treatment", "control")

    result = analyzer.run_ab_test()
    assert result.cuped_applied is False
    assert result.cuped_variance_reduction == 0.0

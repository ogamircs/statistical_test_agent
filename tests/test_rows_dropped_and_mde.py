"""Tests for rows_dropped accounting + achieved_mde."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics.analyzer import ABTestAnalyzer
from src.statistics.models import ABTestResult
from src.statistics.power_analysis import calculate_minimum_detectable_effect


def test_minimum_detectable_effect_positive_for_typical_n() -> None:
    mde = calculate_minimum_detectable_effect(
        n_treatment=400,
        n_control=400,
        significance_level=0.05,
        power_threshold=0.8,
    )
    assert mde > 0
    assert mde < 1.0  # standardized effect; typical large samples land well below 1


def test_minimum_detectable_effect_zero_for_tiny_n() -> None:
    assert calculate_minimum_detectable_effect(
        n_treatment=1,
        n_control=1,
        significance_level=0.05,
        power_threshold=0.8,
    ) == 0.0


def test_ab_test_result_has_new_fields() -> None:
    result = ABTestResult(segment="x", treatment_size=10, control_size=10)
    assert result.rows_dropped == 0
    assert result.achieved_mde == 0.0


def _make_dataset_with_nans() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    treatment_values = rng.normal(loc=10.5, scale=2.0, size=200)
    control_values = rng.normal(loc=10.0, scale=2.0, size=200)
    treatment_values[:5] = np.nan  # 5 dropped from treatment
    control_values[:3] = np.nan  # 3 dropped from control
    return pd.DataFrame(
        {
            "experiment_group": ["treatment"] * 200 + ["control"] * 200,
            "post_effect": np.concatenate([treatment_values, control_values]),
        }
    )


def test_analyzer_reports_rows_dropped_and_achieved_mde() -> None:
    df = _make_dataset_with_nans()
    analyzer = ABTestAnalyzer()
    analyzer.set_dataframe(df)
    analyzer.set_column_mapping({
        "group": "experiment_group",
        "effect_value": "post_effect",
        "post_effect": "post_effect",
    })
    analyzer.set_group_labels("treatment", "control")

    result = analyzer.run_ab_test()

    assert result.rows_dropped == 8  # 5 + 3 NaNs across both arms
    assert result.achieved_mde > 0

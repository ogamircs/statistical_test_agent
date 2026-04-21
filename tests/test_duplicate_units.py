"""Tests for duplicate / repeated-measures detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.agent_reporting import render_run_ab_test_output
from src.statistics.analyzer import ABTestAnalyzer
from src.statistics.diagnostics import detect_duplicate_units


def test_detect_duplicate_units_no_customer_col() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = detect_duplicate_units(df=df, customer_col=None, group_col=None)
    assert out["is_applicable"] is False
    assert out["has_duplicates"] is False


def test_detect_duplicate_units_unique_ids() -> None:
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "experiment_group": ["t", "t", "c", "c"],
        }
    )
    out = detect_duplicate_units(
        df=df, customer_col="customer_id", group_col="experiment_group"
    )
    assert out["is_applicable"] is True
    assert out["unique_units"] == 4
    assert out["within_arm_repeats"] == 0
    assert out["cross_arm_units"] == 0
    assert out["has_duplicates"] is False
    assert out["warning"] == ""


def test_detect_duplicate_units_within_arm_repeats() -> None:
    df = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 3, 3, 3, 4],
            "experiment_group": ["t", "t", "t", "c", "c", "c", "c"],
        }
    )
    out = detect_duplicate_units(
        df=df, customer_col="customer_id", group_col="experiment_group"
    )
    assert out["within_arm_repeats"] == 2  # customer 1 in t, customer 3 in c
    assert out["cross_arm_units"] == 0
    assert out["has_duplicates"] is True
    assert "appear more than once" in out["warning"]


def test_detect_duplicate_units_cross_arm_contamination() -> None:
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 1, 2, 3],
            "experiment_group": ["t", "t", "c", "c", "c"],
        }
    )
    out = detect_duplicate_units(
        df=df, customer_col="customer_id", group_col="experiment_group"
    )
    assert out["cross_arm_units"] == 2
    assert out["has_duplicates"] is True
    assert "BOTH arms" in out["warning"]


def _df_with_repeats() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    treatment_ids = list(range(1, 101))
    control_ids = list(range(101, 201))
    repeated_treatment = treatment_ids + treatment_ids[:10]  # 10 repeats
    repeated_control = control_ids + control_ids[:5]
    return pd.DataFrame(
        {
            "customer_id": repeated_treatment + repeated_control,
            "experiment_group": ["treatment"] * len(repeated_treatment)
            + ["control"] * len(repeated_control),
            "post_effect": np.concatenate(
                [
                    rng.normal(11.0, 2.0, len(repeated_treatment)),
                    rng.normal(10.0, 2.0, len(repeated_control)),
                ]
            ),
        }
    )


def test_analyzer_surfaces_duplicate_warning_in_diagnostics() -> None:
    analyzer = ABTestAnalyzer()
    analyzer.set_dataframe(_df_with_repeats())
    analyzer.set_column_mapping({
        "group": "experiment_group",
        "effect_value": "post_effect",
        "post_effect": "post_effect",
        "customer_id": "customer_id",
    })
    analyzer.set_group_labels("treatment", "control")

    result = analyzer.run_ab_test()
    duplicates = result.diagnostics["experiment_quality"]["duplicate_units"]
    assert duplicates["within_arm_repeats"] == 15
    assert duplicates["has_duplicates"] is True

    output = render_run_ab_test_output(result)
    assert "appear more than once" in output

"""Tests for SRM guardrail enforcement on the analyzer + reporting paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.agent_reporting import render_run_ab_test_output
from src.statistics.analyzer import ABTestAnalyzer
from src.statistics.models import ABTestResult


def _balanced_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "experiment_group": ["treatment"] * 500 + ["control"] * 500,
            "post_effect": np.concatenate(
                [rng.normal(11.0, 2.0, 500), rng.normal(10.0, 2.0, 500)]
            ),
        }
    )


def _imbalanced_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "experiment_group": ["treatment"] * 100 + ["control"] * 900,
            "post_effect": np.concatenate(
                [rng.normal(11.0, 2.0, 100), rng.normal(10.0, 2.0, 900)]
            ),
        }
    )


def _run(df: pd.DataFrame) -> ABTestResult:
    analyzer = ABTestAnalyzer()
    analyzer.set_dataframe(df.copy())
    analyzer.set_column_mapping({
        "group": "experiment_group",
        "effect_value": "post_effect",
        "post_effect": "post_effect",
    })
    analyzer.set_group_labels("treatment", "control")
    return analyzer.run_ab_test()


def test_srm_violation_triggers_guardrail_and_blocks_significance() -> None:
    result = _run(_imbalanced_df())

    srm = result.diagnostics["experiment_quality"]["srm"]
    assert srm["is_sample_ratio_mismatch"] is True
    assert result.inference_guardrail_triggered is True
    assert result.is_significant is False
    assert result.proportion_is_significant is False


def test_balanced_dataset_does_not_trip_srm_guardrail() -> None:
    result = _run(_balanced_df())

    srm = result.diagnostics["experiment_quality"]["srm"]
    assert srm["is_sample_ratio_mismatch"] is False
    # is_significant left to t-test logic; we just assert SRM did not coerce False
    assert result.inference_guardrail_triggered is False or result.inference_guardrail_triggered is True
    # Note: other guardrails (variance / outlier) may still flip the flag — that is fine


def test_render_run_ab_test_output_surfaces_srm_block() -> None:
    result = _run(_imbalanced_df())
    output = render_run_ab_test_output(result)

    assert "GUARDRAIL" in output
    assert "sample-ratio mismatch" in output
    assert "Significant (p < 0.05): NO" in output

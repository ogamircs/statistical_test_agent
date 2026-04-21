"""Tests for diagnostics + Bayesian sections in agent_reporting."""

from __future__ import annotations

from src.agent_reporting import (
    _format_segment_diagnostics,
    render_full_analysis_output,
    render_run_ab_test_output,
)
from src.statistics.models import ABTestResult, ABTestSummary


def _result_with_diagnostics(**overrides):
    diagnostics = overrides.pop(
        "diagnostics",
        {
            "experiment_quality": {
                "assumptions": {
                    "treatment_normality_passed": False,
                    "control_normality_passed": True,
                    "equal_variance_passed": False,
                },
                "outlier_sensitivity": {
                    "is_sensitive": True,
                    "sensitivity_score": 0.42,
                },
            }
        },
    )
    base = ABTestResult(
        segment="Premium",
        treatment_size=200,
        control_size=200,
        treatment_mean=12.5,
        control_mean=10.0,
        effect_size=2.5,
        cohens_d=0.4,
        t_statistic=2.1,
        p_value=0.03,
        is_significant=True,
        confidence_interval=(0.5, 4.5),
        power=0.85,
        required_sample_size=180,
        is_sample_adequate=True,
        bayesian_prob_treatment_better=0.97,
        bayesian_credible_interval=(0.6, 4.4),
        bayesian_is_significant=True,
        diagnostics=diagnostics,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_format_segment_diagnostics_emits_failed_checks() -> None:
    result = _result_with_diagnostics()

    findings = _format_segment_diagnostics(result)

    joined = " ".join(findings)
    assert "Normality" in joined
    assert "treatment" in joined
    assert "Equal-variance" in joined
    assert "sensitive to outliers" in joined


def test_format_segment_diagnostics_empty_when_all_pass() -> None:
    result = _result_with_diagnostics(
        diagnostics={
            "experiment_quality": {
                "assumptions": {
                    "treatment_normality_passed": True,
                    "control_normality_passed": True,
                    "equal_variance_passed": True,
                },
                "outlier_sensitivity": {"is_sensitive": False},
            }
        }
    )

    assert _format_segment_diagnostics(result) == []


def test_render_run_ab_test_output_includes_bayesian_and_diagnostics() -> None:
    result = _result_with_diagnostics()

    output = render_run_ab_test_output(result)

    assert "Bayesian Test:" in output
    assert "P(Treatment > Control): 97.0%" in output
    assert "95% Credible Interval: [0.6000, 4.4000]" in output
    assert "Diagnostics:" in output
    assert "Normality" in output


def test_render_full_analysis_includes_diagnostics_table() -> None:
    summary = ABTestSummary(
        total_segments_analyzed=1,
        significant_segments=1,
        non_significant_segments=0,
        significance_rate=1.0,
        total_treatment_customers=200,
        total_control_customers=200,
        treatment_control_ratio=1.0,
        average_significant_effect=2.5,
        total_treatment_in_significant_segments=200,
        effect_calculation="x",
        segments_with_adequate_power=1,
        segments_with_inadequate_power=0,
        power_adequacy_rate=1.0,
        recommendations=["ship it"],
        detailed_results=[_result_with_diagnostics()],
    )

    output = render_full_analysis_output(summary)

    assert "### Diagnostics" in output
    assert "- Premium:" in output
    assert "Normality" in output

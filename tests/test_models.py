"""Regression tests for canonical result normalization."""

from __future__ import annotations

from dataclasses import dataclass

from src.statistics.models import (
    ABTestResult,
    ABTestSummary,
    SegmentAnalysisFailure,
    canonical_result_as_dict,
)


@dataclass
class _DerivedABTestResult(ABTestResult):
    pooled_std: float = 0.0


def test_canonical_result_prefers_dataclass_fields_over_legacy_mapping_view() -> None:
    result = _DerivedABTestResult(
        segment="Overall",
        treatment_size=10,
        control_size=12,
        treatment_mean=21.5,
        control_mean=19.25,
        effect_size=2.25,
        pooled_std=1.8,
    )

    payload = canonical_result_as_dict(result)

    assert payload["treatment_mean"] == 21.5
    assert payload["control_mean"] == 19.25
    assert payload["effect_size"] == 2.25


def test_ab_test_result_preserves_legacy_mapping_access() -> None:
    result = ABTestResult(
        segment="Overall",
        treatment_size=10,
        control_size=12,
        p_value=0.04,
        effect_size=2.25,
        confidence_interval=(1.0, 3.5),
        is_sample_adequate=True,
    )

    assert result["effect"] == 2.25
    assert result.get("p_value") == 0.04
    assert result["ci_lower"] == 1.0
    assert result["adequate_sample"] is True


def test_ab_test_summary_preserves_legacy_mapping_access() -> None:
    summary = ABTestSummary(
        total_segments_analyzed=3,
        segment_failures=[SegmentAnalysisFailure(segment="New", error="missing column")],
        detailed_results=[ABTestResult(segment="Overall", treatment_size=10, control_size=12)],
    )

    assert summary["total_segments_analyzed"] == 3
    assert "segment_failures" in summary
    assert summary.get("detailed_results")[0]["segment"] == "Overall"
    assert summary["segment_failures"][0]["error"] == "missing column"

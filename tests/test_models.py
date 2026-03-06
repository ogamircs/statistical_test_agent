"""Regression tests for canonical result normalization."""

from __future__ import annotations

from dataclasses import dataclass

from src.statistics.models import ABTestResult, canonical_result_as_dict


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

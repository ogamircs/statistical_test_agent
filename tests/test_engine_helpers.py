"""Direct unit tests for engine_helpers."""

from __future__ import annotations

import numpy as np

from src.statistics.engine_helpers import (
    build_diagnostics,
    sanitize_numeric,
    sanitize_p_value,
    zero_if_tiny,
)


def test_zero_if_tiny_collapses_subepsilon() -> None:
    assert zero_if_tiny(1e-15) == 0.0
    assert zero_if_tiny(-1e-15) == 0.0
    assert zero_if_tiny(1e-3) == 1e-3


def test_sanitize_numeric_drops_nan_and_inf() -> None:
    arr = np.array([1.0, np.nan, 2.0, np.inf, 3.0, -np.inf])
    finite, removed = sanitize_numeric(arr)
    assert finite.tolist() == [1.0, 2.0, 3.0]
    assert removed == 3


def test_sanitize_numeric_handles_empty_array() -> None:
    finite, removed = sanitize_numeric(np.array([]))
    assert finite.size == 0
    assert removed == 0


def test_sanitize_p_value_clamps_invalid_inputs() -> None:
    assert sanitize_p_value(0.05) == 0.05
    assert sanitize_p_value(np.nan) == 1.0
    assert sanitize_p_value(np.inf) == 1.0
    assert sanitize_p_value(-0.1) == 1.0
    assert sanitize_p_value(1.5) == 1.0
    assert sanitize_p_value("not-a-number") == 1.0
    assert sanitize_p_value(None) == 1.0


def test_sanitize_p_value_custom_fallback() -> None:
    assert sanitize_p_value(np.nan, fallback=0.5) == 0.5


def test_build_diagnostics_no_reasons_means_no_guardrail() -> None:
    out = build_diagnostics([], blocks_significance=False)
    assert out["guardrail_triggered"] is False
    assert out["blocks_significance"] is False
    assert out["reasons"] == []


def test_build_diagnostics_passes_extra_kwargs() -> None:
    out = build_diagnostics(["zero_variance"], blocks_significance=True, extra_field=42)
    assert out["guardrail_triggered"] is True
    assert out["blocks_significance"] is True
    assert out["extra_field"] == 42

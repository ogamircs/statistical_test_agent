"""Shared helper utilities for the statsmodels-backed analysis engine."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def zero_if_tiny(value: float, tol: float = 1e-12) -> float:
    """Normalize numerical noise around zero for stable downstream assertions."""
    return 0.0 if abs(value) < tol else value


def sanitize_numeric(values: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return finite numeric values and count of removed invalid entries."""
    array = np.asarray(values, dtype=float).reshape(-1)
    finite_mask = np.isfinite(array)
    removed = int(array.size - np.count_nonzero(finite_mask))
    return array[finite_mask], removed


def sanitize_p_value(value: Any, fallback: float = 1.0) -> float:
    """Clamp non-finite p-values to a valid fallback."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(numeric):
        return fallback
    if numeric < 0.0 or numeric > 1.0:
        return fallback
    return numeric


def build_diagnostics(
    reasons: List[str],
    *,
    blocks_significance: bool,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build a consistent diagnostics payload for statistical guardrails."""
    diagnostics: Dict[str, Any] = {
        "guardrail_triggered": bool(reasons),
        "blocks_significance": blocks_significance,
        "reasons": reasons,
    }
    diagnostics.update(kwargs)
    return diagnostics

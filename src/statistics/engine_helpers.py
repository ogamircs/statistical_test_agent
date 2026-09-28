"""Shared helper utilities for the statsmodels-backed analysis engine."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from statsmodels.stats.multitest import multipletests


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


def combine_total_effect_per_customer(
    *,
    effect_size: float,
    is_significant: bool,
    proportion_effect_per_customer: float,
) -> float:
    """Headline per-customer impact without double counting (TODO.md #42).

    The mean difference already includes the value contributed by
    incremental converters, so adding the proportion-based estimate on top
    counts that value twice. Use the significant mean difference when there
    is one; fall back to the proportion-based estimate only when the mean
    test is not significant but the conversion-rate test is. Shared by the
    pandas and Spark backends.
    """
    if is_significant:
        return float(effect_size)
    return float(proportion_effect_per_customer)


def _correction_safe_p_value(p_value: Any) -> float:
    """Clamp invalid p-values to 1.0 so correction is robust to upstream edge cases."""
    try:
        numeric = float(p_value)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        return 1.0
    return numeric


def _blocks(result: Any, *path: str) -> bool:
    node: Any = result.diagnostics
    for key in path[:-1]:
        node = node.get(key, {}) if isinstance(node, dict) else {}
    return bool(node.get(path[-1], False)) if isinstance(node, dict) else False


def apply_fdr_correction(results: Sequence[Any], *, significance_level: float) -> None:
    """Benjamini-Hochberg correction across segment results, in place.

    Shared by the pandas and Spark backends. Guardrail blocks (t-test,
    proportion test) and sample-ratio mismatch keep a segment
    non-significant after adjustment, matching the unadjusted call.
    """
    p_values = np.array([_correction_safe_p_value(r.p_value) for r in results], dtype=float)
    prop_p_values = np.array(
        [_correction_safe_p_value(r.proportion_p_value) for r in results], dtype=float
    )
    try:
        reject_main, adjusted_main, _, _ = multipletests(
            p_values, alpha=significance_level, method="fdr_bh"
        )
        reject_prop, adjusted_prop, _, _ = multipletests(
            prop_p_values, alpha=significance_level, method="fdr_bh"
        )
    except Exception:
        for result in results:
            result.p_value_adjusted = _correction_safe_p_value(result.p_value)
            result.is_significant_adjusted = result.is_significant
            result.proportion_p_value_adjusted = _correction_safe_p_value(result.proportion_p_value)
            result.proportion_is_significant_adjusted = result.proportion_is_significant
            result.multiple_testing_method = "none"
            result.multiple_testing_applied = False
        return

    for idx, result in enumerate(results):
        result.p_value_adjusted = float(adjusted_main[idx])
        result.proportion_p_value_adjusted = float(adjusted_prop[idx])
        result.multiple_testing_method = "fdr_bh"
        result.multiple_testing_applied = True

        t_test_blocks = _blocks(result, "frequentist", "t_test", "blocks_significance")
        prop_blocks = _blocks(result, "frequentist", "proportion_test", "blocks_significance")
        srm_blocks = _blocks(result, "experiment_quality", "srm", "is_sample_ratio_mismatch")
        result.is_significant_adjusted = (
            bool(reject_main[idx]) and not t_test_blocks and not srm_blocks
        )
        result.proportion_is_significant_adjusted = (
            bool(reject_prop[idx]) and not prop_blocks and not srm_blocks
        )

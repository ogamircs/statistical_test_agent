"""Experiment-quality diagnostics for A/B test result validation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.stats import chisquare, kurtosis, levene, normaltest, shapiro, skew, trim_mean
from scipy.stats.mstats import winsorize

from .engine_helpers import sanitize_numeric, sanitize_p_value


def run_srm_diagnostics(
    *,
    treatment_size: int,
    control_size: int,
    significance_level: float,
    expected_treatment_ratio: float,
) -> Dict[str, Any]:
    """Sample Ratio Mismatch (SRM) diagnostics via chi-square goodness-of-fit."""
    treatment_size = int(max(treatment_size, 0))
    control_size = int(max(control_size, 0))
    total = treatment_size + control_size
    expected_ratio = float(expected_treatment_ratio)

    diagnostics: Dict[str, Any] = {
        "expected_treatment_ratio": expected_ratio,
        "expected_control_ratio": float(1.0 - expected_ratio),
        "observed_treatment_ratio": float(treatment_size / total) if total > 0 else None,
        "observed_control_ratio": float(control_size / total) if total > 0 else None,
        "expected_treatment_size": float(total * expected_ratio) if total > 0 else 0.0,
        "expected_control_size": float(total * (1.0 - expected_ratio)) if total > 0 else 0.0,
        "chi2_statistic": 0.0,
        "p_value": 1.0,
        "is_sample_ratio_mismatch": False,
        "is_applicable": True,
        "reason": None,
    }

    if total <= 0:
        diagnostics["is_applicable"] = False
        diagnostics["reason"] = "no_observations"
        return diagnostics

    if expected_ratio <= 0.0 or expected_ratio >= 1.0:
        diagnostics["is_applicable"] = False
        diagnostics["reason"] = "invalid_expected_ratio"
        return diagnostics

    expected_counts = np.array(
        [diagnostics["expected_treatment_size"], diagnostics["expected_control_size"]],
        dtype=float,
    )
    observed_counts = np.array([treatment_size, control_size], dtype=float)

    if np.any(expected_counts <= 0):
        diagnostics["is_applicable"] = False
        diagnostics["reason"] = "non_positive_expected_counts"
        return diagnostics

    try:
        chi2_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)
        diagnostics["chi2_statistic"] = float(max(chi2_stat, 0.0))
        diagnostics["p_value"] = sanitize_p_value(p_value)
        diagnostics["is_sample_ratio_mismatch"] = diagnostics["p_value"] < significance_level
        return diagnostics
    except Exception:
        diagnostics["is_applicable"] = False
        diagnostics["reason"] = "srm_test_failed"
        return diagnostics


def run_assumption_diagnostics(
    *,
    treatment_data: np.ndarray,
    control_data: np.ndarray,
    significance_level: float,
    variance_epsilon: float,
) -> Dict[str, Any]:
    """Check normality and equal-variance assumptions for mean-comparison inference."""
    treatment_data, treatment_invalid = sanitize_numeric(treatment_data)
    control_data, control_invalid = sanitize_numeric(control_data)

    n_treatment = len(treatment_data)
    n_control = len(control_data)
    min_n = min(n_treatment, n_control)
    max_n = max(n_treatment, n_control)

    treatment_var = float(np.var(treatment_data, ddof=1)) if n_treatment >= 2 else 0.0
    control_var = float(np.var(control_data, ddof=1)) if n_control >= 2 else 0.0

    if min_n >= 8 and max_n > 5000:
        normality_test = "dagostino_k2"
    elif min_n >= 3:
        normality_test = "shapiro"
    else:
        normality_test = "not_applicable"

    def _normality_p_value(values: np.ndarray, variance: float) -> float | None:
        if normality_test == "not_applicable" or len(values) < 3 or variance <= variance_epsilon:
            return None
        try:
            if normality_test == "dagostino_k2":
                _stat, p_val = normaltest(values)
            else:
                _stat, p_val = shapiro(values)
            return sanitize_p_value(p_val)
        except Exception:
            return None

    treatment_normality_p = _normality_p_value(treatment_data, treatment_var)
    control_normality_p = _normality_p_value(control_data, control_var)

    treatment_normality_passed = (
        treatment_normality_p > significance_level if treatment_normality_p is not None else None
    )
    control_normality_passed = (
        control_normality_p > significance_level if control_normality_p is not None else None
    )

    variance_test = "not_applicable"
    equal_variance_p_value: float | None = None
    equal_variance_passed: bool | None = None
    variance_reason: str | None = None
    if n_treatment < 3 or n_control < 3:
        variance_reason = "insufficient_sample_size"
    elif treatment_var <= variance_epsilon or control_var <= variance_epsilon:
        variance_reason = "degenerate_variance"
    else:
        variance_test = "levene"
        try:
            _stat, p_val = levene(treatment_data, control_data, center="median")
            equal_variance_p_value = sanitize_p_value(p_val)
            equal_variance_passed = equal_variance_p_value > significance_level
        except Exception:
            variance_reason = "variance_test_failed"

    return {
        "normality_test": normality_test,
        "treatment_normality_p_value": treatment_normality_p,
        "control_normality_p_value": control_normality_p,
        "treatment_normality_passed": treatment_normality_passed,
        "control_normality_passed": control_normality_passed,
        "variance_test": variance_test,
        "equal_variance_p_value": equal_variance_p_value,
        "equal_variance_passed": equal_variance_passed,
        "non_finite_values_removed": treatment_invalid + control_invalid,
        "variance_reason": variance_reason,
    }


def run_outlier_sensitivity(
    *,
    treatment_data: np.ndarray,
    control_data: np.ndarray,
    baseline_effect: float | None,
    winsor_limit: float,
    trim_fraction: float,
    variance_epsilon: float,
) -> Dict[str, Any]:
    """Estimate how sensitive the mean effect is to robust outlier handling."""
    treatment_data, treatment_invalid = sanitize_numeric(treatment_data)
    control_data, control_invalid = sanitize_numeric(control_data)

    winsor_limit = float(min(max(winsor_limit, 0.0), 0.49))
    trim_fraction = float(min(max(trim_fraction, 0.0), 0.49))

    n_treatment = len(treatment_data)
    n_control = len(control_data)
    if n_treatment == 0 or n_control == 0:
        return {
            "winsor_limit": winsor_limit,
            "trim_fraction": trim_fraction,
            "raw_effect": float(baseline_effect or 0.0),
            "winsorized_effect": 0.0,
            "trimmed_effect": 0.0,
            "winsorized_delta": 0.0,
            "trimmed_delta": 0.0,
            "max_abs_delta": 0.0,
            "sensitivity_score": 0.0,
            "is_sensitive": False,
            "is_applicable": False,
            "reason": "insufficient_sample_size",
            "non_finite_values_removed": treatment_invalid + control_invalid,
        }

    raw_effect = (
        float(np.mean(treatment_data) - np.mean(control_data))
        if baseline_effect is None
        else float(baseline_effect)
    )

    try:
        winsorized_treatment = np.asarray(
            winsorize(treatment_data, limits=(winsor_limit, winsor_limit)),
            dtype=float,
        )
        winsorized_control = np.asarray(
            winsorize(control_data, limits=(winsor_limit, winsor_limit)),
            dtype=float,
        )
        winsorized_effect = float(np.mean(winsorized_treatment) - np.mean(winsorized_control))
    except Exception:
        winsorized_effect = raw_effect

    try:
        trimmed_effect = float(
            trim_mean(treatment_data, proportiontocut=trim_fraction)
            - trim_mean(control_data, proportiontocut=trim_fraction)
        )
    except Exception:
        trimmed_effect = raw_effect

    winsorized_delta = winsorized_effect - raw_effect
    trimmed_delta = trimmed_effect - raw_effect
    max_abs_delta = float(max(abs(winsorized_delta), abs(trimmed_delta)))

    pooled_scale = float(np.std(np.concatenate([treatment_data, control_data]), ddof=1))
    if pooled_scale <= variance_epsilon:
        sensitivity_score = float(np.inf) if max_abs_delta > 0 else 0.0
    else:
        sensitivity_score = float(max_abs_delta / pooled_scale)

    return {
        "winsor_limit": winsor_limit,
        "trim_fraction": trim_fraction,
        "raw_effect": raw_effect,
        "winsorized_effect": winsorized_effect,
        "trimmed_effect": trimmed_effect,
        "winsorized_delta": winsorized_delta,
        "trimmed_delta": trimmed_delta,
        "max_abs_delta": max_abs_delta,
        "sensitivity_score": sensitivity_score,
        "is_sensitive": bool(sensitivity_score > 0.25),
        "is_applicable": True,
        "reason": None,
        "non_finite_values_removed": treatment_invalid + control_invalid,
    }

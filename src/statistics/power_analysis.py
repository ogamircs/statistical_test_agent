"""Power and sample-size helpers for frequentist A/B test analysis."""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.stats.power import TTestIndPower
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def calculate_cohens_d(treatment_data: np.ndarray, control_data: np.ndarray) -> float:
    """Calculate Cohen's d effect size using pooled variance."""
    n_treatment, n_control = len(treatment_data), len(control_data)
    if n_treatment < 2 or n_control < 2:
        return 0.0

    var_treatment = np.var(treatment_data, ddof=1)
    var_control = np.var(control_data, ddof=1)

    pooled_var = (
        ((n_treatment - 1) * var_treatment + (n_control - 1) * var_control)
        / (n_treatment + n_control - 2)
    )
    pooled_std = np.sqrt(max(pooled_var, 0.0))
    if pooled_std <= 0:
        return 0.0

    return float((np.mean(treatment_data) - np.mean(control_data)) / pooled_std)


def calculate_power(
    *,
    effect_size: float,
    n_treatment: int,
    n_control: int,
    significance_level: float,
) -> float:
    """Calculate achieved power for a two-sample test."""
    if effect_size == 0 or n_treatment <= 1 or n_control <= 1:
        return 0.0

    power_analysis = TTestIndPower()
    ratio = n_control / n_treatment if n_treatment > 0 else 1

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            power = power_analysis.solve_power(
                effect_size=abs(effect_size),
                nobs1=n_treatment,
                ratio=ratio,
                alpha=significance_level,
            )
        return float(min(power, 1.0))
    except Exception:
        return 0.0


def calculate_required_sample_size(
    *,
    effect_size: float,
    ratio: float,
    power_threshold: float,
    significance_level: float,
) -> int:
    """Calculate required sample size per group for a target power threshold."""
    if effect_size == 0:
        return int(1e9)

    power_analysis = TTestIndPower()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            n_required = power_analysis.solve_power(
                effect_size=abs(effect_size),
                power=power_threshold,
                ratio=ratio,
                alpha=significance_level,
            )
        n_required_scalar = float(np.atleast_1d(n_required)[0])
        return int(np.ceil(n_required_scalar))
    except Exception:
        return int(1e9)

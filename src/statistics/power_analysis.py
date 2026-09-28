"""Power and sample-size helpers for frequentist A/B test analysis."""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.stats.power import TTestIndPower
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Returned by ``calculate_required_sample_size`` when no finite N can reach the
# target power (zero effect or solver failure). Shared by both backends so the
# "needs ~N per group" guidance is identical regardless of backend.
REQUIRED_SAMPLE_SIZE_UNREACHABLE = int(1e9)

# Standardized (Cohen's d) effect the experiment should be able to detect when
# no target is configured. Cohen's conventional "small" effect: sample adequacy
# is judged against this pre-specified target rather than the observed effect.
DEFAULT_TARGET_EFFECT_SIZE = 0.2


def _record_failure(warnings_sink: list[str] | None, message: str) -> None:
    if warnings_sink is not None:
        warnings_sink.append(message)


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
    warnings_sink: list[str] | None = None,
) -> float:
    """Calculate achieved power for a two-sample test.

    When the solver fails, 0.0 is returned and a message is appended to
    ``warnings_sink`` (if given) so the failure is not mistaken for a real
    zero-power result.
    """
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
        _record_failure(warnings_sink, "power calculation failed; reported power is a 0.0 placeholder")
        return 0.0


def calculate_minimum_detectable_effect(
    *,
    n_treatment: int,
    n_control: int,
    significance_level: float,
    power_threshold: float,
    warnings_sink: list[str] | None = None,
) -> float:
    """Solve for the smallest standardized effect size detectable at current N.

    Returns 0.0 if the underlying solver fails or sample sizes are too small;
    a solver failure is also reported through ``warnings_sink``.
    """
    if n_treatment <= 1 or n_control <= 1:
        return 0.0

    power_analysis = TTestIndPower()
    ratio = n_control / n_treatment if n_treatment > 0 else 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mde = power_analysis.solve_power(
                effect_size=None,
                nobs1=n_treatment,
                ratio=ratio,
                alpha=significance_level,
                power=power_threshold,
            )
        return float(np.atleast_1d(mde)[0])
    except Exception:
        _record_failure(
            warnings_sink,
            "minimum-detectable-effect calculation failed; achieved MDE is a 0.0 placeholder",
        )
        return 0.0


def calculate_required_sample_size(
    *,
    effect_size: float,
    ratio: float,
    power_threshold: float,
    significance_level: float,
    warnings_sink: list[str] | None = None,
) -> int:
    """Calculate required sample size per group for a target power threshold.

    Returns ``REQUIRED_SAMPLE_SIZE_UNREACHABLE`` for a zero effect or when
    the solver fails (the latter is reported through ``warnings_sink``).
    """
    if effect_size == 0:
        return REQUIRED_SAMPLE_SIZE_UNREACHABLE

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
        _record_failure(
            warnings_sink,
            "required-sample-size calculation failed; required N is a placeholder",
        )
        return REQUIRED_SAMPLE_SIZE_UNREACHABLE


def is_sample_adequate(*, achieved_mde: float, target_effect_size: float) -> bool:
    """Judge adequacy by design sensitivity, not by post-hoc observed power.

    The sample is adequate when the minimum detectable standardized effect
    at the configured alpha/power is no larger than the target effect.
    Observed power on the realized effect is a monotone transform of the
    p-value and says nothing new about adequacy (the observed-power fallacy).
    """
    if not np.isfinite(achieved_mde) or achieved_mde <= 0.0:
        return False
    return bool(achieved_mde <= abs(float(target_effect_size)))

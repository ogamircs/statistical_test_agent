"""Delta-method significance test for ratio metrics.

Ratio metrics (revenue per user, sessions per user, CTR-as-ratio-of-sums)
violate the independence assumption of the standard two-sample t-test:
the sum-of-numerators / sum-of-denominators is the right point estimate,
but its variance must be computed with the delta method on per-user
(numerator, denominator) pairs.

References:
    Deng, Knoblich, Lu (2018) — "Applying the Delta Method in Metric
    Analytics" (KDD'18). The implementation here follows their
    "ratio of two means" estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class RatioMetricResult:
    """Outcome of a delta-method ratio comparison between two arms."""

    treatment_ratio: float
    control_ratio: float
    absolute_diff: float
    relative_diff: float
    standard_error: float
    z_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    treatment_n: int
    control_n: int
    is_significant: bool
    reason: Optional[str] = None


def _arm_ratio_and_variance(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> Tuple[float, float, int]:
    """Return (ratio, variance_of_ratio, n) using the delta method.

    Variance derivation (Deng et al. 2018, eq. 8):
        Var(R) ≈ (1/D̄²) * (Var(N) + R²·Var(D) - 2·R·Cov(N,D)) / n
    where R = mean(N)/mean(D).
    """
    n = int(min(len(numerator), len(denominator)))
    if n < 2:
        return 0.0, 0.0, n

    num_mean = float(np.mean(numerator[:n]))
    den_mean = float(np.mean(denominator[:n]))
    if abs(den_mean) < 1e-12:
        # Undefined ratio — signal with NaN so the outer test can bail
        # cleanly. Distinct from "zero variance within a well-defined arm".
        return float("nan"), 0.0, n

    ratio = num_mean / den_mean
    cov_matrix = np.cov(numerator[:n], denominator[:n], ddof=1)
    var_n = float(cov_matrix[0, 0])
    var_d = float(cov_matrix[1, 1])
    cov_nd = float(cov_matrix[0, 1])

    variance = (var_n + ratio * ratio * var_d - 2.0 * ratio * cov_nd) / (den_mean * den_mean * n)
    return ratio, max(variance, 0.0), n


def delta_method_ratio_test(
    *,
    treatment_numerator: np.ndarray,
    treatment_denominator: np.ndarray,
    control_numerator: np.ndarray,
    control_denominator: np.ndarray,
    significance_level: float = 0.05,
) -> RatioMetricResult:
    """Two-sample test for the difference of ratio metrics."""
    treatment_numerator = np.asarray(treatment_numerator, dtype=float)
    treatment_denominator = np.asarray(treatment_denominator, dtype=float)
    control_numerator = np.asarray(control_numerator, dtype=float)
    control_denominator = np.asarray(control_denominator, dtype=float)

    treat_ratio, treat_var, treat_n = _arm_ratio_and_variance(
        treatment_numerator, treatment_denominator
    )
    control_ratio, control_var, control_n = _arm_ratio_and_variance(
        control_numerator, control_denominator
    )

    if treat_n < 2 or control_n < 2:
        return RatioMetricResult(
            treatment_ratio=treat_ratio,
            control_ratio=control_ratio,
            absolute_diff=0.0,
            relative_diff=0.0,
            standard_error=0.0,
            z_statistic=0.0,
            p_value=1.0,
            confidence_interval=(0.0, 0.0),
            treatment_n=treat_n,
            control_n=control_n,
            is_significant=False,
            reason="insufficient_sample_size",
        )

    if np.isnan(treat_ratio) or np.isnan(control_ratio):
        return RatioMetricResult(
            treatment_ratio=0.0 if np.isnan(treat_ratio) else treat_ratio,
            control_ratio=0.0 if np.isnan(control_ratio) else control_ratio,
            absolute_diff=0.0,
            relative_diff=0.0,
            standard_error=0.0,
            z_statistic=0.0,
            p_value=1.0,
            confidence_interval=(0.0, 0.0),
            treatment_n=treat_n,
            control_n=control_n,
            is_significant=False,
            reason="undefined_ratio",
        )

    # Only bail when the COMBINED standard error is zero (both arms fully
    # degenerate). A single near-constant arm is rare in practice but the
    # test is still valid via the variance from the other arm — returning
    # p=1.0 preemptively created false negatives for low-noise denominators.
    diff = treat_ratio - control_ratio
    se = float(np.sqrt(treat_var + control_var))
    if se == 0.0:
        return RatioMetricResult(
            treatment_ratio=treat_ratio,
            control_ratio=control_ratio,
            absolute_diff=diff,
            relative_diff=(diff / control_ratio) if control_ratio != 0 else 0.0,
            standard_error=0.0,
            z_statistic=0.0,
            p_value=1.0,
            confidence_interval=(diff, diff),
            treatment_n=treat_n,
            control_n=control_n,
            is_significant=False,
            reason="zero_variance",
        )

    z = diff / se
    p = float(2.0 * (1.0 - norm.cdf(abs(z))))
    z_alpha = float(norm.ppf(1.0 - significance_level / 2.0))
    half_width = z_alpha * se
    ci = (diff - half_width, diff + half_width)
    relative_diff = (diff / control_ratio) if control_ratio != 0 else 0.0

    return RatioMetricResult(
        treatment_ratio=treat_ratio,
        control_ratio=control_ratio,
        absolute_diff=diff,
        relative_diff=relative_diff,
        standard_error=se,
        z_statistic=z,
        p_value=p,
        confidence_interval=ci,
        treatment_n=treat_n,
        control_n=control_n,
        is_significant=p < significance_level,
    )

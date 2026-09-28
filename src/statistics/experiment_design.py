"""AA (pre-period balance) testing helpers for experiment design checks.

A failed AA check is reported, never "fixed" by resampling: choosing a control
subsample conditional on the AA p-value biases every downstream test
(TODO.md #88). Pre-period imbalance is handled by adjustment (CUPED, DiD,
covariates) and surfaced to the user as a warning.
"""

from __future__ import annotations

import numpy as np
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from .models import AATestResult


def run_aa_test(
    *,
    treatment_pre: np.ndarray,
    control_pre: np.ndarray,
    segment_name: str,
    significance_level: float,
) -> AATestResult:
    """Run a pre-period balance check between treatment and control."""
    n_treatment = len(treatment_pre)
    n_control = len(control_pre)

    if n_treatment < 2 or n_control < 2:
        return AATestResult(
            segment=segment_name,
            treatment_size=n_treatment,
            control_size=n_control,
            treatment_pre_mean=float(np.mean(treatment_pre)) if n_treatment > 0 else 0.0,
            control_pre_mean=float(np.mean(control_pre)) if n_control > 0 else 0.0,
            pre_effect_diff=0.0,
            aa_t_statistic=0.0,
            aa_p_value=1.0,
            is_balanced=True,
        )

    treatment_pre_mean = float(np.mean(treatment_pre))
    control_pre_mean = float(np.mean(control_pre))
    pre_effect_diff = treatment_pre_mean - control_pre_mean

    d_treatment = DescrStatsW(treatment_pre)
    d_control = DescrStatsW(control_pre)
    compare = CompareMeans(d_treatment, d_control)
    t_stat, p_value, _df = compare.ttest_ind(usevar="unequal")

    return AATestResult(
        segment=segment_name,
        treatment_size=n_treatment,
        control_size=n_control,
        treatment_pre_mean=treatment_pre_mean,
        control_pre_mean=control_pre_mean,
        pre_effect_diff=pre_effect_diff,
        aa_t_statistic=float(t_stat),
        aa_p_value=float(p_value),
        is_balanced=float(p_value) > significance_level,
    )

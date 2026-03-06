"""AA testing and control rebalancing helpers for experiment design checks."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
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


def bootstrap_balanced_control(
    *,
    treatment_pre: np.ndarray,
    control_df: pd.DataFrame,
    pre_col: str,
    max_iterations: int,
    target_p_value: float,
    significance_level: float,
) -> Tuple[pd.DataFrame, AATestResult]:
    """Subsample control rows to improve pre-period balance when AA fails."""
    original_control_size = len(control_df)
    control_pre = control_df[pre_col].to_numpy()

    initial_aa = run_aa_test(
        treatment_pre=treatment_pre,
        control_pre=control_pre,
        segment_name="Overall",
        significance_level=significance_level,
    )
    if initial_aa.is_balanced:
        initial_aa.bootstrapping_applied = False
        initial_aa.original_control_size = original_control_size
        initial_aa.balanced_control_size = original_control_size
        return control_df, initial_aa

    np.random.seed(42)
    best_control_df = control_df.copy()
    best_p_value = 0.0
    best_aa_result = None
    bootstrap_iterations = max_iterations

    for iteration in range(max_iterations):
        sample_size = min(len(control_df), len(treatment_pre))
        sample_indices = np.random.choice(len(control_df), size=sample_size, replace=False)
        sampled_control = control_df.iloc[sample_indices]
        sampled_pre = sampled_control[pre_col].to_numpy()

        aa_result = run_aa_test(
            treatment_pre=treatment_pre,
            control_pre=sampled_pre,
            segment_name="Overall",
            significance_level=significance_level,
        )
        if aa_result.aa_p_value > best_p_value:
            best_p_value = float(aa_result.aa_p_value)
            best_control_df = sampled_control.copy()
            best_aa_result = aa_result
            bootstrap_iterations = iteration + 1

            if best_p_value >= target_p_value:
                break

    if best_aa_result is not None:
        best_aa_result.bootstrapping_applied = True
        best_aa_result.original_control_size = original_control_size
        best_aa_result.balanced_control_size = len(best_control_df)
        best_aa_result.bootstrap_iterations = bootstrap_iterations
        best_aa_result.is_balanced = best_p_value > significance_level
        return best_control_df, best_aa_result

    initial_aa.bootstrapping_applied = True
    initial_aa.original_control_size = original_control_size
    initial_aa.balanced_control_size = original_control_size
    initial_aa.bootstrap_iterations = max_iterations
    return control_df, initial_aa

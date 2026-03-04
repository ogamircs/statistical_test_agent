"""
Statsmodels-driven statistical engine for A/B testing.

This module centralizes inferential statistics and keeps computation logic
separate from data management/orchestration.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import confint_proportions_2indep, test_proportions_2indep
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from .models import AATestResult


class StatsmodelsABTestEngine:
    """Encapsulates frequentist and Bayesian A/B test computations."""

    def __init__(
        self,
        significance_level: float = 0.05,
        power_threshold: float = 0.8,
        bayesian_samples: int = 10_000,
    ) -> None:
        self.significance_level = significance_level
        self.power_threshold = power_threshold
        self.bayesian_samples = bayesian_samples

    @staticmethod
    def _zero_if_tiny(value: float, tol: float = 1e-12) -> float:
        """Normalize numerical noise around zero for stable downstream assertions."""
        return 0.0 if abs(value) < tol else value

    def calculate_cohens_d(self, treatment_data: np.ndarray, control_data: np.ndarray) -> float:
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

    def calculate_power(self, effect_size: float, n_treatment: int, n_control: int) -> float:
        """Calculate achieved power for two-sample test."""
        if effect_size == 0 or n_treatment <= 1 or n_control <= 1:
            return 0.0

        power_analysis = TTestIndPower()
        ratio = n_control / n_treatment if n_treatment > 0 else 1

        try:
            power = power_analysis.solve_power(
                effect_size=abs(effect_size),
                nobs1=n_treatment,
                ratio=ratio,
                alpha=self.significance_level,
            )
            return float(min(power, 1.0))
        except Exception:
            return 0.0

    def calculate_required_sample_size(self, effect_size: float, ratio: float = 1.0) -> int:
        """Calculate required sample size per group for target power."""
        if effect_size == 0:
            return int(1e9)

        power_analysis = TTestIndPower()
        try:
            n_required = power_analysis.solve_power(
                effect_size=abs(effect_size),
                power=self.power_threshold,
                ratio=ratio,
                alpha=self.significance_level,
            )
            return int(np.ceil(n_required))
        except Exception:
            return int(1e9)

    def run_aa_test(
        self,
        treatment_pre: np.ndarray,
        control_pre: np.ndarray,
        segment_name: str = "Overall",
    ) -> AATestResult:
        """Run pre-period balance check between treatment and control."""
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

        is_balanced = float(p_value) > self.significance_level

        return AATestResult(
            segment=segment_name,
            treatment_size=n_treatment,
            control_size=n_control,
            treatment_pre_mean=treatment_pre_mean,
            control_pre_mean=control_pre_mean,
            pre_effect_diff=pre_effect_diff,
            aa_t_statistic=float(t_stat),
            aa_p_value=float(p_value),
            is_balanced=is_balanced,
        )

    def bootstrap_balanced_control(
        self,
        treatment_pre: np.ndarray,
        control_df: pd.DataFrame,
        pre_col: str,
        max_iterations: int = 1000,
        target_p_value: float = 0.10,
    ) -> Tuple[pd.DataFrame, AATestResult]:
        """
        Subsample control to improve pre-period balance when AA test fails.
        """
        original_control_size = len(control_df)
        control_pre = control_df[pre_col].to_numpy()

        initial_aa = self.run_aa_test(treatment_pre, control_pre)
        if initial_aa.is_balanced:
            initial_aa.bootstrapping_applied = False
            initial_aa.original_control_size = original_control_size
            initial_aa.balanced_control_size = original_control_size
            return control_df, initial_aa

        np.random.seed(42)
        best_control_df = control_df.copy()
        best_p_value = 0.0
        best_aa_result = None

        for iteration in range(max_iterations):
            sample_size = min(len(control_df), len(treatment_pre))
            sample_indices = np.random.choice(len(control_df), size=sample_size, replace=False)
            sampled_control = control_df.iloc[sample_indices]
            sampled_pre = sampled_control[pre_col].to_numpy()

            aa_result = self.run_aa_test(treatment_pre, sampled_pre)
            if aa_result.aa_p_value > best_p_value:
                best_p_value = float(aa_result.aa_p_value)
                best_control_df = sampled_control.copy()
                best_aa_result = aa_result

                if best_p_value >= target_p_value:
                    break

        if best_aa_result is not None:
            best_aa_result.bootstrapping_applied = True
            best_aa_result.original_control_size = original_control_size
            best_aa_result.balanced_control_size = len(best_control_df)
            best_aa_result.bootstrap_iterations = iteration + 1
            best_aa_result.is_balanced = best_p_value > self.significance_level
            return best_control_df, best_aa_result

        initial_aa.bootstrapping_applied = True
        initial_aa.original_control_size = original_control_size
        initial_aa.balanced_control_size = original_control_size
        initial_aa.bootstrap_iterations = max_iterations
        return control_df, initial_aa

    def estimate_treatment_effect(
        self,
        treatment_data: np.ndarray,
        control_data: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Estimate treatment effect with statsmodels OLS + HC3 robust covariance.

        Model: y = beta0 + beta1 * treatment_indicator + error
        where beta1 is mean(treatment) - mean(control).
        """
        n_treatment = len(treatment_data)
        n_control = len(control_data)

        if n_treatment < 2 or n_control < 2:
            return {
                "treatment_mean": float(np.mean(treatment_data)) if n_treatment else 0.0,
                "control_mean": float(np.mean(control_data)) if n_control else 0.0,
                "effect_size": 0.0,
                "t_statistic": 0.0,
                "p_value": 1.0,
                "confidence_interval": (0.0, 0.0),
            }

        var_treatment = float(np.var(treatment_data, ddof=1))
        var_control = float(np.var(control_data, ddof=1))
        if var_treatment == 0.0 and var_control == 0.0:
            effect = self._zero_if_tiny(float(np.mean(treatment_data) - np.mean(control_data)))
            return {
                "treatment_mean": float(np.mean(treatment_data)),
                "control_mean": float(np.mean(control_data)),
                "effect_size": effect,
                "t_statistic": 0.0,
                "p_value": 1.0,
                "confidence_interval": (effect, effect),
            }

        y = np.concatenate([treatment_data, control_data])
        treatment_indicator = np.concatenate([np.ones(n_treatment), np.zeros(n_control)])

        try:
            X = sm.add_constant(treatment_indicator, has_constant="add")
            ols_result = sm.OLS(y, X).fit(cov_type="HC3")

            ci = ols_result.conf_int(alpha=self.significance_level)
            effect_ci = ci[1]

            effect = self._zero_if_tiny(float(ols_result.params[1]))
            return {
                "treatment_mean": float(np.mean(treatment_data)),
                "control_mean": float(np.mean(control_data)),
                "effect_size": effect,
                "t_statistic": float(ols_result.tvalues[1]),
                "p_value": float(ols_result.pvalues[1]),
                "confidence_interval": (float(effect_ci[0]), float(effect_ci[1])),
            }
        except Exception:
            # CompareMeans fallback keeps behavior resilient.
            d_treatment = DescrStatsW(treatment_data)
            d_control = DescrStatsW(control_data)
            compare = CompareMeans(d_treatment, d_control)
            t_stat, p_value, _df = compare.ttest_ind(usevar="unequal")
            ci_low, ci_high = compare.tconfint_diff(alpha=self.significance_level, usevar="unequal")

            effect = self._zero_if_tiny(float(np.mean(treatment_data) - np.mean(control_data)))
            return {
                "treatment_mean": float(np.mean(treatment_data)),
                "control_mean": float(np.mean(control_data)),
                "effect_size": effect,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "confidence_interval": (float(ci_low), float(ci_high)),
            }

    def estimate_did_effect(
        self,
        treatment_pre: np.ndarray,
        treatment_post: np.ndarray,
        control_pre: np.ndarray,
        control_post: np.ndarray,
    ) -> Dict[str, float]:
        """
        Estimate DiD with statsmodels OLS on per-user deltas.

        delta = post - pre
        delta = beta0 + beta1 * treatment_indicator + error
        beta1 is DiD effect estimate.
        """
        treatment_change = treatment_post - treatment_pre
        control_change = control_post - control_pre

        if len(treatment_change) < 2 or len(control_change) < 2:
            did_effect = self._zero_if_tiny(float(np.mean(treatment_change) - np.mean(control_change)))
            return {
                "did_effect": did_effect,
                "treatment_change": float(np.mean(treatment_change)),
                "control_change": float(np.mean(control_change)),
            }

        delta = np.concatenate([treatment_change, control_change])
        treatment_indicator = np.concatenate(
            [np.ones(len(treatment_change)), np.zeros(len(control_change))]
        )

        try:
            X = sm.add_constant(treatment_indicator, has_constant="add")
            ols_result = sm.OLS(delta, X).fit(cov_type="HC3")
            did_effect = self._zero_if_tiny(float(ols_result.params[1]))
        except Exception:
            did_effect = self._zero_if_tiny(float(np.mean(treatment_change) - np.mean(control_change)))

        return {
            "did_effect": did_effect,
            "treatment_change": float(np.mean(treatment_change)),
            "control_change": float(np.mean(control_change)),
        }

    def run_proportion_test(
        self,
        treatment_data: np.ndarray,
        control_data: np.ndarray,
    ) -> Dict[str, float]:
        """
        Run two-proportion z-test with statsmodels.

        Conversion is defined as non-zero metric value.
        """
        treatment_conversions = int(np.sum(treatment_data != 0))
        control_conversions = int(np.sum(control_data != 0))

        n_treatment = len(treatment_data)
        n_control = len(control_data)

        p_treatment = treatment_conversions / n_treatment if n_treatment > 0 else 0.0
        p_control = control_conversions / n_control if n_control > 0 else 0.0

        if (
            (treatment_conversions == 0 and control_conversions == 0)
            or (treatment_conversions == n_treatment and control_conversions == n_control)
        ):
            return {
                "treatment_proportion": p_treatment,
                "control_proportion": p_control,
                "proportion_diff": p_treatment - p_control,
                "z_stat": 0.0,
                "p_value": 1.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
            }

        try:
            z_stat, p_value = test_proportions_2indep(
                treatment_conversions,
                n_treatment,
                control_conversions,
                n_control,
                method="wald",
                alternative="two-sided",
            )
            ci_lower, ci_upper = confint_proportions_2indep(
                treatment_conversions,
                n_treatment,
                control_conversions,
                n_control,
                method="wald",
            )

            return {
                "treatment_proportion": p_treatment,
                "control_proportion": p_control,
                "proportion_diff": p_treatment - p_control,
                "z_stat": float(z_stat),
                "p_value": float(p_value),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
            }
        except Exception:
            return {
                "treatment_proportion": p_treatment,
                "control_proportion": p_control,
                "proportion_diff": p_treatment - p_control,
                "z_stat": 0.0,
                "p_value": 1.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
            }

    def run_bayesian_test(
        self,
        treatment_post: np.ndarray,
        control_post: np.ndarray,
        treatment_pre: np.ndarray | None = None,
        control_pre: np.ndarray | None = None,
        n_samples: int | None = None,
    ) -> Dict[str, Any]:
        """
        Monte Carlo Bayesian effect estimate.

        Uses DiD deltas when pre-period metrics are available.
        """
        if n_samples is None:
            n_samples = self.bayesian_samples

        n_treatment = len(treatment_post)
        n_control = len(control_post)

        if n_treatment < 2 or n_control < 2:
            return {
                "prob_treatment_better": 0.5,
                "expected_loss_treatment": 0.0,
                "expected_loss_control": 0.0,
                "credible_interval": (0.0, 0.0),
                "relative_uplift": 0.0,
                "total_effect": 0.0,
                "did_effect": 0.0,
                "treatment_change": 0.0,
                "control_change": 0.0,
            }

        use_did = (
            treatment_pre is not None
            and control_pre is not None
            and len(treatment_pre) == n_treatment
            and len(control_pre) == n_control
        )

        if use_did:
            treatment_change = treatment_post - treatment_pre
            control_change = control_post - control_pre
            mean_t = float(np.mean(treatment_change))
            mean_c = float(np.mean(control_change))
            var_t = float(np.var(treatment_change, ddof=1)) if n_treatment > 1 else 0.0
            var_c = float(np.var(control_change, ddof=1)) if n_control > 1 else 0.0
            did_effect = mean_t - mean_c
            treatment_change_mean = mean_t
            control_change_mean = mean_c
        else:
            mean_t = float(np.mean(treatment_post))
            mean_c = float(np.mean(control_post))
            var_t = float(np.var(treatment_post, ddof=1)) if n_treatment > 1 else 0.0
            var_c = float(np.var(control_post, ddof=1)) if n_control > 1 else 0.0
            did_effect = 0.0
            treatment_change_mean = 0.0
            control_change_mean = 0.0

        if var_t == 0 and var_c == 0:
            diff = mean_t - mean_c
            total_effect = did_effect * n_treatment if use_did else diff * n_treatment
            return {
                "prob_treatment_better": 1.0 if diff > 0 else (0.0 if diff < 0 else 0.5),
                "expected_loss_treatment": 0.0 if diff >= 0 else abs(diff),
                "expected_loss_control": 0.0 if diff <= 0 else abs(diff),
                "credible_interval": (diff, diff),
                "relative_uplift": (diff / mean_c) if mean_c != 0 else 0.0,
                "total_effect": float(total_effect),
                "did_effect": float(did_effect),
                "treatment_change": float(treatment_change_mean),
                "control_change": float(control_change_mean),
            }

        se_t = np.sqrt(max(var_t, 1e-10) / n_treatment)
        se_c = np.sqrt(max(var_c, 1e-10) / n_control)

        np.random.seed(42)

        if n_treatment > 30 and n_control > 30:
            treatment_samples = np.random.normal(mean_t, se_t, n_samples)
            control_samples = np.random.normal(mean_c, se_c, n_samples)
        else:
            treatment_samples = mean_t + se_t * np.random.standard_t(max(n_treatment - 1, 1), n_samples)
            control_samples = mean_c + se_c * np.random.standard_t(max(n_control - 1, 1), n_samples)

        diff_samples = treatment_samples - control_samples

        prob_treatment_better = float(np.mean(diff_samples > 0))
        expected_loss_treatment = float(np.mean(np.abs(diff_samples) * (diff_samples < 0)))
        expected_loss_control = float(np.mean(np.abs(diff_samples) * (diff_samples > 0)))

        ci_lower = float(np.percentile(diff_samples, 2.5))
        ci_upper = float(np.percentile(diff_samples, 97.5))

        mean_diff = float(np.mean(diff_samples))
        if use_did:
            baseline = float(np.mean(control_post)) if abs(mean_c) < 1e-10 else mean_c
            relative_uplift = mean_diff / abs(baseline) if baseline != 0 else 0.0
        else:
            relative_uplift = mean_diff / mean_c if mean_c != 0 else 0.0

        total_effect = mean_diff * n_treatment

        return {
            "prob_treatment_better": prob_treatment_better,
            "expected_loss_treatment": expected_loss_treatment,
            "expected_loss_control": expected_loss_control,
            "credible_interval": (ci_lower, ci_upper),
            "relative_uplift": float(relative_uplift),
            "total_effect": float(total_effect),
            "did_effect": float(did_effect),
            "treatment_change": float(treatment_change_mean),
            "control_change": float(control_change_mean),
        }

"""
Statsmodels-driven statistical engine for A/B testing.

This module centralizes inferential statistics and keeps computation logic
separate from data management/orchestration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chisquare, kurtosis, levene, norm, normaltest, shapiro, skew, trim_mean
from scipy.stats.mstats import winsorize
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import confint_proportions_2indep, test_proportions_2indep
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from .models import AATestResult


class StatsmodelsABTestEngine:
    """Encapsulates frequentist and Bayesian A/B test computations."""

    MIN_RECOMMENDED_SAMPLE_SIZE = 8
    MIN_EXPECTED_PROPORTION_CELL = 5
    VARIANCE_EPSILON = 1e-12
    DEFAULT_EXPECTED_TREATMENT_RATIO = 0.5
    DEFAULT_WINSOR_LIMIT = 0.05
    DEFAULT_TRIM_FRACTION = 0.10
    DEFAULT_COUNT_MODEL = "auto"
    DEFAULT_HEAVY_TAIL_STRATEGY = "robust"
    DEFAULT_SEQUENTIAL_METHOD = "obrien_fleming"
    DEFAULT_FUTILITY_MIN_INFORMATION_FRACTION = 0.75
    DEFAULT_FUTILITY_P_VALUE_THRESHOLD = 0.5

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

    @staticmethod
    def _sanitize_numeric(values: np.ndarray) -> Tuple[np.ndarray, int]:
        """Return finite numeric values and count of removed invalid entries."""
        array = np.asarray(values, dtype=float).reshape(-1)
        finite_mask = np.isfinite(array)
        removed = int(array.size - np.count_nonzero(finite_mask))
        return array[finite_mask], removed

    @staticmethod
    def _sanitize_p_value(value: Any, fallback: float = 1.0) -> float:
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

    @staticmethod
    def _build_diagnostics(
        reasons: List[str],
        *,
        blocks_significance: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "guardrail_triggered": bool(reasons),
            "blocks_significance": blocks_significance,
            "reasons": reasons,
        }
        diagnostics.update(kwargs)
        return diagnostics

    def evaluate_sequential_decision(
        self,
        *,
        p_value: float,
        effect_size: float,
        confidence_interval: Tuple[float, float],
        look_index: int,
        max_looks: int,
        method: str = DEFAULT_SEQUENTIAL_METHOD,
        futility_min_information_fraction: float = DEFAULT_FUTILITY_MIN_INFORMATION_FRACTION,
        futility_p_value_threshold: float = DEFAULT_FUTILITY_P_VALUE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Pragmatic group-sequential decision support for interim looks.

        Uses alpha-spending to derive an interim efficacy threshold and applies a
        conservative futility recommendation when the interval still overlaps zero
        and observed evidence is weak late in the sequence.
        """
        safe_max_looks = max(int(max_looks), 1)
        safe_look_index = min(max(int(look_index), 1), safe_max_looks)
        information_fraction = float(safe_look_index / safe_max_looks)
        safe_method = str(method or self.DEFAULT_SEQUENTIAL_METHOD).strip().lower()
        safe_futility_info = float(
            min(max(futility_min_information_fraction, 0.0), 1.0)
        )
        safe_futility_p = self._sanitize_p_value(
            futility_p_value_threshold,
            fallback=self.DEFAULT_FUTILITY_P_VALUE_THRESHOLD,
        )

        if safe_method == "pocock":
            alpha_spent = float(
                self.significance_level
                * (np.log(1.0 + (np.e - 1.0) * information_fraction))
            )
        else:
            # O'Brien-Fleming-like alpha spending (two-sided).
            safe_method = "obrien_fleming"
            z_alpha_two_sided = float(norm.ppf(1.0 - self.significance_level / 2.0))
            alpha_spent = float(
                2.0 - 2.0 * norm.cdf(z_alpha_two_sided / np.sqrt(information_fraction))
            )

        alpha_spent = float(min(max(alpha_spent, 0.0), self.significance_level))
        ci_low, ci_high = float(confidence_interval[0]), float(confidence_interval[1])
        ci_crosses_zero = ci_low <= 0.0 <= ci_high
        is_final_look = safe_look_index >= safe_max_looks
        sanitized_p_value = self._sanitize_p_value(p_value)

        stop_recommended = False
        decision = "continue"
        rationale = (
            f"Interim look {safe_look_index}/{safe_max_looks}: "
            f"p={sanitized_p_value:.4g} does not cross efficacy ({alpha_spent:.4g}) "
            "or futility thresholds."
        )

        if is_final_look:
            stop_recommended = True
            if sanitized_p_value <= self.significance_level:
                decision = "final_accept"
                rationale = (
                    f"Final look reached with p={sanitized_p_value:.4g} <= "
                    f"alpha={self.significance_level:.4g}; recommend concluding efficacy."
                )
            else:
                decision = "final_reject"
                rationale = (
                    f"Final look reached with p={sanitized_p_value:.4g} > "
                    f"alpha={self.significance_level:.4g}; recommend concluding no effect."
                )
        elif sanitized_p_value <= alpha_spent:
            stop_recommended = True
            decision = "stop_efficacy"
            rationale = (
                f"Interim efficacy boundary crossed: p={sanitized_p_value:.4g} <= "
                f"alpha_spent={alpha_spent:.4g} at look {safe_look_index}/{safe_max_looks}."
            )
        elif (
            information_fraction >= safe_futility_info
            and sanitized_p_value >= safe_futility_p
            and ci_crosses_zero
        ):
            stop_recommended = True
            decision = "stop_futility"
            rationale = (
                f"Futility threshold met at look {safe_look_index}/{safe_max_looks}: "
                f"p={sanitized_p_value:.4g} >= {safe_futility_p:.4g}, "
                f"information_fraction={information_fraction:.3f} >= {safe_futility_info:.3f}, "
                "and confidence interval crosses zero."
            )
        else:
            decision = "continue"
            rationale = (
                f"Continue to next look: p={sanitized_p_value:.4g}, "
                f"efficacy threshold={alpha_spent:.4g}, "
                f"information_fraction={information_fraction:.3f}."
            )

        return {
            "enabled": True,
            "method": safe_method,
            "look_index": safe_look_index,
            "max_looks": safe_max_looks,
            "information_fraction": information_fraction,
            "alpha_spent": alpha_spent,
            "stop_recommended": stop_recommended,
            "decision": decision,
            "rationale": rationale,
            "thresholds": {
                "significance_level": float(self.significance_level),
                "efficacy_p_value_threshold": alpha_spent,
                "futility_p_value_threshold": safe_futility_p,
                "futility_min_information_fraction": safe_futility_info,
                "information_fraction": information_fraction,
                "ci_crosses_zero": ci_crosses_zero,
            },
            "observed": {
                "p_value": sanitized_p_value,
                "effect_size": float(effect_size),
                "confidence_interval": (ci_low, ci_high),
            },
        }

    @staticmethod
    def _coerce_covariate_frame(
        covariates: pd.DataFrame | np.ndarray | None,
        *,
        n_rows: int,
        names: Sequence[str] | None = None,
        prefix: str = "covariate",
    ) -> pd.DataFrame:
        """Coerce covariate inputs to a numeric DataFrame with stable column names."""
        if covariates is None:
            return pd.DataFrame(index=np.arange(n_rows))

        if isinstance(covariates, pd.DataFrame):
            frame = covariates.copy()
        else:
            array = np.asarray(covariates)
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            if array.ndim != 2:
                return pd.DataFrame(index=np.arange(n_rows))
            frame = pd.DataFrame(array)

        if len(frame) != n_rows:
            return pd.DataFrame(index=np.arange(n_rows))

        frame = frame.reset_index(drop=True)
        frame = frame.apply(pd.to_numeric, errors="coerce")

        if names is not None and len(names) == frame.shape[1]:
            frame.columns = [str(name) for name in names]
        else:
            frame.columns = [f"{prefix}_{idx + 1}" for idx in range(frame.shape[1])]

        return frame

    def _infer_metric_type(self, values: np.ndarray, requested_metric_type: str = "auto") -> str:
        """Infer metric family unless an explicit supported type is requested."""
        requested = str(requested_metric_type or "auto").strip().lower()
        supported = {"auto", "continuous", "binary", "count", "heavy_tail"}
        if requested in supported and requested != "auto":
            return requested

        if self._is_binary_metric(values):
            return "binary"
        if self._is_count_metric(values):
            return "count"
        if self._is_heavy_tail_metric(values):
            return "heavy_tail"
        return "continuous"

    @staticmethod
    def _is_binary_metric(values: np.ndarray) -> bool:
        finite = np.asarray(values, dtype=float)
        if finite.size == 0:
            return False
        unique_vals = np.unique(finite)
        return bool(np.all(np.isin(unique_vals, [0.0, 1.0])))

    def _is_count_metric(self, values: np.ndarray) -> bool:
        finite = np.asarray(values, dtype=float)
        if finite.size == 0:
            return False
        if np.any(finite < 0):
            return False
        if self._is_binary_metric(finite):
            return False
        return bool(np.all(np.isclose(finite, np.round(finite), atol=1e-8)))

    def _is_heavy_tail_metric(self, values: np.ndarray) -> bool:
        finite = np.asarray(values, dtype=float)
        if finite.size < 20:
            return False
        try:
            abs_skew = abs(float(skew(finite, bias=False)))
            excess_kurtosis = float(kurtosis(finite, fisher=True, bias=False))
        except Exception:
            return False

        if not np.isfinite(abs_skew) or not np.isfinite(excess_kurtosis):
            return False

        if abs_skew > 2.0 or excess_kurtosis > 7.0:
            return True

        p95 = float(np.percentile(finite, 95))
        p99 = float(np.percentile(finite, 99))
        p50 = float(np.percentile(finite, 50))
        spread = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        if abs(p50) <= self.VARIANCE_EPSILON:
            return (p99 - p95) > max(5.0 * spread, 1.0)
        return abs(p99 / p50) > 15.0

    def _extract_term_inference(
        self,
        model_result: Any,
        *,
        term: str = "treatment",
    ) -> Dict[str, float]:
        """
        Extract coefficient, inferential statistic, p-value, and CI for a model term.
        """
        params = model_result.params
        try:
            coef = float(params[term])
            term_index = list(getattr(params, "index", [])).index(term)
        except Exception:
            term_index = 1
            coef = float(np.asarray(params)[term_index])

        stat_value: float
        if hasattr(model_result, "tvalues"):
            try:
                stat_value = float(model_result.tvalues[term])
            except Exception:
                stat_value = float(np.asarray(model_result.tvalues)[term_index])
        elif hasattr(model_result, "zvalues"):
            try:
                stat_value = float(model_result.zvalues[term])
            except Exception:
                stat_value = float(np.asarray(model_result.zvalues)[term_index])
        else:
            stat_value = 0.0

        try:
            raw_p = model_result.pvalues[term]
        except Exception:
            raw_p = np.asarray(model_result.pvalues)[term_index] if hasattr(model_result, "pvalues") else 1.0
        p_value = self._sanitize_p_value(raw_p)

        try:
            conf = model_result.conf_int(alpha=self.significance_level)
            if hasattr(conf, "loc"):
                ci_low = float(conf.loc[term][0])
                ci_high = float(conf.loc[term][1])
            else:
                row = np.asarray(conf)[term_index]
                ci_low = float(row[0])
                ci_high = float(row[1])
        except Exception:
            ci_low, ci_high = coef, coef

        return {
            "coef": float(coef),
            "statistic": float(stat_value),
            "p_value": float(p_value),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }

    def run_srm_diagnostics(
        self,
        treatment_size: int,
        control_size: int,
        expected_treatment_ratio: float = DEFAULT_EXPECTED_TREATMENT_RATIO,
    ) -> Dict[str, Any]:
        """
        Sample Ratio Mismatch (SRM) diagnostics via chi-square goodness-of-fit.
        """
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
            diagnostics["p_value"] = self._sanitize_p_value(p_value)
            diagnostics["is_sample_ratio_mismatch"] = (
                diagnostics["p_value"] < self.significance_level
            )
            return diagnostics
        except Exception:
            diagnostics["is_applicable"] = False
            diagnostics["reason"] = "srm_test_failed"
            return diagnostics

    def run_assumption_diagnostics(
        self,
        treatment_data: np.ndarray,
        control_data: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Assumption diagnostics for mean-comparison inference.

        Includes normality checks (Shapiro or D'Agostino K^2) and a
        Brown-Forsythe/Levene variance homogeneity test.
        """
        treatment_data, treatment_invalid = self._sanitize_numeric(treatment_data)
        control_data, control_invalid = self._sanitize_numeric(control_data)

        n_treatment = len(treatment_data)
        n_control = len(control_data)
        min_n = min(n_treatment, n_control)
        max_n = max(n_treatment, n_control)

        treatment_var = (
            float(np.var(treatment_data, ddof=1)) if n_treatment >= 2 else 0.0
        )
        control_var = (
            float(np.var(control_data, ddof=1)) if n_control >= 2 else 0.0
        )

        # Use one normality test family consistently across both groups.
        if min_n >= 8 and max_n > 5000:
            normality_test = "dagostino_k2"
        elif min_n >= 3:
            normality_test = "shapiro"
        else:
            normality_test = "not_applicable"

        def _normality_p_value(values: np.ndarray, variance: float) -> float | None:
            if normality_test == "not_applicable":
                return None
            if len(values) < 3:
                return None
            if variance <= self.VARIANCE_EPSILON:
                return None
            try:
                if normality_test == "dagostino_k2":
                    _stat, p_val = normaltest(values)
                else:
                    _stat, p_val = shapiro(values)
                return self._sanitize_p_value(p_val)
            except Exception:
                return None

        treatment_normality_p = _normality_p_value(treatment_data, treatment_var)
        control_normality_p = _normality_p_value(control_data, control_var)

        treatment_normality_passed = (
            treatment_normality_p > self.significance_level
            if treatment_normality_p is not None
            else None
        )
        control_normality_passed = (
            control_normality_p > self.significance_level
            if control_normality_p is not None
            else None
        )

        variance_test = "not_applicable"
        equal_variance_p_value: float | None = None
        equal_variance_passed: bool | None = None
        variance_reason: str | None = None
        if n_treatment >= 2 and n_control >= 2:
            variance_test = "levene"
            try:
                _stat, p_val = levene(treatment_data, control_data, center="median")
                equal_variance_p_value = self._sanitize_p_value(p_val)
                equal_variance_passed = equal_variance_p_value > self.significance_level
            except Exception:
                variance_reason = "variance_test_failed"
        else:
            variance_reason = "insufficient_sample_size"

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
        self,
        treatment_data: np.ndarray,
        control_data: np.ndarray,
        baseline_effect: float | None = None,
        winsor_limit: float = DEFAULT_WINSOR_LIMIT,
        trim_fraction: float = DEFAULT_TRIM_FRACTION,
    ) -> Dict[str, Any]:
        """
        Estimate sensitivity of the mean effect to outlier-robust transforms.
        """
        treatment_data, treatment_invalid = self._sanitize_numeric(treatment_data)
        control_data, control_invalid = self._sanitize_numeric(control_data)

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
            winsorized_effect = float(
                np.mean(winsorized_treatment) - np.mean(winsorized_control)
            )
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
        if pooled_scale <= self.VARIANCE_EPSILON:
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
            n_required_scalar = float(np.atleast_1d(n_required)[0])
            return int(np.ceil(n_required_scalar))
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
        *,
        metric_type: str = "auto",
        treatment_covariates: pd.DataFrame | np.ndarray | None = None,
        control_covariates: pd.DataFrame | np.ndarray | None = None,
        covariate_names: Sequence[str] | None = None,
        count_model: str = DEFAULT_COUNT_MODEL,
        heavy_tail_strategy: str = DEFAULT_HEAVY_TAIL_STRATEGY,
    ) -> Dict[str, Any]:
        """
        Estimate treatment effect using a metric-aware statsmodels model family.

        Supported auto-selected families:
        - binary: GLM Binomial (logit link)
        - count: GLM Poisson / Negative Binomial
        - heavy_tail: robust linear model (Huber) or log-transform OLS
        - continuous: OLS with HC3 robust covariance
        """
        treatment_array = np.asarray(treatment_data, dtype=float).reshape(-1)
        control_array = np.asarray(control_data, dtype=float).reshape(-1)

        treatment_covariate_frame = self._coerce_covariate_frame(
            treatment_covariates,
            n_rows=len(treatment_array),
            names=covariate_names,
            prefix="covariate",
        )
        control_covariate_frame = self._coerce_covariate_frame(
            control_covariates,
            n_rows=len(control_array),
            names=covariate_names,
            prefix="covariate",
        )

        if treatment_covariate_frame.shape[1] != control_covariate_frame.shape[1]:
            treatment_covariate_frame = pd.DataFrame(index=np.arange(len(treatment_array)))
            control_covariate_frame = pd.DataFrame(index=np.arange(len(control_array)))

        covariate_columns = list(treatment_covariate_frame.columns)

        treatment_frame = pd.DataFrame({"y": treatment_array, "treatment": 1.0})
        control_frame = pd.DataFrame({"y": control_array, "treatment": 0.0})
        for column in covariate_columns:
            treatment_frame[column] = treatment_covariate_frame[column]
            control_frame[column] = control_covariate_frame[column]

        model_frame = pd.concat([treatment_frame, control_frame], ignore_index=True)
        numeric_columns = ["y", "treatment", *covariate_columns]
        finite_mask = np.isfinite(model_frame[numeric_columns].to_numpy(dtype=float)).all(axis=1)
        non_finite_removed = int(len(model_frame) - np.count_nonzero(finite_mask))
        model_frame = model_frame.loc[finite_mask].reset_index(drop=True)

        if covariate_columns:
            covariate_columns = [
                column
                for column in covariate_columns
                if model_frame[column].nunique(dropna=True) > 1
            ]

        treatment_values = model_frame.loc[model_frame["treatment"] == 1.0, "y"].to_numpy()
        control_values = model_frame.loc[model_frame["treatment"] == 0.0, "y"].to_numpy()

        n_treatment = len(treatment_values)
        n_control = len(control_values)
        treatment_mean = float(np.mean(treatment_values)) if n_treatment > 0 else 0.0
        control_mean = float(np.mean(control_values)) if n_control > 0 else 0.0
        raw_effect = self._zero_if_tiny(treatment_mean - control_mean)

        reasons: List[str] = []
        if non_finite_removed > 0:
            reasons.append("non_finite_values_removed")

        small_n = (
            n_treatment < self.MIN_RECOMMENDED_SAMPLE_SIZE
            or n_control < self.MIN_RECOMMENDED_SAMPLE_SIZE
        )
        if small_n:
            reasons.append("small_sample_size")

        if n_treatment < 2 or n_control < 2:
            diagnostics = self._build_diagnostics(
                [*reasons, "insufficient_sample_size"],
                blocks_significance=True,
                small_n=True,
                degenerate_variance=False,
                non_finite_values_removed=non_finite_removed,
            )
            return {
                "treatment_mean": treatment_mean,
                "control_mean": control_mean,
                "effect_size": 0.0,
                "t_statistic": 0.0,
                "p_value": 1.0,
                "confidence_interval": (0.0, 0.0),
                "metric_type": "continuous",
                "model_type": "insufficient_data",
                "model_effect": 0.0,
                "model_confidence_interval": (0.0, 0.0),
                "model_effect_scale": "mean_difference",
                "model_effect_exponentiated": 1.0,
                "covariate_adjusted": bool(covariate_columns),
                "covariates_used": covariate_columns,
                "diagnostics": diagnostics,
            }

        var_treatment = float(np.var(treatment_values, ddof=1))
        var_control = float(np.var(control_values, ddof=1))
        degenerate_variance = (
            var_treatment <= self.VARIANCE_EPSILON or var_control <= self.VARIANCE_EPSILON
        )
        if degenerate_variance:
            reasons.append("degenerate_variance")

        selected_metric_type = self._infer_metric_type(
            model_frame["y"].to_numpy(dtype=float),
            requested_metric_type=metric_type,
        )

        if (
            var_treatment <= self.VARIANCE_EPSILON
            and var_control <= self.VARIANCE_EPSILON
        ):
            # Deterministic case: when both groups are constant, significance depends
            # entirely on whether their means differ.
            if raw_effect == 0.0:
                t_statistic = 0.0
                p_value = 1.0
            else:
                t_statistic = float(np.sign(raw_effect) * np.inf)
                p_value = 0.0

            diagnostics = self._build_diagnostics(
                reasons,
                blocks_significance=False,
                small_n=small_n,
                degenerate_variance=True,
                non_finite_values_removed=non_finite_removed,
                metric_type=selected_metric_type,
                model_type="deterministic",
            )
            return {
                "treatment_mean": treatment_mean,
                "control_mean": control_mean,
                "effect_size": raw_effect,
                "t_statistic": t_statistic,
                "p_value": p_value,
                "confidence_interval": (raw_effect, raw_effect),
                "metric_type": selected_metric_type,
                "model_type": "deterministic",
                "model_effect": raw_effect,
                "model_confidence_interval": (raw_effect, raw_effect),
                "model_effect_scale": "mean_difference",
                "model_effect_exponentiated": 1.0,
                "covariate_adjusted": bool(covariate_columns),
                "covariates_used": covariate_columns,
                "diagnostics": diagnostics,
            }

        if (
            selected_metric_type in {"binary", "count"}
            and np.unique(model_frame["y"].to_numpy(dtype=float)).size <= 1
        ):
            diagnostics = self._build_diagnostics(
                [*reasons, "degenerate_outcome"],
                blocks_significance=True,
                small_n=small_n,
                degenerate_variance=degenerate_variance,
                non_finite_values_removed=non_finite_removed,
                metric_type=selected_metric_type,
                model_type=f"glm_{selected_metric_type}",
            )
            return {
                "treatment_mean": treatment_mean,
                "control_mean": control_mean,
                "effect_size": raw_effect,
                "t_statistic": 0.0,
                "p_value": 1.0,
                "confidence_interval": (0.0, 0.0),
                "metric_type": selected_metric_type,
                "model_type": f"glm_{selected_metric_type}",
                "model_effect": 0.0,
                "model_confidence_interval": (0.0, 0.0),
                "model_effect_scale": "log_odds" if selected_metric_type == "binary" else "log_rate",
                "model_effect_exponentiated": 1.0,
                "covariate_adjusted": bool(covariate_columns),
                "covariates_used": covariate_columns,
                "diagnostics": diagnostics,
            }

        X = sm.add_constant(model_frame[["treatment", *covariate_columns]], has_constant="add")
        y = model_frame["y"].to_numpy(dtype=float)

        model_type = "ols_hc3"
        model_effect_scale = "mean_difference"
        model_effect_exponentiated = 1.0
        model_fit_reason: str | None = None

        try:
            if selected_metric_type == "binary":
                model_type = "glm_binomial"
                model_effect_scale = "log_odds"
                fitted = sm.GLM(y, X, family=sm.families.Binomial()).fit()
                term_stats = self._extract_term_inference(fitted, term="treatment")
                model_effect_exponentiated = float(np.exp(term_stats["coef"]))
            elif selected_metric_type == "count":
                requested_count_model = str(count_model or self.DEFAULT_COUNT_MODEL).strip().lower()
                mean_y = float(np.mean(y)) if y.size else 0.0
                var_y = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
                overdispersed = mean_y > self.VARIANCE_EPSILON and var_y > 1.5 * mean_y

                use_negative_binomial = (
                    requested_count_model == "negative_binomial"
                    or (requested_count_model == "auto" and overdispersed)
                )
                family = (
                    sm.families.NegativeBinomial(alpha=1.0)
                    if use_negative_binomial
                    else sm.families.Poisson()
                )
                model_type = "glm_negative_binomial" if use_negative_binomial else "glm_poisson"
                model_effect_scale = "log_rate"
                fitted = sm.GLM(y, X, family=family).fit()
                term_stats = self._extract_term_inference(fitted, term="treatment")
                model_effect_exponentiated = float(np.exp(term_stats["coef"]))
            elif selected_metric_type == "heavy_tail":
                strategy = str(heavy_tail_strategy or self.DEFAULT_HEAVY_TAIL_STRATEGY).strip().lower()
                if strategy == "log_transform":
                    if np.min(y) <= -1.0:
                        shift = abs(float(np.min(y))) + 1.0
                        transformed_y = np.log(y + shift)
                    else:
                        transformed_y = np.log1p(y)
                    model_type = "ols_log1p_hc3"
                    model_effect_scale = "log_mean_difference"
                    fitted = sm.OLS(transformed_y, X).fit(cov_type="HC3")
                    term_stats = self._extract_term_inference(fitted, term="treatment")
                    model_effect_exponentiated = float(np.exp(term_stats["coef"]))
                else:
                    model_type = "rlm_huber"
                    model_effect_scale = "location_shift"
                    fitted = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
                    term_stats = self._extract_term_inference(fitted, term="treatment")
            else:
                model_type = "ols_hc3"
                model_effect_scale = "mean_difference"
                fitted = sm.OLS(y, X).fit(cov_type="HC3")
                term_stats = self._extract_term_inference(fitted, term="treatment")

        except Exception:
            # OLS fallback keeps behavior resilient when specialized families fail.
            model_fit_reason = "model_fit_failed_fallback_to_ols"
            try:
                model_type = "ols_hc3_fallback"
                model_effect_scale = "mean_difference"
                fitted = sm.OLS(y, X).fit(cov_type="HC3")
                term_stats = self._extract_term_inference(fitted, term="treatment")
            except Exception:
                d_treatment = DescrStatsW(treatment_values)
                d_control = DescrStatsW(control_values)
                compare = CompareMeans(d_treatment, d_control)
                t_stat, p_val, _df = compare.ttest_ind(usevar="unequal")
                ci_low, ci_high = compare.tconfint_diff(
                    alpha=self.significance_level,
                    usevar="unequal",
                )
                model_type = "compare_means_fallback"
                model_effect_scale = "mean_difference"
                term_stats = {
                    "coef": raw_effect,
                    "statistic": float(t_stat),
                    "p_value": self._sanitize_p_value(p_val),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                }

        model_effect = self._zero_if_tiny(float(term_stats["coef"]))
        raw_p_value = float(term_stats["p_value"])
        p_value = self._sanitize_p_value(raw_p_value)
        if p_value != raw_p_value:
            reasons.append("invalid_p_value")
        if model_fit_reason is not None:
            reasons.append(model_fit_reason)

        diagnostics = self._build_diagnostics(
            reasons,
            blocks_significance=small_n,
            small_n=small_n,
            degenerate_variance=degenerate_variance,
            non_finite_values_removed=non_finite_removed,
            metric_type=selected_metric_type,
            model_type=model_type,
            covariate_adjusted=bool(covariate_columns),
            covariates_used=covariate_columns,
        )

        return {
            "treatment_mean": treatment_mean,
            "control_mean": control_mean,
            "effect_size": raw_effect,
            "t_statistic": float(term_stats["statistic"]),
            "p_value": p_value,
            "confidence_interval": (float(term_stats["ci_low"]), float(term_stats["ci_high"])),
            "metric_type": selected_metric_type,
            "model_type": model_type,
            "model_effect": model_effect,
            "model_confidence_interval": (
                float(term_stats["ci_low"]),
                float(term_stats["ci_high"]),
            ),
            "model_effect_scale": model_effect_scale,
            "model_effect_exponentiated": float(model_effect_exponentiated),
            "covariate_adjusted": bool(covariate_columns),
            "covariates_used": covariate_columns,
            "diagnostics": diagnostics,
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
    ) -> Dict[str, Any]:
        """
        Run two-proportion z-test with statsmodels.

        Conversion is defined as non-zero metric value.
        """
        treatment_data, treatment_invalid = self._sanitize_numeric(treatment_data)
        control_data, control_invalid = self._sanitize_numeric(control_data)

        treatment_conversions = int(np.sum(treatment_data != 0))
        control_conversions = int(np.sum(control_data != 0))

        n_treatment = len(treatment_data)
        n_control = len(control_data)

        p_treatment = treatment_conversions / n_treatment if n_treatment > 0 else 0.0
        p_control = control_conversions / n_control if n_control > 0 else 0.0

        non_finite_removed = treatment_invalid + control_invalid
        small_n = (
            n_treatment < self.MIN_RECOMMENDED_SAMPLE_SIZE
            or n_control < self.MIN_RECOMMENDED_SAMPLE_SIZE
        )
        invalid_inputs = n_treatment <= 0 or n_control <= 0
        degenerate_proportions = (
            n_treatment > 0
            and n_control > 0
            and (
                (treatment_conversions == 0 and control_conversions == 0)
                or (treatment_conversions == n_treatment and control_conversions == n_control)
            )
        )
        expected_counts_too_small = False
        if n_treatment > 0 and n_control > 0:
            expected_counts = [
                treatment_conversions,
                n_treatment - treatment_conversions,
                control_conversions,
                n_control - control_conversions,
            ]
            expected_counts_too_small = any(
                count < self.MIN_EXPECTED_PROPORTION_CELL for count in expected_counts
            )

        reasons: List[str] = []
        if non_finite_removed > 0:
            reasons.append("non_finite_values_removed")
        if invalid_inputs:
            reasons.append("invalid_proportion_inputs")
        if small_n:
            reasons.append("small_sample_size")
        if expected_counts_too_small:
            reasons.append("expected_counts_too_small")
        if degenerate_proportions:
            reasons.append("degenerate_proportions")

        blocks_significance = (
            invalid_inputs or small_n or expected_counts_too_small or degenerate_proportions
        )

        diagnostics = self._build_diagnostics(
            reasons,
            blocks_significance=blocks_significance,
            small_n=small_n,
            invalid_proportion_inputs=invalid_inputs,
            expected_counts_too_small=expected_counts_too_small,
            degenerate_proportions=degenerate_proportions,
            non_finite_values_removed=non_finite_removed,
        )

        if invalid_inputs or degenerate_proportions:
            return {
                "treatment_proportion": p_treatment,
                "control_proportion": p_control,
                "proportion_diff": p_treatment - p_control,
                "z_stat": 0.0,
                "p_value": 1.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "diagnostics": diagnostics,
            }

        try:
            z_stat, p_value = test_proportions_2indep(
                treatment_conversions,
                n_treatment,
                control_conversions,
                n_control,
                method="score",
                alternative="two-sided",
            )
            ci_lower, ci_upper = confint_proportions_2indep(
                treatment_conversions,
                n_treatment,
                control_conversions,
                n_control,
                method="score",
            )

            raw_p_value = float(p_value)
            if not np.isfinite(raw_p_value) or raw_p_value < 0.0 or raw_p_value > 1.0:
                raw_p_value = 1.0
                diagnostics["guardrail_triggered"] = True
                diagnostics["blocks_significance"] = True
                diagnostics["invalid_proportion_inputs"] = True
                diagnostics["reasons"] = [*diagnostics["reasons"], "invalid_p_value"]

            return {
                "treatment_proportion": p_treatment,
                "control_proportion": p_control,
                "proportion_diff": p_treatment - p_control,
                "z_stat": float(z_stat),
                "p_value": raw_p_value,
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "diagnostics": diagnostics,
            }
        except Exception:
            fallback_diagnostics = self._build_diagnostics(
                [*reasons, "proportion_test_failed"],
                blocks_significance=True,
                small_n=small_n,
                invalid_proportion_inputs=True,
                expected_counts_too_small=expected_counts_too_small,
                degenerate_proportions=degenerate_proportions,
                non_finite_values_removed=non_finite_removed,
            )
            return {
                "treatment_proportion": p_treatment,
                "control_proportion": p_control,
                "proportion_diff": p_treatment - p_control,
                "z_stat": 0.0,
                "p_value": 1.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "diagnostics": fallback_diagnostics,
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

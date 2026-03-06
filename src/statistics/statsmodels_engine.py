"""Statsmodels-driven statistical engine for A/B testing."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import confint_proportions_2indep, test_proportions_2indep

from .bayesian import run_bayesian_test as compute_bayesian_test
from .diagnostics import (
    run_assumption_diagnostics as compute_assumption_diagnostics,
    run_outlier_sensitivity as compute_outlier_sensitivity,
    run_srm_diagnostics as compute_srm_diagnostics,
)
from .engine_helpers import build_diagnostics, sanitize_numeric, sanitize_p_value, zero_if_tiny
from .experiment_design import (
    bootstrap_balanced_control as compute_bootstrap_balanced_control,
    run_aa_test as compute_aa_test,
)
from .model_families import (
    coerce_covariate_frame,
    estimate_did_effect as compute_did_effect,
    estimate_treatment_effect as compute_treatment_effect,
    extract_term_inference,
    infer_metric_type,
    is_binary_metric,
    is_count_metric,
    is_heavy_tail_metric,
)
from .models import AATestResult
from .power_analysis import (
    calculate_cohens_d as compute_cohens_d,
    calculate_power as compute_power,
    calculate_required_sample_size as compute_required_sample_size,
)


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
        return zero_if_tiny(value, tol)

    @staticmethod
    def _sanitize_numeric(values: np.ndarray) -> Tuple[np.ndarray, int]:
        return sanitize_numeric(values)

    @staticmethod
    def _sanitize_p_value(value: Any, fallback: float = 1.0) -> float:
        return sanitize_p_value(value, fallback)

    @staticmethod
    def _build_diagnostics(
        reasons: List[str],
        *,
        blocks_significance: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return build_diagnostics(reasons, blocks_significance=blocks_significance, **kwargs)

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
        """Pragmatic group-sequential decision support for interim looks."""
        safe_max_looks = max(int(max_looks), 1)
        safe_look_index = min(max(int(look_index), 1), safe_max_looks)
        information_fraction = float(safe_look_index / safe_max_looks)
        safe_method = str(method or self.DEFAULT_SEQUENTIAL_METHOD).strip().lower()
        safe_futility_info = float(min(max(futility_min_information_fraction, 0.0), 1.0))
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
        covariates: Any,
        *,
        n_rows: int,
        names: Sequence[str] | None = None,
        prefix: str = "covariate",
    ):
        return coerce_covariate_frame(
            covariates,
            n_rows=n_rows,
            names=names,
            prefix=prefix,
        )

    def _infer_metric_type(self, values: np.ndarray, requested_metric_type: str = "auto") -> str:
        return infer_metric_type(
            values,
            requested_metric_type=requested_metric_type,
            variance_epsilon=self.VARIANCE_EPSILON,
        )

    @staticmethod
    def _is_binary_metric(values: np.ndarray) -> bool:
        return is_binary_metric(values)

    def _is_count_metric(self, values: np.ndarray) -> bool:
        return is_count_metric(values)

    def _is_heavy_tail_metric(self, values: np.ndarray) -> bool:
        return is_heavy_tail_metric(values, variance_epsilon=self.VARIANCE_EPSILON)

    def _extract_term_inference(
        self,
        model_result: Any,
        *,
        term: str = "treatment",
    ) -> Dict[str, float]:
        return extract_term_inference(
            model_result,
            term=term,
            significance_level=self.significance_level,
        )

    def run_srm_diagnostics(
        self,
        treatment_size: int,
        control_size: int,
        expected_treatment_ratio: float = DEFAULT_EXPECTED_TREATMENT_RATIO,
    ) -> Dict[str, Any]:
        return compute_srm_diagnostics(
            treatment_size=treatment_size,
            control_size=control_size,
            expected_treatment_ratio=expected_treatment_ratio,
            significance_level=self.significance_level,
        )

    def run_assumption_diagnostics(
        self,
        treatment_data: np.ndarray,
        control_data: np.ndarray,
    ) -> Dict[str, Any]:
        return compute_assumption_diagnostics(
            treatment_data=treatment_data,
            control_data=control_data,
            significance_level=self.significance_level,
            variance_epsilon=self.VARIANCE_EPSILON,
        )

    def run_outlier_sensitivity(
        self,
        treatment_data: np.ndarray,
        control_data: np.ndarray,
        baseline_effect: float | None = None,
        winsor_limit: float = DEFAULT_WINSOR_LIMIT,
        trim_fraction: float = DEFAULT_TRIM_FRACTION,
    ) -> Dict[str, Any]:
        return compute_outlier_sensitivity(
            treatment_data=treatment_data,
            control_data=control_data,
            baseline_effect=baseline_effect,
            winsor_limit=winsor_limit,
            trim_fraction=trim_fraction,
            variance_epsilon=self.VARIANCE_EPSILON,
        )

    def calculate_cohens_d(self, treatment_data: np.ndarray, control_data: np.ndarray) -> float:
        return compute_cohens_d(treatment_data, control_data)

    def calculate_power(self, effect_size: float, n_treatment: int, n_control: int) -> float:
        return compute_power(
            effect_size=effect_size,
            n_treatment=n_treatment,
            n_control=n_control,
            significance_level=self.significance_level,
        )

    def calculate_required_sample_size(self, effect_size: float, ratio: float = 1.0) -> int:
        return compute_required_sample_size(
            effect_size=effect_size,
            ratio=ratio,
            power_threshold=self.power_threshold,
            significance_level=self.significance_level,
        )

    def run_aa_test(
        self,
        treatment_pre: np.ndarray,
        control_pre: np.ndarray,
        segment_name: str = "Overall",
    ) -> AATestResult:
        return compute_aa_test(
            treatment_pre=treatment_pre,
            control_pre=control_pre,
            segment_name=segment_name,
            significance_level=self.significance_level,
        )

    def bootstrap_balanced_control(
        self,
        treatment_pre: np.ndarray,
        control_df,
        pre_col: str,
        max_iterations: int = 1000,
        target_p_value: float = 0.10,
    ):
        return compute_bootstrap_balanced_control(
            treatment_pre=treatment_pre,
            control_df=control_df,
            pre_col=pre_col,
            max_iterations=max_iterations,
            target_p_value=target_p_value,
            significance_level=self.significance_level,
        )

    def estimate_treatment_effect(
        self,
        treatment_data: np.ndarray,
        control_data: np.ndarray,
        *,
        metric_type: str = "auto",
        treatment_covariates=None,
        control_covariates=None,
        covariate_names: Sequence[str] | None = None,
        count_model: str = DEFAULT_COUNT_MODEL,
        heavy_tail_strategy: str = DEFAULT_HEAVY_TAIL_STRATEGY,
    ) -> Dict[str, Any]:
        return compute_treatment_effect(
            treatment_data=treatment_data,
            control_data=control_data,
            significance_level=self.significance_level,
            min_recommended_sample_size=self.MIN_RECOMMENDED_SAMPLE_SIZE,
            variance_epsilon=self.VARIANCE_EPSILON,
            metric_type=metric_type,
            treatment_covariates=treatment_covariates,
            control_covariates=control_covariates,
            covariate_names=covariate_names,
            count_model=count_model,
            heavy_tail_strategy=heavy_tail_strategy,
        )

    def estimate_did_effect(
        self,
        treatment_pre: np.ndarray,
        treatment_post: np.ndarray,
        control_pre: np.ndarray,
        control_post: np.ndarray,
    ) -> Dict[str, float]:
        return compute_did_effect(
            treatment_pre=treatment_pre,
            treatment_post=treatment_post,
            control_pre=control_pre,
            control_post=control_post,
        )

    def run_proportion_test(
        self,
        treatment_data: np.ndarray,
        control_data: np.ndarray,
    ) -> Dict[str, Any]:
        """Run two-proportion z-test with statsmodels."""
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

        if invalid_inputs or expected_counts_too_small or degenerate_proportions:
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
        if n_samples is None:
            n_samples = self.bayesian_samples
        return compute_bayesian_test(
            treatment_post=treatment_post,
            control_post=control_post,
            treatment_pre=treatment_pre,
            control_pre=control_pre,
            n_samples=n_samples,
        )

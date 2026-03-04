"""
Summary and recommendation builder for A/B test result sets.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from .models import ABTestResult, normalize_ab_test_results


class ABTestSummaryBuilder:
    """Build summary payloads and recommendations from segment-level results."""

    def generate_summary(self, results: Iterable[Any]) -> Dict[str, Any]:
        """Generate comprehensive summary used by UI and agent output."""
        canonical_results = normalize_ab_test_results(results)
        results = canonical_results

        if not results:
            return {"error": "No results to summarize"}

        aa_failed_segments = [r for r in results if not r.aa_test_passed]
        bootstrapped_segments = [r for r in results if r.bootstrapping_applied]

        t_significant_results = [r for r in results if r.is_significant]
        t_significant_adjusted_results = [r for r in results if r.is_significant_adjusted]
        prop_significant_results = [r for r in results if r.proportion_is_significant]
        prop_significant_adjusted_results = [
            r for r in results if r.proportion_is_significant_adjusted
        ]
        bayesian_significant_results = [r for r in results if r.bayesian_is_significant]
        inference_guardrailed = [r for r in results if r.inference_guardrail_triggered]
        proportion_guardrailed = [r for r in results if r.proportion_guardrail_triggered]
        srm_mismatch_results = [
            r
            for r in results
            if bool(
                r.diagnostics.get("experiment_quality", {})
                .get("srm", {})
                .get("is_sample_ratio_mismatch", False)
            )
        ]
        assumption_warning_results = [
            r
            for r in results
            if (
                (
                    r.diagnostics.get("experiment_quality", {})
                    .get("assumptions", {})
                    .get("treatment_normality_passed")
                    is False
                )
                or (
                    r.diagnostics.get("experiment_quality", {})
                    .get("assumptions", {})
                    .get("control_normality_passed")
                    is False
                )
                or (
                    r.diagnostics.get("experiment_quality", {})
                    .get("assumptions", {})
                    .get("equal_variance_passed")
                    is False
                )
            )
        ]
        outlier_sensitive_results = [
            r
            for r in results
            if bool(
                r.diagnostics.get("experiment_quality", {})
                .get("outlier_sensitivity", {})
                .get("is_sensitive", False)
            )
        ]

        t_test_total_effect = sum(r.effect_size * r.treatment_size for r in t_significant_results)
        avg_t_test_effect = (
            float(np.mean([r.effect_size for r in t_significant_results]))
            if t_significant_results
            else 0.0
        )
        total_treatment_in_t_significant = sum(r.treatment_size for r in t_significant_results)

        prop_total_effect = sum(r.proportion_effect for r in prop_significant_results)
        avg_prop_effect = (
            float(np.mean([r.proportion_effect_per_customer for r in prop_significant_results]))
            if prop_significant_results
            else 0.0
        )
        total_treatment_in_prop_significant = sum(r.treatment_size for r in prop_significant_results)

        combined_total_effect = sum(r.total_effect for r in results)

        avg_did_effect = float(np.mean([r.did_effect for r in results]))
        total_did_effect = sum(r.did_effect * r.treatment_size for r in results)

        avg_bayesian_prob = float(np.mean([r.bayesian_prob_treatment_better for r in results]))
        avg_expected_loss = float(
            np.mean(
                [
                    min(r.bayesian_expected_loss_treatment, r.bayesian_expected_loss_control)
                    for r in results
                ]
            )
        )
        total_bayesian_effect = sum(r.bayesian_total_effect for r in results)

        adequate_samples = [r for r in results if r.is_sample_adequate]
        inadequate_samples = [r for r in results if not r.is_sample_adequate]

        total_treatment = sum(r.treatment_size for r in results)
        total_control = sum(r.control_size for r in results)
        multiple_testing_applied_segments = [r for r in results if r.multiple_testing_applied]
        multiple_testing_method = (
            multiple_testing_applied_segments[0].multiple_testing_method
            if multiple_testing_applied_segments
            else "none"
        )
        covariate_adjusted_segments = [r for r in results if r.covariate_adjustment_applied]
        sequential_enabled_segments = [r for r in results if r.sequential_mode_enabled]
        sequential_stop_segments = [r for r in sequential_enabled_segments if r.sequential_stop_recommended]
        sequential_continue_segments = [
            r for r in sequential_enabled_segments if r.sequential_decision == "continue"
        ]
        sequential_decision_breakdown: Dict[str, int] = {}
        for result in sequential_enabled_segments:
            decision = str(result.sequential_decision or "unknown")
            sequential_decision_breakdown[decision] = (
                sequential_decision_breakdown.get(decision, 0) + 1
            )

        model_type_breakdown: Dict[str, int] = {}
        metric_type_breakdown: Dict[str, int] = {}
        for result in results:
            model_type_breakdown[result.model_type] = model_type_breakdown.get(result.model_type, 0) + 1
            metric_type_breakdown[result.metric_type] = (
                metric_type_breakdown.get(result.metric_type, 0) + 1
            )

        return {
            "total_segments_analyzed": len(results),
            "aa_test_passed_segments": len(results) - len(aa_failed_segments),
            "aa_test_failed_segments": len(aa_failed_segments),
            "bootstrapped_segments": len(bootstrapped_segments),
            "aa_failed_segment_names": [r.segment for r in aa_failed_segments],
            "did_avg_effect": avg_did_effect,
            "did_total_effect": total_did_effect,
            "t_test_significant_segments": len(t_significant_results),
            "t_test_significant_segments_adjusted": len(t_significant_adjusted_results),
            "t_test_significance_rate": len(t_significant_results) / len(results) if results else 0.0,
            "t_test_significance_rate_adjusted": (
                len(t_significant_adjusted_results) / len(results) if results else 0.0
            ),
            "t_test_avg_effect": avg_t_test_effect,
            "t_test_total_effect": t_test_total_effect,
            "t_test_effect_calculation": f"{avg_t_test_effect:.4f} × {total_treatment_in_t_significant} = {t_test_total_effect:.2f}",
            "prop_test_significant_segments": len(prop_significant_results),
            "prop_test_significant_segments_adjusted": len(prop_significant_adjusted_results),
            "prop_test_significance_rate": len(prop_significant_results) / len(results) if results else 0.0,
            "prop_test_significance_rate_adjusted": (
                len(prop_significant_adjusted_results) / len(results) if results else 0.0
            ),
            "prop_test_avg_effect": avg_prop_effect,
            "prop_test_total_effect": prop_total_effect,
            "prop_test_effect_calculation": f"{avg_prop_effect:.4f} × {total_treatment_in_prop_significant} = {prop_total_effect:.2f}",
            "multiple_testing_method": multiple_testing_method,
            "multiple_testing_applied_segments": len(multiple_testing_applied_segments),
            "covariate_adjusted_segments": len(covariate_adjusted_segments),
            "sequential_mode_segments": len(sequential_enabled_segments),
            "sequential_stop_recommended_segments": len(sequential_stop_segments),
            "sequential_continue_segments": len(sequential_continue_segments),
            "sequential_stop_segment_names": [r.segment for r in sequential_stop_segments],
            "sequential_decision_breakdown": sequential_decision_breakdown,
            "metric_type_breakdown": metric_type_breakdown,
            "model_type_breakdown": model_type_breakdown,
            "inference_guardrail_segments": len(inference_guardrailed),
            "proportion_guardrail_segments": len(proportion_guardrailed),
            "srm_mismatch_segments": len(srm_mismatch_results),
            "srm_mismatch_segment_names": [r.segment for r in srm_mismatch_results],
            "assumption_warning_segments": len(assumption_warning_results),
            "assumption_warning_segment_names": [r.segment for r in assumption_warning_results],
            "outlier_sensitive_segments": len(outlier_sensitive_results),
            "outlier_sensitive_segment_names": [r.segment for r in outlier_sensitive_results],
            "guardrail_segment_names": sorted(
                {
                    r.segment
                    for r in [*inference_guardrailed, *proportion_guardrailed]
                }
            ),
            "combined_total_effect": combined_total_effect,
            "combined_effect_calculation": f"T-test ({t_test_total_effect:.2f}) + Proportion ({prop_total_effect:.2f}) = {combined_total_effect:.2f}",
            "bayesian_significant_segments": len(bayesian_significant_results),
            "bayesian_significance_rate": len(bayesian_significant_results) / len(results) if results else 0.0,
            "bayesian_avg_prob_treatment_better": avg_bayesian_prob,
            "bayesian_avg_expected_loss": avg_expected_loss,
            "bayesian_total_effect": total_bayesian_effect,
            # Legacy fields
            "significant_segments": len(t_significant_results),
            "non_significant_segments": len(results) - len(t_significant_results),
            "significance_rate": len(t_significant_results) / len(results) if results else 0.0,
            "average_significant_effect": avg_t_test_effect,
            "total_treatment_in_significant_segments": total_treatment_in_t_significant,
            "total_effect_size": t_test_total_effect,
            "effect_calculation": f"{avg_t_test_effect:.4f} × {total_treatment_in_t_significant} = {t_test_total_effect:.2f}",
            "total_treatment_customers": total_treatment,
            "total_control_customers": total_control,
            "treatment_control_ratio": total_treatment / total_control if total_control > 0 else None,
            "segments_with_adequate_power": len(adequate_samples),
            "segments_with_inadequate_power": len(inadequate_samples),
            "power_adequacy_rate": len(adequate_samples) / len(results) if results else 0.0,
            "detailed_results": [
                {
                    "segment": r.segment,
                    "treatment_n": r.treatment_size,
                    "control_n": r.control_size,
                    "treatment_pre_mean": r.treatment_pre_mean,
                    "treatment_post_mean": r.treatment_post_mean,
                    "control_pre_mean": r.control_pre_mean,
                    "control_post_mean": r.control_post_mean,
                    "aa_test_passed": r.aa_test_passed,
                    "aa_p_value": r.aa_p_value,
                    "bootstrapping_applied": r.bootstrapping_applied,
                    "original_control_size": r.original_control_size,
                    "did_treatment_change": r.did_treatment_change,
                    "did_control_change": r.did_control_change,
                    "did_effect": r.did_effect,
                    "effect": r.effect_size,
                    "cohens_d": r.cohens_d,
                    "p_value": r.p_value,
                    "significant": r.is_significant,
                    "p_value_adjusted": r.p_value_adjusted,
                    "significant_adjusted": r.is_significant_adjusted,
                    "power": r.power,
                    "adequate_sample": r.is_sample_adequate,
                    "ci_lower": r.confidence_interval[0],
                    "ci_upper": r.confidence_interval[1],
                    "metric_type": r.metric_type,
                    "model_type": r.model_type,
                    "model_effect": r.model_effect,
                    "model_ci_lower": r.model_confidence_interval[0],
                    "model_ci_upper": r.model_confidence_interval[1],
                    "model_effect_scale": r.model_effect_scale,
                    "model_effect_exponentiated": r.model_effect_exponentiated,
                    "covariate_adjustment_applied": r.covariate_adjustment_applied,
                    "covariates_used": r.covariates_used,
                    "covariate_adjusted_effect": r.covariate_adjusted_effect,
                    "covariate_adjusted_p_value": r.covariate_adjusted_p_value,
                    "covariate_adjusted_ci_lower": r.covariate_adjusted_confidence_interval[0],
                    "covariate_adjusted_ci_upper": r.covariate_adjusted_confidence_interval[1],
                    "covariate_adjusted_model_type": r.covariate_adjusted_model_type,
                    "covariate_adjusted_effect_scale": r.covariate_adjusted_effect_scale,
                    "covariate_adjusted_effect_exponentiated": r.covariate_adjusted_effect_exponentiated,
                    "treatment_prop": r.treatment_proportion,
                    "control_prop": r.control_proportion,
                    "prop_diff": r.proportion_diff,
                    "prop_p_value": r.proportion_p_value,
                    "prop_significant": r.proportion_is_significant,
                    "prop_p_value_adjusted": r.proportion_p_value_adjusted,
                    "prop_significant_adjusted": r.proportion_is_significant_adjusted,
                    "multiple_testing_method": r.multiple_testing_method,
                    "multiple_testing_applied": r.multiple_testing_applied,
                    "inference_guardrail_triggered": r.inference_guardrail_triggered,
                    "proportion_guardrail_triggered": r.proportion_guardrail_triggered,
                    "diagnostics": r.diagnostics,
                    "sequential_mode_enabled": r.sequential_mode_enabled,
                    "sequential_method": r.sequential_method,
                    "sequential_look_index": r.sequential_look_index,
                    "sequential_max_looks": r.sequential_max_looks,
                    "sequential_information_fraction": r.sequential_information_fraction,
                    "sequential_alpha_spent": r.sequential_alpha_spent,
                    "sequential_stop_recommended": r.sequential_stop_recommended,
                    "sequential_decision": r.sequential_decision,
                    "sequential_rationale": r.sequential_rationale,
                    "sequential_thresholds": r.sequential_thresholds,
                    "srm_p_value": (
                        r.diagnostics.get("experiment_quality", {})
                        .get("srm", {})
                        .get("p_value")
                    ),
                    "srm_is_mismatch": (
                        r.diagnostics.get("experiment_quality", {})
                        .get("srm", {})
                        .get("is_sample_ratio_mismatch", False)
                    ),
                    "assumption_diagnostics": (
                        r.diagnostics.get("experiment_quality", {})
                        .get("assumptions", {})
                    ),
                    "outlier_sensitivity_diagnostics": (
                        r.diagnostics.get("experiment_quality", {})
                        .get("outlier_sensitivity", {})
                    ),
                    "prop_effect": r.proportion_effect,
                    "prop_effect_per_customer": r.proportion_effect_per_customer,
                    "total_effect": r.total_effect,
                    "total_effect_per_customer": r.total_effect_per_customer,
                    "bayesian_prob": r.bayesian_prob_treatment_better,
                    "bayesian_credible_lower": r.bayesian_credible_interval[0],
                    "bayesian_credible_upper": r.bayesian_credible_interval[1],
                    "bayesian_expected_loss": min(
                        r.bayesian_expected_loss_treatment,
                        r.bayesian_expected_loss_control,
                    ),
                    "bayesian_relative_uplift": r.bayesian_relative_uplift,
                    "bayesian_significant": r.bayesian_is_significant,
                    "bayesian_total_effect": r.bayesian_total_effect,
                    "bayesian_total_effect_per_customer": r.bayesian_total_effect_per_customer,
                }
                for r in results
            ],
            "recommendations": self._generate_recommendations(results),
        }

    def _generate_recommendations(self, results: List[ABTestResult]) -> List[str]:
        """Generate actionable recommendations from result set."""
        recommendations: List[str] = []

        aa_failed = [r for r in results if not r.aa_test_passed]
        bootstrapped = [r for r in results if r.bootstrapping_applied]

        if aa_failed:
            segments = [r.segment for r in aa_failed]
            recommendations.append(
                f"AA TEST WARNING: {len(aa_failed)} segment(s) failed the AA test (imbalanced pre-experiment): {', '.join(segments)}. "
                "Treatment and control groups had different baseline characteristics."
            )

        if bootstrapped:
            segments = [
                f"{r.segment} (control: {r.original_control_size} → {r.control_size})"
                for r in bootstrapped
            ]
            recommendations.append(
                f"BOOTSTRAPPING APPLIED: {len(bootstrapped)} segment(s) used bootstrapped control group for balance: {', '.join(segments)}. "
                "Results should be interpreted with caution."
            )

        guardrailed = [
            r
            for r in results
            if r.inference_guardrail_triggered or r.proportion_guardrail_triggered
        ]
        if guardrailed:
            segments = [r.segment for r in guardrailed]
            recommendations.append(
                f"INFERENCE GUARDRAILS: {len(guardrailed)} segment(s) triggered stability checks ({', '.join(segments)}). "
                "Treat frequentist significance as directional and prioritize additional data collection."
            )

        sequential_enabled = [r for r in results if r.sequential_mode_enabled]
        sequential_stop = [r for r in sequential_enabled if r.sequential_stop_recommended]
        sequential_continue = [
            r for r in sequential_enabled if r.sequential_decision == "continue"
        ]
        if sequential_stop:
            segments = [f"{r.segment} ({r.sequential_decision})" for r in sequential_stop]
            recommendations.append(
                f"SEQUENTIAL DECISION: stop recommended for {len(sequential_stop)} segment(s): {', '.join(segments)}. "
                "Validate practical impact and guardrails before acting on early stopping."
            )
        if sequential_enabled and sequential_continue:
            segments = [r.segment for r in sequential_continue]
            recommendations.append(
                f"SEQUENTIAL DECISION: continue data collection for {len(sequential_continue)} segment(s): {', '.join(segments)}. "
                "Current interim evidence has not crossed configured efficacy/futility boundaries."
            )

        srm_mismatch = [
            r
            for r in results
            if bool(
                r.diagnostics.get("experiment_quality", {})
                .get("srm", {})
                .get("is_sample_ratio_mismatch", False)
            )
        ]
        if srm_mismatch:
            segments = [r.segment for r in srm_mismatch]
            recommendations.append(
                f"SRM WARNING: {len(srm_mismatch)} segment(s) show sample ratio mismatch ({', '.join(segments)}). "
                "Randomization may be compromised; validate assignment and traffic filters."
            )

        assumption_warnings = [
            r
            for r in results
            if (
                (
                    r.diagnostics.get("experiment_quality", {})
                    .get("assumptions", {})
                    .get("treatment_normality_passed")
                    is False
                )
                or (
                    r.diagnostics.get("experiment_quality", {})
                    .get("assumptions", {})
                    .get("control_normality_passed")
                    is False
                )
                or (
                    r.diagnostics.get("experiment_quality", {})
                    .get("assumptions", {})
                    .get("equal_variance_passed")
                    is False
                )
            )
        ]
        if assumption_warnings:
            segments = [r.segment for r in assumption_warnings]
            recommendations.append(
                f"ASSUMPTION WARNING: {len(assumption_warnings)} segment(s) violate normality/variance checks ({', '.join(segments)}). "
                "Interpret mean-based frequentist tests with caution and compare against robust alternatives."
            )

        outlier_sensitive = [
            r
            for r in results
            if bool(
                r.diagnostics.get("experiment_quality", {})
                .get("outlier_sensitivity", {})
                .get("is_sensitive", False)
            )
        ]
        if outlier_sensitive:
            segments = [r.segment for r in outlier_sensitive]
            recommendations.append(
                f"OUTLIER SENSITIVITY: {len(outlier_sensitive)} segment(s) show material trimmed/winsorized deltas ({', '.join(segments)}). "
                "Review outliers and report robust effect estimates alongside raw means."
            )

        if any(r.multiple_testing_applied for r in results):
            recommendations.append(
                "MULTIPLE TESTING: Segment-level frequentist p-values were corrected using Benjamini-Hochberg (FDR). "
                "Prefer adjusted significance when deciding rollouts."
            )

        significant_results = [r for r in results if r.is_significant]
        inadequate_samples = [r for r in results if not r.is_sample_adequate]

        if significant_results:
            positive_effects = [r for r in significant_results if r.effect_size > 0]
            negative_effects = [r for r in significant_results if r.effect_size < 0]

            if positive_effects:
                segments = [r.segment for r in positive_effects]
                recommendations.append(
                    f"POSITIVE IMPACT: Treatment shows significant positive effect in {len(positive_effects)} segment(s): {', '.join(segments)}. "
                    "Consider rolling out treatment to these segments."
                )

            if negative_effects:
                segments = [r.segment for r in negative_effects]
                recommendations.append(
                    f"NEGATIVE IMPACT: Treatment shows significant negative effect in {len(negative_effects)} segment(s): {', '.join(segments)}. "
                    "Investigate root cause before broader rollout."
                )
        else:
            recommendations.append(
                "NO SIGNIFICANT EFFECTS: No segments showed statistically significant differences. "
                "Consider extending experiment duration or increasing sample size."
            )

        if inadequate_samples:
            segments = [
                f"{r.segment} (needs ~{r.required_sample_size} per group)"
                for r in inadequate_samples[:3]
            ]
            recommendations.append(
                f"SAMPLE SIZE: {len(inadequate_samples)} segment(s) have insufficient statistical power. "
                f"Examples: {'; '.join(segments)}"
            )

        imbalanced = [
            r
            for r in results
            if (r.control_size > 0 and r.treatment_size / r.control_size > 2)
            or (r.treatment_size > 0 and r.control_size / r.treatment_size > 2)
        ]
        if imbalanced:
            recommendations.append(
                f"GROUP IMBALANCE: {len(imbalanced)} segment(s) have imbalanced treatment/control ratios. "
                "Consider using stratified randomization in future experiments."
            )

        return recommendations

"""
Summary and recommendation builder for A/B test result sets.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .models import ABTestResult


class ABTestSummaryBuilder:
    """Build summary payloads and recommendations from segment-level results."""

    def generate_summary(self, results: List[ABTestResult]) -> Dict[str, Any]:
        """Generate comprehensive summary used by UI and agent output."""
        if not results:
            return {"error": "No results to summarize"}

        aa_failed_segments = [r for r in results if not r.aa_test_passed]
        bootstrapped_segments = [r for r in results if r.bootstrapping_applied]

        t_significant_results = [r for r in results if r.is_significant]
        prop_significant_results = [r for r in results if r.proportion_is_significant]
        bayesian_significant_results = [r for r in results if r.bayesian_is_significant]

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

        return {
            "total_segments_analyzed": len(results),
            "aa_test_passed_segments": len(results) - len(aa_failed_segments),
            "aa_test_failed_segments": len(aa_failed_segments),
            "bootstrapped_segments": len(bootstrapped_segments),
            "aa_failed_segment_names": [r.segment for r in aa_failed_segments],
            "did_avg_effect": avg_did_effect,
            "did_total_effect": total_did_effect,
            "t_test_significant_segments": len(t_significant_results),
            "t_test_significance_rate": len(t_significant_results) / len(results) if results else 0.0,
            "t_test_avg_effect": avg_t_test_effect,
            "t_test_total_effect": t_test_total_effect,
            "t_test_effect_calculation": f"{avg_t_test_effect:.4f} × {total_treatment_in_t_significant} = {t_test_total_effect:.2f}",
            "prop_test_significant_segments": len(prop_significant_results),
            "prop_test_significance_rate": len(prop_significant_results) / len(results) if results else 0.0,
            "prop_test_avg_effect": avg_prop_effect,
            "prop_test_total_effect": prop_total_effect,
            "prop_test_effect_calculation": f"{avg_prop_effect:.4f} × {total_treatment_in_prop_significant} = {prop_total_effect:.2f}",
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
                    "power": r.power,
                    "adequate_sample": r.is_sample_adequate,
                    "ci_lower": r.confidence_interval[0],
                    "ci_upper": r.confidence_interval[1],
                    "treatment_prop": r.treatment_proportion,
                    "control_prop": r.control_proportion,
                    "prop_diff": r.proportion_diff,
                    "prop_p_value": r.proportion_p_value,
                    "prop_significant": r.proportion_is_significant,
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

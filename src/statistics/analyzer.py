"""
A/B Test Statistical Analyzer

Facade orchestrating A/B analysis with a modular architecture:
- data_manager: dataframe lifecycle and schema inference
- statsmodels_engine: inferential statistics and effect estimation
- summary_builder: report aggregation and recommendations
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .data_manager import ABTestDataManager
from .models import AATestResult, ABTestResult
from .statsmodels_engine import StatsmodelsABTestEngine
from .summary_builder import ABTestSummaryBuilder


class ABTestAnalyzer:
    """
    Comprehensive A/B Test Analyzer.

    Public API is backward compatible with the previous implementation,
    while internals are split into dedicated components.
    """

    def __init__(self, significance_level: float = 0.05, power_threshold: float = 0.8):
        self.significance_level = significance_level
        self.power_threshold = power_threshold

        self.data_manager = ABTestDataManager()
        self.stats_engine = StatsmodelsABTestEngine(
            significance_level=significance_level,
            power_threshold=power_threshold,
        )
        self.summary_builder = ABTestSummaryBuilder()

    # ---------------------------------------------------------------------
    # Compatibility properties
    # ---------------------------------------------------------------------
    @property
    def df(self) -> Optional[pd.DataFrame]:
        return self.data_manager.df

    @df.setter
    def df(self, value: Optional[pd.DataFrame]) -> None:
        self.data_manager.df = value

    @property
    def column_mapping(self) -> Dict[str, str]:
        return self.data_manager.column_mapping

    @column_mapping.setter
    def column_mapping(self, value: Dict[str, str]) -> None:
        self.data_manager.column_mapping = value

    @property
    def treatment_label(self) -> Optional[Any]:
        return self.data_manager.treatment_label

    @treatment_label.setter
    def treatment_label(self, value: Optional[Any]) -> None:
        self.data_manager.treatment_label = value

    @property
    def control_label(self) -> Optional[Any]:
        return self.data_manager.control_label

    @control_label.setter
    def control_label(self, value: Optional[Any]) -> None:
        self.data_manager.control_label = value

    # ---------------------------------------------------------------------
    # Data and schema API
    # ---------------------------------------------------------------------
    def load_data(self, filepath: str) -> Dict[str, Any]:
        return self.data_manager.load_data(filepath)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.data_manager.set_dataframe(df)

    def detect_columns(self) -> Dict[str, List[str]]:
        return self.data_manager.detect_columns()

    def set_column_mapping(self, mapping: Dict[str, str]) -> None:
        self.data_manager.set_column_mapping(mapping)

    def get_group_values(self) -> Dict[str, List[Any]]:
        return self.data_manager.get_group_values()

    def set_group_labels(self, treatment_label: Any, control_label: Any) -> None:
        self.data_manager.set_group_labels(treatment_label, control_label)

    def auto_configure(self) -> Dict[str, Any]:
        return self.data_manager.auto_configure()

    def query_data(self, query: str) -> pd.DataFrame:
        return self.data_manager.query_data(query)

    def get_data_summary(self) -> Dict[str, Any]:
        return self.data_manager.get_data_summary()

    def get_segment_distribution(self) -> Dict[str, Any]:
        return self.data_manager.get_segment_distribution()

    # ---------------------------------------------------------------------
    # Statistical engine API (delegated)
    # ---------------------------------------------------------------------
    def calculate_cohens_d(self, treatment_data: np.ndarray, control_data: np.ndarray) -> float:
        return self.stats_engine.calculate_cohens_d(treatment_data, control_data)

    def calculate_power(self, effect_size: float, n_treatment: int, n_control: int) -> float:
        return self.stats_engine.calculate_power(effect_size, n_treatment, n_control)

    def calculate_required_sample_size(self, effect_size: float, ratio: float = 1.0) -> int:
        return self.stats_engine.calculate_required_sample_size(effect_size, ratio)

    def run_aa_test(
        self,
        treatment_pre: np.ndarray,
        control_pre: np.ndarray,
        segment_name: str = "Overall",
    ) -> AATestResult:
        return self.stats_engine.run_aa_test(treatment_pre, control_pre, segment_name)

    def bootstrap_balanced_control(
        self,
        treatment_pre: np.ndarray,
        control_df: pd.DataFrame,
        pre_col: str,
        max_iterations: int = 1000,
        target_p_value: float = 0.10,
    ):
        return self.stats_engine.bootstrap_balanced_control(
            treatment_pre,
            control_df,
            pre_col,
            max_iterations=max_iterations,
            target_p_value=target_p_value,
        )

    def run_proportion_test(self, treatment_data: np.ndarray, control_data: np.ndarray) -> Dict[str, float]:
        return self.stats_engine.run_proportion_test(treatment_data, control_data)

    def run_bayesian_test(
        self,
        treatment_post: np.ndarray,
        control_post: np.ndarray,
        treatment_pre: np.ndarray | None = None,
        control_pre: np.ndarray | None = None,
        n_samples: int = 10000,
    ) -> Dict[str, Any]:
        return self.stats_engine.run_bayesian_test(
            treatment_post=treatment_post,
            control_post=control_post,
            treatment_pre=treatment_pre,
            control_pre=control_pre,
            n_samples=n_samples,
        )

    # ---------------------------------------------------------------------
    # Core analysis orchestration
    # ---------------------------------------------------------------------
    def run_ab_test(self, segment_filter: Optional[str] = None) -> ABTestResult:
        """
        Run A/B test for overall data or a specific segment.
        """
        if self.df is None:
            raise ValueError("No data loaded")

        required_cols = ["group", "effect_value"]
        for col in required_cols:
            if col not in self.column_mapping:
                raise ValueError(f"Column mapping for '{col}' not set")

        if self.treatment_label is None or self.control_label is None:
            raise ValueError("Treatment/control labels not set")

        group_col = self.column_mapping["group"]
        effect_col = self.column_mapping["effect_value"]
        pre_effect_col = self.column_mapping.get("pre_effect")
        post_effect_col = self.column_mapping.get("post_effect", effect_col)

        df_filtered = self.df.copy()
        segment_name = "Overall"

        if segment_filter and "segment" in self.column_mapping:
            segment_col = self.column_mapping["segment"]
            df_filtered = df_filtered[df_filtered[segment_col] == segment_filter]
            segment_name = str(segment_filter)

        treatment_df = df_filtered[df_filtered[group_col] == self.treatment_label]
        control_df = df_filtered[df_filtered[group_col] == self.control_label]

        treatment_post_series = treatment_df[post_effect_col].dropna()
        control_post_series = control_df[post_effect_col].dropna()

        if len(treatment_post_series) < 2 or len(control_post_series) < 2:
            raise ValueError(
                f"Insufficient data for segment '{segment_name}': "
                f"Treatment n={len(treatment_post_series)}, Control n={len(control_post_series)}"
            )

        aa_test_passed = True
        aa_p_value = 1.0
        bootstrapping_applied = False
        original_control_size = len(control_post_series)
        treatment_pre_mean = 0.0
        control_pre_mean = 0.0

        treatment_pre_aligned = None
        control_pre_aligned = None
        treatment_post_aligned = treatment_post_series.to_numpy()
        control_post_aligned = control_post_series.to_numpy()

        has_pre_effect = pre_effect_col is not None and pre_effect_col in df_filtered.columns

        if has_pre_effect:
            treatment_aligned_df = treatment_df.dropna(subset=[pre_effect_col, post_effect_col])
            control_aligned_df = control_df.dropna(subset=[pre_effect_col, post_effect_col])

            treatment_pre = treatment_aligned_df[pre_effect_col].to_numpy()
            control_pre = control_aligned_df[pre_effect_col].to_numpy()

            if len(treatment_pre) >= 2 and len(control_pre) >= 2:
                aa_result = self.run_aa_test(treatment_pre, control_pre, segment_name)
                aa_test_passed = aa_result.is_balanced
                aa_p_value = aa_result.aa_p_value
                treatment_pre_mean = aa_result.treatment_pre_mean
                control_pre_mean = aa_result.control_pre_mean

                control_df_for_analysis = control_aligned_df
                if not aa_test_passed and len(control_aligned_df) > 0:
                    control_df_for_analysis, aa_result = self.bootstrap_balanced_control(
                        treatment_pre=treatment_pre,
                        control_df=control_aligned_df,
                        pre_col=pre_effect_col,
                    )
                    aa_test_passed = aa_result.is_balanced
                    aa_p_value = aa_result.aa_p_value
                    bootstrapping_applied = aa_result.bootstrapping_applied
                    original_control_size = aa_result.original_control_size
                    control_pre_mean = aa_result.control_pre_mean

                treatment_pre_aligned = treatment_aligned_df[pre_effect_col].to_numpy()
                treatment_post_aligned = treatment_aligned_df[post_effect_col].to_numpy()
                control_pre_aligned = control_df_for_analysis[pre_effect_col].to_numpy()
                control_post_aligned = control_df_for_analysis[post_effect_col].to_numpy()

                if len(treatment_pre_aligned) > 0:
                    treatment_pre_mean = float(np.mean(treatment_pre_aligned))
                if len(control_pre_aligned) > 0:
                    control_pre_mean = float(np.mean(control_pre_aligned))

        # Fallback to post-only arrays if alignment collapsed due to missing data
        if len(treatment_post_aligned) < 2 or len(control_post_aligned) < 2:
            treatment_post_aligned = treatment_post_series.to_numpy()
            control_post_aligned = control_post_series.to_numpy()
            treatment_pre_aligned = None
            control_pre_aligned = None
            has_pre_effect = False

        effect_metrics = self.stats_engine.estimate_treatment_effect(
            treatment_data=treatment_post_aligned,
            control_data=control_post_aligned,
        )

        treatment_mean = effect_metrics["treatment_mean"]
        control_mean = effect_metrics["control_mean"]
        effect_size = effect_metrics["effect_size"]
        t_stat = effect_metrics["t_statistic"]
        p_value = effect_metrics["p_value"]
        confidence_interval = effect_metrics["confidence_interval"]

        treatment_post_mean = treatment_mean
        control_post_mean = control_mean

        cohens_d = self.calculate_cohens_d(treatment_post_aligned, control_post_aligned)
        power = self.calculate_power(cohens_d, len(treatment_post_aligned), len(control_post_aligned))
        required_n = self.calculate_required_sample_size(
            cohens_d,
            ratio=(len(control_post_aligned) / len(treatment_post_aligned))
            if len(treatment_post_aligned) > 0
            else 1.0,
        )

        is_significant = p_value < self.significance_level

        if has_pre_effect and treatment_pre_aligned is not None and control_pre_aligned is not None:
            did_metrics = self.stats_engine.estimate_did_effect(
                treatment_pre=treatment_pre_aligned,
                treatment_post=treatment_post_aligned,
                control_pre=control_pre_aligned,
                control_post=control_post_aligned,
            )
            did_treatment_change = did_metrics["treatment_change"]
            did_control_change = did_metrics["control_change"]
            did_effect = did_metrics["did_effect"]
        else:
            did_treatment_change = 0.0
            did_control_change = 0.0
            did_effect = effect_size

        prop_results = self.run_proportion_test(treatment_post_aligned, control_post_aligned)
        proportion_diff = prop_results["proportion_diff"]
        proportion_is_significant = prop_results["p_value"] < self.significance_level

        if proportion_is_significant and proportion_diff > 0:
            proportion_effect_per_customer = proportion_diff * control_mean
            proportion_effect = proportion_effect_per_customer * len(treatment_post_aligned)
        else:
            proportion_effect_per_customer = 0.0
            proportion_effect = 0.0

        t_test_total_effect = effect_size * len(treatment_post_aligned) if is_significant else 0.0
        total_effect = t_test_total_effect + proportion_effect
        total_effect_per_customer = (
            (effect_size if is_significant else 0.0) + proportion_effect_per_customer
        )

        bayesian_results = self.run_bayesian_test(
            treatment_post=treatment_post_aligned,
            control_post=control_post_aligned,
            treatment_pre=treatment_pre_aligned,
            control_pre=control_pre_aligned,
        )
        bayesian_is_significant = (
            bayesian_results["prob_treatment_better"] > 0.95
            or bayesian_results["prob_treatment_better"] < 0.05
        )

        return ABTestResult(
            segment=segment_name,
            treatment_size=len(treatment_post_aligned),
            control_size=len(control_post_aligned),
            treatment_pre_mean=treatment_pre_mean,
            treatment_post_mean=treatment_post_mean,
            control_pre_mean=control_pre_mean,
            control_post_mean=control_post_mean,
            treatment_mean=treatment_mean,
            control_mean=control_mean,
            effect_size=effect_size,
            cohens_d=cohens_d,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=confidence_interval,
            power=power,
            required_sample_size=required_n,
            is_sample_adequate=power >= self.power_threshold,
            did_treatment_change=did_treatment_change,
            did_control_change=did_control_change,
            did_effect=did_effect,
            aa_test_passed=aa_test_passed,
            aa_p_value=aa_p_value,
            bootstrapping_applied=bootstrapping_applied,
            original_control_size=original_control_size,
            treatment_proportion=prop_results["treatment_proportion"],
            control_proportion=prop_results["control_proportion"],
            proportion_diff=proportion_diff,
            proportion_z_stat=prop_results["z_stat"],
            proportion_p_value=prop_results["p_value"],
            proportion_is_significant=proportion_is_significant,
            proportion_effect=proportion_effect,
            proportion_effect_per_customer=proportion_effect_per_customer,
            total_effect=total_effect,
            total_effect_per_customer=total_effect_per_customer,
            bayesian_prob_treatment_better=bayesian_results["prob_treatment_better"],
            bayesian_expected_loss_treatment=bayesian_results["expected_loss_treatment"],
            bayesian_expected_loss_control=bayesian_results["expected_loss_control"],
            bayesian_credible_interval=bayesian_results["credible_interval"],
            bayesian_relative_uplift=bayesian_results["relative_uplift"],
            bayesian_is_significant=bayesian_is_significant,
            bayesian_total_effect=bayesian_results["total_effect"],
            bayesian_total_effect_per_customer=(
                bayesian_results["total_effect"] / len(treatment_post_aligned)
                if len(treatment_post_aligned) > 0
                else 0.0
            ),
        )

    def run_segmented_analysis(self) -> List[ABTestResult]:
        """Run A/B tests for all segments or overall if no segment column."""
        if "segment" not in self.column_mapping:
            return [self.run_ab_test()]

        segment_col = self.column_mapping["segment"]
        segments = self.df[segment_col].dropna().unique()

        results: List[ABTestResult] = []
        for segment in segments:
            try:
                results.append(self.run_ab_test(segment_filter=segment))
            except ValueError as error:
                print(f"Skipping segment '{segment}': {error}")

        return results

    def generate_summary(self, results: List[ABTestResult]) -> Dict[str, Any]:
        """Generate aggregate summary and recommendations."""
        return self.summary_builder.generate_summary(results)

    def _generate_recommendations(self, results: List[ABTestResult]) -> List[str]:
        """Backward-compatible recommendation helper."""
        return self.summary_builder._generate_recommendations(results)

"""
A/B Test Statistical Analyzer

Facade orchestrating A/B analysis with a modular architecture:
- data_manager: dataframe lifecycle and schema inference
- statsmodels_engine: inferential statistics and effect estimation
- summary_builder: report aggregation and recommendations
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

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

    def run_proportion_test(self, treatment_data: np.ndarray, control_data: np.ndarray) -> Dict[str, Any]:
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

    @staticmethod
    def _parse_covariate_columns(raw_covariates: Any) -> List[str]:
        """Normalize optional covariate mapping to a deduplicated list of column names."""
        if raw_covariates is None:
            return []
        if isinstance(raw_covariates, str):
            candidates = [part.strip() for part in raw_covariates.split(",") if part.strip()]
        elif isinstance(raw_covariates, Sequence):
            candidates = [str(value).strip() for value in raw_covariates if str(value).strip()]
        else:
            return []

        deduped: List[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _resolve_covariate_columns(
        self,
        *,
        df: pd.DataFrame,
        group_col: str,
        post_effect_col: str,
        pre_effect_col: Optional[str],
    ) -> List[str]:
        """Collect numeric covariates from mapping and pre-period metrics when available."""
        covariates = self._parse_covariate_columns(self.column_mapping.get("covariates"))
        if pre_effect_col is not None and pre_effect_col in df.columns and pre_effect_col not in covariates:
            covariates.append(pre_effect_col)

        excluded = {group_col, post_effect_col}
        segment_col = self.column_mapping.get("segment")
        if segment_col:
            excluded.add(segment_col)

        resolved: List[str] = []
        for column in covariates:
            if column in excluded:
                continue
            if column not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
            if column not in resolved:
                resolved.append(column)
        return resolved

    @staticmethod
    def _coerce_int(value: Any, *, default: int, minimum: int = 1) -> int:
        """Best-effort integer coercion with lower-bound clamping."""
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = default
        return max(numeric, minimum)

    @staticmethod
    def _coerce_float(
        value: Any,
        *,
        default: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        """Best-effort float coercion with bounded range."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        return float(min(max(numeric, minimum), maximum))

    def _resolve_sequential_config(
        self,
        sequential_config: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve optional sequential testing configuration.

        Sequential mode is opt-in through either:
        - run_ab_test(..., sequential_config={...})
        - self.column_mapping["sequential"] = {...}
        """
        raw_config: Any = (
            sequential_config
            if sequential_config is not None
            else self.column_mapping.get("sequential")
        )

        if raw_config in (None, False):
            return {"enabled": False}

        if raw_config is True:
            raw_dict: Dict[str, Any] = {}
        elif isinstance(raw_config, Mapping):
            raw_dict = dict(raw_config)
        else:
            return {"enabled": False}

        enabled = bool(raw_dict.get("enabled", True))
        if not enabled:
            return {"enabled": False}

        look_index = self._coerce_int(
            raw_dict.get(
                "look_index",
                raw_dict.get("current_look", raw_dict.get("interim_look", 1)),
            ),
            default=1,
            minimum=1,
        )
        max_looks = self._coerce_int(
            raw_dict.get(
                "max_looks",
                raw_dict.get("total_looks", raw_dict.get("planned_looks", look_index)),
            ),
            default=look_index,
            minimum=1,
        )
        look_index = min(look_index, max_looks)

        method = str(
            raw_dict.get(
                "spending_method",
                raw_dict.get("method", self.stats_engine.DEFAULT_SEQUENTIAL_METHOD),
            )
        ).strip().lower()

        return {
            "enabled": True,
            "look_index": look_index,
            "max_looks": max_looks,
            "spending_method": method,
            "futility_min_information_fraction": self._coerce_float(
                raw_dict.get(
                    "futility_min_information_fraction",
                    self.stats_engine.DEFAULT_FUTILITY_MIN_INFORMATION_FRACTION,
                ),
                default=self.stats_engine.DEFAULT_FUTILITY_MIN_INFORMATION_FRACTION,
                minimum=0.0,
                maximum=1.0,
            ),
            "futility_p_value_threshold": self._coerce_float(
                raw_dict.get(
                    "futility_p_value_threshold",
                    self.stats_engine.DEFAULT_FUTILITY_P_VALUE_THRESHOLD,
                ),
                default=self.stats_engine.DEFAULT_FUTILITY_P_VALUE_THRESHOLD,
                minimum=0.0,
                maximum=1.0,
            ),
        }

    # ---------------------------------------------------------------------
    # Core analysis orchestration
    # ---------------------------------------------------------------------
    def run_ab_test(
        self,
        segment_filter: Optional[str] = None,
        sequential_config: Optional[Mapping[str, Any]] = None,
    ) -> ABTestResult:
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
        treatment_analysis_df = treatment_df.dropna(subset=[post_effect_col]).copy()
        control_analysis_df = control_df.dropna(subset=[post_effect_col]).copy()

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
                treatment_analysis_df = treatment_aligned_df.copy()
                control_analysis_df = control_df_for_analysis.copy()

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
            treatment_analysis_df = treatment_df.dropna(subset=[post_effect_col]).copy()
            control_analysis_df = control_df.dropna(subset=[post_effect_col]).copy()

        metric_type_option = str(self.column_mapping.get("metric_type", "auto"))
        count_model_option = str(self.column_mapping.get("count_model", "auto"))
        heavy_tail_strategy_option = str(
            self.column_mapping.get("heavy_tail_strategy", "robust")
        )

        covariate_columns = self._resolve_covariate_columns(
            df=df_filtered,
            group_col=group_col,
            post_effect_col=post_effect_col,
            pre_effect_col=pre_effect_col,
        )
        treatment_model_df = treatment_analysis_df.copy()
        control_model_df = control_analysis_df.copy()
        if covariate_columns:
            required_model_columns = [post_effect_col, *covariate_columns]
            treatment_model_df = treatment_model_df.dropna(subset=required_model_columns)
            control_model_df = control_model_df.dropna(subset=required_model_columns)

            if len(treatment_model_df) >= 2 and len(control_model_df) >= 2:
                treatment_post_aligned = treatment_model_df[post_effect_col].to_numpy()
                control_post_aligned = control_model_df[post_effect_col].to_numpy()

                if (
                    has_pre_effect
                    and pre_effect_col is not None
                    and pre_effect_col in treatment_model_df.columns
                    and pre_effect_col in control_model_df.columns
                ):
                    treatment_pre_aligned = treatment_model_df[pre_effect_col].to_numpy()
                    control_pre_aligned = control_model_df[pre_effect_col].to_numpy()
                    treatment_pre_mean = float(np.mean(treatment_pre_aligned))
                    control_pre_mean = float(np.mean(control_pre_aligned))

        effect_metrics = self.stats_engine.estimate_treatment_effect(
            treatment_data=treatment_post_aligned,
            control_data=control_post_aligned,
            metric_type=metric_type_option,
            count_model=count_model_option,
            heavy_tail_strategy=heavy_tail_strategy_option,
        )

        treatment_mean = effect_metrics["treatment_mean"]
        control_mean = effect_metrics["control_mean"]
        effect_size = effect_metrics["effect_size"]
        t_stat = effect_metrics["t_statistic"]
        p_value = effect_metrics["p_value"]
        confidence_interval = effect_metrics["confidence_interval"]
        metric_type_selected = effect_metrics.get("metric_type", "continuous")
        model_type = effect_metrics.get("model_type", "ols_hc3")
        model_effect = effect_metrics.get("model_effect", effect_size)
        model_confidence_interval = effect_metrics.get(
            "model_confidence_interval",
            confidence_interval,
        )
        model_effect_scale = effect_metrics.get("model_effect_scale", "mean_difference")
        model_effect_exponentiated = effect_metrics.get("model_effect_exponentiated", 1.0)
        t_test_diagnostics = effect_metrics.get("diagnostics", {})
        inference_guardrail_triggered = bool(t_test_diagnostics.get("guardrail_triggered", False))
        inference_blocks_significance = bool(t_test_diagnostics.get("blocks_significance", False))

        covariate_adjustment_applied = False
        covariates_used: List[str] = []
        covariate_adjusted_effect = effect_size
        covariate_adjusted_p_value = p_value
        covariate_adjusted_confidence_interval = confidence_interval
        covariate_adjusted_model_type = "none"
        covariate_adjusted_effect_scale = model_effect_scale
        covariate_adjusted_effect_exponentiated = model_effect_exponentiated
        covariate_adjusted_diagnostics: Dict[str, Any] = {}

        if covariate_columns and len(treatment_model_df) >= 2 and len(control_model_df) >= 2:
            adjusted_effect_metrics = self.stats_engine.estimate_treatment_effect(
                treatment_data=treatment_model_df[post_effect_col].to_numpy(),
                control_data=control_model_df[post_effect_col].to_numpy(),
                metric_type=metric_type_option,
                treatment_covariates=treatment_model_df[covariate_columns],
                control_covariates=control_model_df[covariate_columns],
                covariate_names=covariate_columns,
                count_model=count_model_option,
                heavy_tail_strategy=heavy_tail_strategy_option,
            )

            covariate_adjustment_applied = bool(
                adjusted_effect_metrics.get("covariate_adjusted", False)
            )
            covariates_used = list(adjusted_effect_metrics.get("covariates_used", covariate_columns))
            covariate_adjusted_effect = adjusted_effect_metrics.get(
                "model_effect",
                adjusted_effect_metrics.get("effect_size", effect_size),
            )
            covariate_adjusted_p_value = adjusted_effect_metrics.get("p_value", p_value)
            covariate_adjusted_confidence_interval = adjusted_effect_metrics.get(
                "model_confidence_interval",
                adjusted_effect_metrics.get("confidence_interval", confidence_interval),
            )
            covariate_adjusted_model_type = adjusted_effect_metrics.get("model_type", "none")
            covariate_adjusted_effect_scale = adjusted_effect_metrics.get(
                "model_effect_scale",
                model_effect_scale,
            )
            covariate_adjusted_effect_exponentiated = adjusted_effect_metrics.get(
                "model_effect_exponentiated",
                1.0,
            )
            covariate_adjusted_diagnostics = adjusted_effect_metrics.get("diagnostics", {})

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

        is_significant = p_value < self.significance_level and not inference_blocks_significance

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
        proportion_diagnostics = prop_results.get("diagnostics", {})
        proportion_guardrail_triggered = bool(
            proportion_diagnostics.get("guardrail_triggered", False)
        )
        proportion_blocks_significance = bool(
            proportion_diagnostics.get("blocks_significance", False)
        )
        proportion_is_significant = (
            prop_results["p_value"] < self.significance_level and not proportion_blocks_significance
        )

        srm_diagnostics = self.stats_engine.run_srm_diagnostics(
            treatment_size=len(treatment_post_aligned),
            control_size=len(control_post_aligned),
        )
        assumption_diagnostics = self.stats_engine.run_assumption_diagnostics(
            treatment_data=treatment_post_aligned,
            control_data=control_post_aligned,
        )
        outlier_sensitivity = self.stats_engine.run_outlier_sensitivity(
            treatment_data=treatment_post_aligned,
            control_data=control_post_aligned,
            baseline_effect=effect_size,
        )

        diagnostics = {
            "frequentist": {
                "t_test": t_test_diagnostics,
                "proportion_test": proportion_diagnostics,
                "model_inference": {
                    "metric_type": metric_type_selected,
                    "model_type": model_type,
                    "effect_scale": model_effect_scale,
                    "model_effect": model_effect,
                    "model_effect_exponentiated": model_effect_exponentiated,
                    "covariate_adjustment_applied": covariate_adjustment_applied,
                    "covariates_used": covariates_used,
                    "covariate_adjusted_model_type": covariate_adjusted_model_type,
                    "covariate_adjusted_effect_scale": covariate_adjusted_effect_scale,
                },
                "covariate_adjusted": covariate_adjusted_diagnostics,
            },
            "experiment_quality": {
                "srm": srm_diagnostics,
                "assumptions": assumption_diagnostics,
                "outlier_sensitivity": outlier_sensitivity,
            },
        }

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

        sequential_settings = self._resolve_sequential_config(sequential_config)
        if sequential_settings.get("enabled", False):
            sequential_results = self.stats_engine.evaluate_sequential_decision(
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=confidence_interval,
                look_index=sequential_settings["look_index"],
                max_looks=sequential_settings["max_looks"],
                method=sequential_settings["spending_method"],
                futility_min_information_fraction=sequential_settings[
                    "futility_min_information_fraction"
                ],
                futility_p_value_threshold=sequential_settings["futility_p_value_threshold"],
            )
        else:
            sequential_results = {
                "enabled": False,
                "method": "none",
                "look_index": 0,
                "max_looks": 0,
                "information_fraction": 0.0,
                "alpha_spent": 0.0,
                "stop_recommended": False,
                "decision": "not_requested",
                "rationale": "",
                "thresholds": {},
            }

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
            p_value_adjusted=p_value,
            is_significant_adjusted=is_significant,
            confidence_interval=confidence_interval,
            metric_type=metric_type_selected,
            model_type=model_type,
            model_effect=model_effect,
            model_confidence_interval=model_confidence_interval,
            model_effect_scale=model_effect_scale,
            model_effect_exponentiated=model_effect_exponentiated,
            covariate_adjustment_applied=covariate_adjustment_applied,
            covariates_used=covariates_used,
            covariate_adjusted_effect=covariate_adjusted_effect,
            covariate_adjusted_p_value=covariate_adjusted_p_value,
            covariate_adjusted_confidence_interval=covariate_adjusted_confidence_interval,
            covariate_adjusted_model_type=covariate_adjusted_model_type,
            covariate_adjusted_effect_scale=covariate_adjusted_effect_scale,
            covariate_adjusted_effect_exponentiated=covariate_adjusted_effect_exponentiated,
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
            proportion_p_value_adjusted=prop_results["p_value"],
            proportion_is_significant_adjusted=proportion_is_significant,
            multiple_testing_method="none",
            multiple_testing_applied=False,
            inference_guardrail_triggered=inference_guardrail_triggered,
            proportion_guardrail_triggered=proportion_guardrail_triggered,
            diagnostics=diagnostics,
            sequential_mode_enabled=bool(sequential_results["enabled"]),
            sequential_method=str(sequential_results["method"]),
            sequential_look_index=int(sequential_results["look_index"]),
            sequential_max_looks=int(sequential_results["max_looks"]),
            sequential_information_fraction=float(
                sequential_results["information_fraction"]
            ),
            sequential_alpha_spent=float(sequential_results["alpha_spent"]),
            sequential_stop_recommended=bool(sequential_results["stop_recommended"]),
            sequential_decision=str(sequential_results["decision"]),
            sequential_rationale=str(sequential_results["rationale"]),
            sequential_thresholds=dict(sequential_results["thresholds"]),
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

    def run_segmented_analysis(
        self,
        sequential_config: Optional[Mapping[str, Any]] = None,
    ) -> List[ABTestResult]:
        """Run A/B tests for all segments or overall if no segment column."""
        if "segment" not in self.column_mapping:
            return [self.run_ab_test(sequential_config=sequential_config)]

        segment_col = self.column_mapping["segment"]
        segments = self.df[segment_col].dropna().unique()

        results: List[ABTestResult] = []
        for segment in segments:
            try:
                results.append(
                    self.run_ab_test(
                        segment_filter=segment,
                        sequential_config=sequential_config,
                    )
                )
            except ValueError as error:
                print(f"Skipping segment '{segment}': {error}")

        if len(results) > 1:
            self._apply_multiple_testing_correction(results)

        return results

    @staticmethod
    def _sanitize_p_value(p_value: Any) -> float:
        """Clamp invalid p-values to 1.0 so correction is robust to upstream edge cases."""
        try:
            numeric = float(p_value)
        except (TypeError, ValueError):
            return 1.0
        if not np.isfinite(numeric):
            return 1.0
        if numeric < 0.0 or numeric > 1.0:
            return 1.0
        return numeric

    def _apply_multiple_testing_correction(self, results: List[ABTestResult]) -> None:
        """Apply BH/FDR correction across segment-level frequentist p-values."""
        p_values = np.array([self._sanitize_p_value(r.p_value) for r in results], dtype=float)
        prop_p_values = np.array(
            [self._sanitize_p_value(r.proportion_p_value) for r in results],
            dtype=float,
        )

        try:
            reject_main, adjusted_main, _, _ = multipletests(
                p_values,
                alpha=self.significance_level,
                method="fdr_bh",
            )
            reject_prop, adjusted_prop, _, _ = multipletests(
                prop_p_values,
                alpha=self.significance_level,
                method="fdr_bh",
            )
        except Exception:
            for result in results:
                result.p_value_adjusted = self._sanitize_p_value(result.p_value)
                result.is_significant_adjusted = result.is_significant
                result.proportion_p_value_adjusted = self._sanitize_p_value(result.proportion_p_value)
                result.proportion_is_significant_adjusted = result.proportion_is_significant
                result.multiple_testing_method = "none"
                result.multiple_testing_applied = False
            return

        for idx, result in enumerate(results):
            result.p_value_adjusted = float(adjusted_main[idx])
            result.proportion_p_value_adjusted = float(adjusted_prop[idx])
            result.multiple_testing_method = "fdr_bh"
            result.multiple_testing_applied = True

            t_test_blocks = bool(
                result.diagnostics.get("frequentist", {})
                .get("t_test", {})
                .get("blocks_significance", False)
            )
            prop_blocks = bool(
                result.diagnostics.get("frequentist", {})
                .get("proportion_test", {})
                .get("blocks_significance", False)
            )
            result.is_significant_adjusted = bool(reject_main[idx]) and not t_test_blocks
            result.proportion_is_significant_adjusted = bool(reject_prop[idx]) and not prop_blocks

    def generate_summary(self, results: List[ABTestResult]) -> Dict[str, Any]:
        """Generate aggregate summary and recommendations."""
        return self.summary_builder.generate_summary(results)

    def _generate_recommendations(self, results: List[ABTestResult]) -> List[str]:
        """Backward-compatible recommendation helper."""
        return self.summary_builder._generate_recommendations(results)

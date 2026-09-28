"""
A/B Test Statistical Analyzer

Facade orchestrating A/B analysis with a modular architecture:
- data_manager: dataframe lifecycle and schema inference
- statsmodels_engine: inferential statistics and effect estimation
- summary_builder: report aggregation and recommendations
- segment_preparer: group splitting, pre-period alignment, AA/bootstrap
- covariate_resolver: covariate parsing, alignment, adjusted models
- sequential_config: sequential testing configuration and decisions
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .covariate_resolver import CovariateResolver, CupedResult, apply_cuped
from .data_manager import ABTestDataManager
from .diagnostics import (
    DEFAULT_EXPECTED_TREATMENT_RATIO,
    describe_statistical_fallbacks,
    detect_duplicate_units,
    resolve_expected_treatment_ratio,
    validate_expected_treatment_ratio,
)
from .engine_helpers import apply_fdr_correction, combine_total_effect_per_customer
from .models import AATestResult, ABTestResult, ABTestSummary
from .power_analysis import (
    DEFAULT_TARGET_EFFECT_SIZE,
    calculate_minimum_detectable_effect,
    calculate_power,
    calculate_required_sample_size,
    is_sample_adequate,
)
from .segment_preparer import SegmentPreparer, _PreparedSegmentData
from .sequential_config import evaluate_sequential_decision
from .statsmodels_engine import StatsmodelsABTestEngine
from .summary_builder import ABTestSummaryBuilder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AnalysisSelection:
    """Resolved inputs for a single segment-level A/B analysis."""

    segment_name: str
    df_filtered: pd.DataFrame
    group_col: str
    effect_col: str
    pre_effect_col: Optional[str]
    post_effect_col: str
    metric_type_option: str
    count_model_option: str
    heavy_tail_strategy_option: str


class ABTestAnalyzer:
    """
    Comprehensive A/B Test Analyzer.

    Public API is backward compatible with the previous implementation,
    while internals are split into dedicated components.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        power_threshold: float = 0.8,
        seed: int = 42,
        expected_treatment_ratio: float = DEFAULT_EXPECTED_TREATMENT_RATIO,
        target_effect_size: float = DEFAULT_TARGET_EFFECT_SIZE,
    ):
        self.significance_level = significance_level
        self.power_threshold = power_threshold
        self.seed = seed
        # Designed treatment share for the SRM check (0.5 = 50/50). A
        # per-experiment override lives in column_mapping["expected_treatment_ratio"].
        self.expected_treatment_ratio = validate_expected_treatment_ratio(
            expected_treatment_ratio
        )
        # Standardized effect the sample must be able to detect to count as
        # adequate; override per experiment via column_mapping["target_effect_size"].
        self.target_effect_size = float(target_effect_size)

        self.data_manager = ABTestDataManager()
        self.stats_engine = StatsmodelsABTestEngine(
            significance_level=significance_level,
            power_threshold=power_threshold,
            seed=seed,
        )
        self.summary_builder = ABTestSummaryBuilder()
        self.segment_preparer = SegmentPreparer(
            stats_engine=self.stats_engine,
            significance_level=significance_level,
        )
        self.covariate_resolver = CovariateResolver(stats_engine=self.stats_engine)
        self.segment_failures: List[Dict[str, str]] = []

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
    def load_data(self, filepath: str, **kwargs: Any) -> Dict[str, Any]:
        """Load a CSV file.

        Extra keyword arguments (e.g. ``format=`` from the shared backend
        protocol) are accepted for interface compatibility and ignored by
        the pandas backend.
        """
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

    # ---------------------------------------------------------------------
    # Backward-compatible static helpers (delegated)
    # ---------------------------------------------------------------------
    @staticmethod
    def _parse_covariate_columns(raw_covariates: Any) -> List[str]:
        return CovariateResolver.parse_covariate_columns(raw_covariates)

    # ---------------------------------------------------------------------
    # Backward-compatible internal shims (tests may call these directly)
    # ---------------------------------------------------------------------
    def _prepare_segment_data(self, selection: _AnalysisSelection) -> _PreparedSegmentData:
        """Shim: delegates to segment_preparer.prepare()."""
        return self.segment_preparer.prepare(
            segment_name=selection.segment_name,
            df_filtered=selection.df_filtered,
            group_col=selection.group_col,
            pre_effect_col=selection.pre_effect_col,
            post_effect_col=selection.post_effect_col,
            treatment_label=self.treatment_label,
            control_label=self.control_label,
        )

    def _apply_covariate_alignment(
        self,
        selection: _AnalysisSelection,
        prepared: _PreparedSegmentData,
    ) -> tuple[List[str], pd.DataFrame, pd.DataFrame]:
        """Shim: delegates to covariate_resolver for resolution and alignment."""
        covariate_columns = self.covariate_resolver.resolve_covariate_columns(
            df=selection.df_filtered,
            group_col=selection.group_col,
            post_effect_col=selection.post_effect_col,
            pre_effect_col=selection.pre_effect_col,
            column_mapping=self.column_mapping,
        )
        (
            treatment_model_df,
            control_model_df,
            prepared.treatment_post_aligned,
            prepared.control_post_aligned,
            prepared.treatment_pre_aligned,
            prepared.control_pre_aligned,
            prepared.treatment_pre_mean,
            prepared.control_pre_mean,
        ) = self.covariate_resolver.apply_covariate_alignment(
            covariate_columns=covariate_columns,
            post_effect_col=selection.post_effect_col,
            pre_effect_col=selection.pre_effect_col,
            has_pre_effect=prepared.has_pre_effect,
            treatment_analysis_df=prepared.treatment_analysis_df,
            control_analysis_df=prepared.control_analysis_df,
            treatment_post_aligned=prepared.treatment_post_aligned,
            control_post_aligned=prepared.control_post_aligned,
            treatment_pre_aligned=prepared.treatment_pre_aligned,
            control_pre_aligned=prepared.control_pre_aligned,
        )
        return covariate_columns, treatment_model_df, control_model_df

    # ---------------------------------------------------------------------
    # Core analysis orchestration
    # ---------------------------------------------------------------------
    def _validate_analysis_state(self) -> None:
        """Verify required dataframe, mappings, and group labels are available."""
        if self.df is None:
            raise ValueError("No data loaded")

        for column_name in ("group", "effect_value"):
            if column_name not in self.column_mapping:
                raise ValueError(f"Column mapping for '{column_name}' not set")

        if self.treatment_label is None or self.control_label is None:
            raise ValueError("Treatment/control labels not set")

    def _resolve_analysis_selection(
        self,
        segment_filter: Optional[str],
    ) -> _AnalysisSelection:
        """Resolve the segment slice and analysis options for a single run."""
        if self.df is None:
            raise ValueError("No data loaded")

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

        return _AnalysisSelection(
            segment_name=segment_name,
            df_filtered=df_filtered,
            group_col=group_col,
            effect_col=effect_col,
            pre_effect_col=pre_effect_col,
            post_effect_col=post_effect_col,
            metric_type_option=str(self.column_mapping.get("metric_type", "auto")),
            count_model_option=str(self.column_mapping.get("count_model", "auto")),
            heavy_tail_strategy_option=str(
                self.column_mapping.get("heavy_tail_strategy", "robust")
            ),
        )

    @staticmethod
    def _build_experiment_diagnostics(
        *,
        metric_type_selected: str,
        model_type: str,
        model_effect_scale: str,
        model_effect: float,
        model_effect_exponentiated: float,
        covariate_adjustment_applied: bool,
        covariates_used: List[str],
        covariate_adjusted_model_type: str,
        covariate_adjusted_effect_scale: str,
        t_test_diagnostics: Dict[str, Any],
        proportion_diagnostics: Dict[str, Any],
        covariate_adjusted_diagnostics: Dict[str, Any],
        srm_diagnostics: Dict[str, Any],
        assumption_diagnostics: Dict[str, Any],
        outlier_sensitivity: Dict[str, Any],
        duplicate_units: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assemble the nested diagnostics payload returned on each segment result."""
        experiment_quality: Dict[str, Any] = {
            "srm": srm_diagnostics,
            "assumptions": assumption_diagnostics,
            "outlier_sensitivity": outlier_sensitivity,
        }
        if duplicate_units is not None:
            experiment_quality["duplicate_units"] = duplicate_units
        return {
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
            "experiment_quality": experiment_quality,
        }

    def run_ab_test(
        self,
        segment_filter: Optional[str] = None,
        sequential_config: Optional[Mapping[str, Any]] = None,
    ) -> ABTestResult:
        """
        Run A/B test for overall data or a specific segment.
        """
        self._validate_analysis_state()
        selection = self._resolve_analysis_selection(segment_filter)

        # --- Segment preparation (delegated) ---
        prepared = self.segment_preparer.prepare(
            segment_name=selection.segment_name,
            df_filtered=selection.df_filtered,
            group_col=selection.group_col,
            pre_effect_col=selection.pre_effect_col,
            post_effect_col=selection.post_effect_col,
            treatment_label=self.treatment_label,
            control_label=self.control_label,
        )

        # --- Covariate alignment (delegated) ---
        covariate_columns = self.covariate_resolver.resolve_covariate_columns(
            df=selection.df_filtered,
            group_col=selection.group_col,
            post_effect_col=selection.post_effect_col,
            pre_effect_col=selection.pre_effect_col,
            column_mapping=self.column_mapping,
        )
        (
            treatment_model_df,
            control_model_df,
            prepared.treatment_post_aligned,
            prepared.control_post_aligned,
            prepared.treatment_pre_aligned,
            prepared.control_pre_aligned,
            prepared.treatment_pre_mean,
            prepared.control_pre_mean,
        ) = self.covariate_resolver.apply_covariate_alignment(
            covariate_columns=covariate_columns,
            post_effect_col=selection.post_effect_col,
            pre_effect_col=selection.pre_effect_col,
            has_pre_effect=prepared.has_pre_effect,
            treatment_analysis_df=prepared.treatment_analysis_df,
            control_analysis_df=prepared.control_analysis_df,
            treatment_post_aligned=prepared.treatment_post_aligned,
            control_post_aligned=prepared.control_post_aligned,
            treatment_pre_aligned=prepared.treatment_pre_aligned,
            control_pre_aligned=prepared.control_pre_aligned,
        )

        statistical_warnings: List[str] = []
        if not prepared.aa_test_passed:
            statistical_warnings.append(
                f"Pre-period balance (AA) check failed (p={prepared.aa_p_value:.4f}): "
                "arms differed before the experiment. Prefer the CUPED, DiD or "
                "covariate-adjusted estimate over the raw post-period difference."
            )

        # Raw (unadjusted) post-period values. CUPED residuals are only valid
        # for the primary mean-difference estimate; the proportion test counts
        # non-zero values as conversions, and DiD / Bayesian / per-arm means
        # must stay on the metric's own scale (TODO.md #89).
        raw_treatment_post = np.asarray(prepared.treatment_post_aligned)
        raw_control_post = np.asarray(prepared.control_post_aligned)
        estimation_metric_type = selection.metric_type_option

        # --- CUPED variance reduction (opt-in via column_mapping["cuped"]) ---
        cuped_result: Optional[CupedResult] = None
        if bool(self.column_mapping.get("cuped", False)):
            raw_metric_type = self.stats_engine._infer_metric_type(
                np.concatenate([raw_treatment_post, raw_control_post]),
                selection.metric_type_option,
            )
            if raw_metric_type != "continuous":
                # Residualizing a binary/count/heavy-tail metric breaks the
                # family-specific model (GLM on non-0/1 or negative values).
                cuped_result = CupedResult(
                    applied=False,
                    theta=0.0,
                    variance_reduction=0.0,
                    treatment_adjusted=raw_treatment_post,
                    control_adjusted=raw_control_post,
                    reason=f"cuped_not_supported_for_{raw_metric_type}_metric",
                )
                statistical_warnings.append(
                    "CUPED was requested but skipped: it is only applied to continuous "
                    f"metrics and this metric was detected as '{raw_metric_type}'."
                )
            else:
                cuped_result = apply_cuped(
                    treatment_post=raw_treatment_post,
                    control_post=raw_control_post,
                    treatment_pre=prepared.treatment_pre_aligned,
                    control_pre=prepared.control_pre_aligned,
                )
                if cuped_result.applied:
                    prepared.treatment_post_aligned = cuped_result.treatment_adjusted
                    prepared.control_post_aligned = cuped_result.control_adjusted
                    # Residuals can look heavy-tailed; keep the family chosen
                    # on the raw metric.
                    estimation_metric_type = "continuous"

        # --- Primary effect estimation ---
        effect_metrics = self.stats_engine.estimate_treatment_effect(
            treatment_data=prepared.treatment_post_aligned,
            control_data=prepared.control_post_aligned,
            metric_type=estimation_metric_type,
            count_model=selection.count_model_option,
            heavy_tail_strategy=selection.heavy_tail_strategy_option,
        )

        effect_size = effect_metrics["effect_size"]
        p_value = effect_metrics["p_value"]
        confidence_interval = effect_metrics["confidence_interval"]
        metric_type_selected = effect_metrics.get("metric_type", "continuous")
        model_type = effect_metrics.get("model_type", "ols_hc3")
        model_effect = effect_metrics.get("model_effect", effect_size)
        model_effect_scale = effect_metrics.get("model_effect_scale", "mean_difference")
        model_effect_exponentiated = effect_metrics.get("model_effect_exponentiated", 1.0)
        t_test_diagnostics = effect_metrics.get("diagnostics", {})
        inference_guardrail_triggered = bool(t_test_diagnostics.get("guardrail_triggered", False))
        inference_blocks_significance = bool(t_test_diagnostics.get("blocks_significance", False))

        # --- Covariate-adjusted model (delegated) ---
        covariate_effects = self.covariate_resolver.run_covariate_adjusted_effect_model(
            post_effect_col=selection.post_effect_col,
            metric_type_option=selection.metric_type_option,
            count_model_option=selection.count_model_option,
            heavy_tail_strategy_option=selection.heavy_tail_strategy_option,
            covariate_columns=covariate_columns,
            treatment_model_df=treatment_model_df,
            control_model_df=control_model_df,
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=confidence_interval,
            model_effect_scale=model_effect_scale,
            model_effect_exponentiated=model_effect_exponentiated,
        )

        treatment_post_mean = float(np.mean(raw_treatment_post))
        control_post_mean = float(np.mean(raw_control_post))

        statistical_warnings.extend(
            describe_statistical_fallbacks(
                "Primary effect model", t_test_diagnostics, model_type=model_type
            )
        )

        n_treatment = len(prepared.treatment_post_aligned)
        n_control = len(prepared.control_post_aligned)
        cohens_d = self.calculate_cohens_d(
            prepared.treatment_post_aligned,
            prepared.control_post_aligned,
        )
        # Observed (post-hoc) power: informational only, never used for adequacy.
        power = calculate_power(
            effect_size=cohens_d,
            n_treatment=n_treatment,
            n_control=n_control,
            significance_level=self.significance_level,
            warnings_sink=statistical_warnings,
        )
        target_effect_size = self._resolve_target_effect_size()
        achieved_mde = calculate_minimum_detectable_effect(
            n_treatment=n_treatment,
            n_control=n_control,
            significance_level=self.significance_level,
            power_threshold=self.power_threshold,
            warnings_sink=statistical_warnings,
        )
        sample_adequate = is_sample_adequate(
            achieved_mde=achieved_mde,
            target_effect_size=target_effect_size,
        )
        required_n = calculate_required_sample_size(
            effect_size=target_effect_size,
            ratio=(n_control / n_treatment) if n_treatment > 0 else 1.0,
            power_threshold=self.power_threshold,
            significance_level=self.significance_level,
            warnings_sink=statistical_warnings,
        )

        is_significant = p_value < self.significance_level and not inference_blocks_significance

        if (
            prepared.has_pre_effect
            and prepared.treatment_pre_aligned is not None
            and prepared.control_pre_aligned is not None
        ):
            did_metrics = self.stats_engine.estimate_did_effect(
                treatment_pre=prepared.treatment_pre_aligned,
                treatment_post=raw_treatment_post,
                control_pre=prepared.control_pre_aligned,
                control_post=raw_control_post,
            )
            did_treatment_change = did_metrics["treatment_change"]
            did_control_change = did_metrics["control_change"]
            did_effect = did_metrics["did_effect"]
        else:
            did_treatment_change = 0.0
            did_control_change = 0.0
            did_effect = effect_size

        prop_results = self.run_proportion_test(raw_treatment_post, raw_control_post)
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
        statistical_warnings.extend(
            describe_statistical_fallbacks("Proportion test", proportion_diagnostics)
        )

        srm_diagnostics = self.stats_engine.run_srm_diagnostics(
            treatment_size=n_treatment,
            control_size=n_control,
            expected_treatment_ratio=resolve_expected_treatment_ratio(
                self.column_mapping, self.expected_treatment_ratio
            ),
        )
        srm_mismatch = bool(srm_diagnostics.get("is_sample_ratio_mismatch", False))
        if srm_mismatch:
            inference_guardrail_triggered = True
            inference_blocks_significance = True
            is_significant = False
            proportion_blocks_significance = True
            proportion_is_significant = False
        assumption_diagnostics = self.stats_engine.run_assumption_diagnostics(
            treatment_data=prepared.treatment_post_aligned,
            control_data=prepared.control_post_aligned,
        )
        outlier_sensitivity = self.stats_engine.run_outlier_sensitivity(
            treatment_data=prepared.treatment_post_aligned,
            control_data=prepared.control_post_aligned,
            baseline_effect=effect_size,
        )

        customer_col = self.data_manager.column_mapping.get("customer_id")
        duplicate_units = detect_duplicate_units(
            df=selection.df_filtered,
            customer_col=customer_col,
            group_col=selection.group_col,
        )

        diagnostics = self._build_experiment_diagnostics(
            metric_type_selected=metric_type_selected,
            model_type=model_type,
            model_effect_scale=model_effect_scale,
            model_effect=model_effect,
            model_effect_exponentiated=model_effect_exponentiated,
            covariate_adjustment_applied=covariate_effects["covariate_adjustment_applied"],
            covariates_used=covariate_effects["covariates_used"],
            covariate_adjusted_model_type=covariate_effects["covariate_adjusted_model_type"],
            covariate_adjusted_effect_scale=covariate_effects["covariate_adjusted_effect_scale"],
            t_test_diagnostics=t_test_diagnostics,
            proportion_diagnostics=proportion_diagnostics,
            covariate_adjusted_diagnostics=covariate_effects["covariate_adjusted_diagnostics"],
            srm_diagnostics=srm_diagnostics,
            assumption_diagnostics=assumption_diagnostics,
            outlier_sensitivity=outlier_sensitivity,
            duplicate_units=duplicate_units,
        )

        if proportion_is_significant and proportion_diff > 0:
            proportion_effect_per_customer = proportion_diff * control_post_mean
            proportion_effect = proportion_effect_per_customer * n_treatment
        else:
            proportion_effect_per_customer = 0.0
            proportion_effect = 0.0

        total_effect_per_customer = combine_total_effect_per_customer(
            effect_size=effect_size,
            is_significant=is_significant,
            proportion_effect_per_customer=proportion_effect_per_customer,
        )
        total_effect = total_effect_per_customer * n_treatment

        bayesian_results = self.run_bayesian_test(
            treatment_post=raw_treatment_post,
            control_post=raw_control_post,
            treatment_pre=prepared.treatment_pre_aligned,
            control_pre=prepared.control_pre_aligned,
        )
        bayesian_is_significant = (
            bayesian_results["prob_treatment_better"] > 0.95
            or bayesian_results["prob_treatment_better"] < 0.05
        )

        # --- Sequential testing (delegated) ---
        sequential_results = evaluate_sequential_decision(
            sequential_config=sequential_config,
            column_mapping=self.column_mapping,
            stats_engine=self.stats_engine,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
        )

        # When sequential mode is active, the alpha-spending boundary
        # (encoded in sequential_results["decision"]) is the canonical
        # significance call — not the fixed-sample alpha.
        if sequential_results["enabled"]:
            sequential_significant = sequential_results["decision"] in (
                "stop_efficacy",
                "final_accept",
            )
            is_significant = sequential_significant and not inference_blocks_significance

        return self._assemble_ab_test_result(
            selection=selection,
            prepared=prepared,
            effect_metrics=effect_metrics,
            covariate_effects=covariate_effects,
            prop_results=prop_results,
            bayesian_results=bayesian_results,
            sequential_results=sequential_results,
            diagnostics=diagnostics,
            cuped_result=cuped_result,
            treatment_post_mean=treatment_post_mean,
            control_post_mean=control_post_mean,
            cohens_d=cohens_d,
            is_significant=is_significant,
            inference_guardrail_triggered=inference_guardrail_triggered,
            proportion_is_significant=proportion_is_significant,
            proportion_guardrail_triggered=proportion_guardrail_triggered,
            proportion_diff=proportion_diff,
            proportion_effect=proportion_effect,
            proportion_effect_per_customer=proportion_effect_per_customer,
            total_effect=total_effect,
            total_effect_per_customer=total_effect_per_customer,
            bayesian_is_significant=bayesian_is_significant,
            did_treatment_change=did_treatment_change,
            did_control_change=did_control_change,
            did_effect=did_effect,
            power=power,
            required_n=required_n,
            achieved_mde=achieved_mde,
            is_sample_adequate=sample_adequate,
            target_effect_size=target_effect_size,
            statistical_warnings=statistical_warnings,
        )

    def _resolve_target_effect_size(self) -> float:
        raw: Any = self.column_mapping.get("target_effect_size")
        if raw in (None, ""):
            return abs(self.target_effect_size)
        try:
            value = abs(float(raw))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"target_effect_size must be a positive number; got {raw!r}"
            ) from error
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"target_effect_size must be a positive number; got {raw!r}")
        return value

    def _assemble_ab_test_result(
        self,
        *,
        selection: _AnalysisSelection,
        prepared: _PreparedSegmentData,
        effect_metrics: Dict[str, Any],
        covariate_effects: Dict[str, Any],
        prop_results: Dict[str, Any],
        bayesian_results: Dict[str, Any],
        sequential_results: Dict[str, Any],
        diagnostics: Dict[str, Any],
        cuped_result: Any,
        treatment_post_mean: float,
        control_post_mean: float,
        cohens_d: float,
        is_significant: bool,
        inference_guardrail_triggered: bool,
        proportion_is_significant: bool,
        proportion_guardrail_triggered: bool,
        proportion_diff: float,
        proportion_effect: float,
        proportion_effect_per_customer: float,
        total_effect: float,
        total_effect_per_customer: float,
        bayesian_is_significant: bool,
        did_treatment_change: float,
        did_control_change: float,
        did_effect: float,
        power: float,
        required_n: int,
        achieved_mde: float,
        is_sample_adequate: bool,
        target_effect_size: float,
        statistical_warnings: List[str],
    ) -> ABTestResult:
        """Pack the result of one segment-level analysis into ABTestResult.

        Pure code-motion: no logic lives here that did not previously
        live inline at the bottom of run_ab_test.
        """
        treatment_n = len(prepared.treatment_post_aligned)
        control_n = len(prepared.control_post_aligned)
        return ABTestResult(
            segment=selection.segment_name,
            treatment_size=treatment_n,
            control_size=control_n,
            treatment_pre_mean=prepared.treatment_pre_mean,
            treatment_post_mean=treatment_post_mean,
            control_pre_mean=prepared.control_pre_mean,
            control_post_mean=control_post_mean,
            treatment_mean=effect_metrics["treatment_mean"],
            control_mean=effect_metrics["control_mean"],
            effect_size=effect_metrics["effect_size"],
            cohens_d=cohens_d,
            t_statistic=effect_metrics["t_statistic"],
            p_value=effect_metrics["p_value"],
            is_significant=is_significant,
            p_value_adjusted=effect_metrics["p_value"],
            is_significant_adjusted=is_significant,
            confidence_interval=effect_metrics["confidence_interval"],
            metric_type=effect_metrics.get("metric_type", "continuous"),
            model_type=effect_metrics.get("model_type", "ols_hc3"),
            model_effect=effect_metrics.get("model_effect", effect_metrics["effect_size"]),
            model_confidence_interval=effect_metrics.get(
                "model_confidence_interval", effect_metrics["confidence_interval"]
            ),
            model_effect_scale=effect_metrics.get("model_effect_scale", "mean_difference"),
            model_effect_exponentiated=effect_metrics.get("model_effect_exponentiated", 1.0),
            covariate_adjustment_applied=covariate_effects["covariate_adjustment_applied"],
            covariates_used=covariate_effects["covariates_used"],
            covariate_adjusted_effect=covariate_effects["covariate_adjusted_effect"],
            covariate_adjusted_p_value=covariate_effects["covariate_adjusted_p_value"],
            covariate_adjusted_confidence_interval=covariate_effects[
                "covariate_adjusted_confidence_interval"
            ],
            covariate_adjusted_model_type=covariate_effects["covariate_adjusted_model_type"],
            covariate_adjusted_effect_scale=covariate_effects["covariate_adjusted_effect_scale"],
            covariate_adjusted_effect_exponentiated=covariate_effects[
                "covariate_adjusted_effect_exponentiated"
            ],
            power=power,
            required_sample_size=required_n,
            is_sample_adequate=is_sample_adequate,
            did_treatment_change=did_treatment_change,
            did_control_change=did_control_change,
            did_effect=did_effect,
            aa_test_passed=prepared.aa_test_passed,
            aa_p_value=prepared.aa_p_value,
            bootstrapping_applied=prepared.bootstrapping_applied,
            original_control_size=prepared.original_control_size,
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
            sequential_information_fraction=float(sequential_results["information_fraction"]),
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
                bayesian_results["total_effect"] / treatment_n if treatment_n > 0 else 0.0
            ),
            rows_dropped=prepared.rows_dropped,
            achieved_mde=achieved_mde,
            target_effect_size=target_effect_size,
            statistical_warnings=list(dict.fromkeys(statistical_warnings)),
            cuped_applied=bool(cuped_result and cuped_result.applied),
            cuped_theta=float(cuped_result.theta) if cuped_result else 0.0,
            cuped_variance_reduction=(
                float(cuped_result.variance_reduction) if cuped_result else 0.0
            ),
        )

    def run_segmented_analysis(
        self,
        sequential_config: Optional[Mapping[str, Any]] = None,
    ) -> List[ABTestResult]:
        """Run A/B tests for all segments or overall if no segment column."""
        self.segment_failures = []

        if "segment" not in self.column_mapping:
            return [self.run_ab_test(sequential_config=sequential_config)]

        if self.df is None:
            raise ValueError("No data loaded")

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
                failure = {
                    "segment": str(segment),
                    "error": str(error),
                    "error_type": type(error).__name__,
                }
                self.segment_failures.append(failure)
                logger.warning(
                    "Skipping segment during segmented analysis",
                    extra={"segment": str(segment), "error": str(error)},
                )

        if len(results) > 1:
            self._apply_multiple_testing_correction(results)

        return results

    def _apply_multiple_testing_correction(self, results: List[ABTestResult]) -> None:
        """Apply BH/FDR correction across segment-level frequentist p-values."""
        apply_fdr_correction(results, significance_level=self.significance_level)

    def generate_summary(self, results: List[ABTestResult]) -> ABTestSummary:
        """Generate aggregate summary and recommendations."""
        return self.summary_builder.generate_summary(
            results,
            segment_failures=self.segment_failures,
        )

    def _generate_recommendations(self, results: List[ABTestResult]) -> List[str]:
        """Backward-compatible recommendation helper."""
        return self.summary_builder._generate_recommendations(results)

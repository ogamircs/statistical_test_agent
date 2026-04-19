"""
Covariate Resolution and Adjustment for A/B Test Analysis

Handles parsing covariate columns, resolving them against the dataframe schema,
aligning analysis inputs, and running covariate-adjusted effect models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .statsmodels_engine import StatsmodelsABTestEngine


class CovariateResolver:
    """Parse, resolve, and apply covariate adjustments for segment-level analysis.

    Parameters
    ----------
    stats_engine:
        The inferential statistics engine (used for adjusted effect estimation).
    """

    def __init__(self, stats_engine: StatsmodelsABTestEngine) -> None:
        self.stats_engine = stats_engine

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    @staticmethod
    def parse_covariate_columns(raw_covariates: Any) -> List[str]:
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

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def resolve_covariate_columns(
        self,
        *,
        df: pd.DataFrame,
        group_col: str,
        post_effect_col: str,
        pre_effect_col: Optional[str],
        column_mapping: Dict[str, Any],
    ) -> List[str]:
        """Collect numeric covariates from mapping and pre-period metrics when available."""
        covariates = self.parse_covariate_columns(column_mapping.get("covariates"))
        if pre_effect_col is not None and pre_effect_col in df.columns and pre_effect_col not in covariates:
            covariates.append(pre_effect_col)

        excluded = {group_col, post_effect_col}
        segment_col = column_mapping.get("segment")
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

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------
    def apply_covariate_alignment(
        self,
        *,
        covariate_columns: List[str],
        post_effect_col: str,
        pre_effect_col: Optional[str],
        has_pre_effect: bool,
        treatment_analysis_df: pd.DataFrame,
        control_analysis_df: pd.DataFrame,
        treatment_post_aligned: np.ndarray,
        control_post_aligned: np.ndarray,
        treatment_pre_aligned: Optional[np.ndarray],
        control_pre_aligned: Optional[np.ndarray],
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        float,
        float,
    ]:
        """Restrict analysis inputs to rows that support covariate-adjusted inference.

        Returns
        -------
        tuple of (treatment_model_df, control_model_df,
                  treatment_post_aligned, control_post_aligned,
                  treatment_pre_aligned, control_pre_aligned,
                  treatment_pre_mean, control_pre_mean)
        """
        treatment_model_df = treatment_analysis_df.copy()
        control_model_df = control_analysis_df.copy()
        t_pre_mean = float(np.mean(treatment_pre_aligned)) if treatment_pre_aligned is not None and len(treatment_pre_aligned) > 0 else 0.0
        c_pre_mean = float(np.mean(control_pre_aligned)) if control_pre_aligned is not None and len(control_pre_aligned) > 0 else 0.0

        if not covariate_columns:
            return (
                treatment_model_df,
                control_model_df,
                treatment_post_aligned,
                control_post_aligned,
                treatment_pre_aligned,
                control_pre_aligned,
                t_pre_mean,
                c_pre_mean,
            )

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
                t_pre_mean = float(np.mean(treatment_pre_aligned))
                c_pre_mean = float(np.mean(control_pre_aligned))

        return (
            treatment_model_df,
            control_model_df,
            treatment_post_aligned,
            control_post_aligned,
            treatment_pre_aligned,
            control_pre_aligned,
            t_pre_mean,
            c_pre_mean,
        )

    # ------------------------------------------------------------------
    # Adjusted effect model
    # ------------------------------------------------------------------
    def run_covariate_adjusted_effect_model(
        self,
        *,
        post_effect_col: str,
        metric_type_option: str,
        count_model_option: str,
        heavy_tail_strategy_option: str,
        covariate_columns: List[str],
        treatment_model_df: pd.DataFrame,
        control_model_df: pd.DataFrame,
        effect_size: float,
        p_value: float,
        confidence_interval: Tuple[float, float],
        model_effect_scale: str,
        model_effect_exponentiated: float,
    ) -> Dict[str, Any]:
        """Fit the covariate-adjusted effect model when enough aligned rows remain."""
        if not covariate_columns or len(treatment_model_df) < 2 or len(control_model_df) < 2:
            return {
                "covariate_adjustment_applied": False,
                "covariates_used": [],
                "covariate_adjusted_effect": effect_size,
                "covariate_adjusted_p_value": p_value,
                "covariate_adjusted_confidence_interval": confidence_interval,
                "covariate_adjusted_model_type": "none",
                "covariate_adjusted_effect_scale": model_effect_scale,
                "covariate_adjusted_effect_exponentiated": model_effect_exponentiated,
                "covariate_adjusted_diagnostics": {},
            }

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

        return {
            "covariate_adjustment_applied": bool(
                adjusted_effect_metrics.get("covariate_adjusted", False)
            ),
            "covariates_used": list(
                adjusted_effect_metrics.get("covariates_used", covariate_columns)
            ),
            "covariate_adjusted_effect": adjusted_effect_metrics.get(
                "model_effect",
                adjusted_effect_metrics.get("effect_size", effect_size),
            ),
            "covariate_adjusted_p_value": adjusted_effect_metrics.get("p_value", p_value),
            "covariate_adjusted_confidence_interval": adjusted_effect_metrics.get(
                "model_confidence_interval",
                adjusted_effect_metrics.get("confidence_interval", confidence_interval),
            ),
            "covariate_adjusted_model_type": adjusted_effect_metrics.get("model_type", "none"),
            "covariate_adjusted_effect_scale": adjusted_effect_metrics.get(
                "model_effect_scale",
                model_effect_scale,
            ),
            "covariate_adjusted_effect_exponentiated": adjusted_effect_metrics.get(
                "model_effect_exponentiated",
                1.0,
            ),
            "covariate_adjusted_diagnostics": adjusted_effect_metrics.get("diagnostics", {}),
        }

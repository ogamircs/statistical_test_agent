"""
Segment Preparation for A/B Test Analysis

Extracts group splitting, post-only baseline construction, pre-period alignment
with AA testing and bootstrap balancing into a dedicated component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .statsmodels_engine import StatsmodelsABTestEngine


@dataclass
class _PreparedSegmentData:
    """Aligned treatment/control inputs for the downstream statistical stages."""

    segment_name: str
    treatment_df: pd.DataFrame
    control_df: pd.DataFrame
    treatment_analysis_df: pd.DataFrame
    control_analysis_df: pd.DataFrame
    treatment_post_series: pd.Series
    control_post_series: pd.Series
    treatment_post_aligned: np.ndarray
    control_post_aligned: np.ndarray
    treatment_pre_aligned: Optional[np.ndarray]
    control_pre_aligned: Optional[np.ndarray]
    treatment_pre_mean: float
    control_pre_mean: float
    aa_test_passed: bool
    aa_p_value: float
    bootstrapping_applied: bool
    original_control_size: int
    has_pre_effect: bool


class SegmentPreparer:
    """Resolve group slices, pre/post alignment, and AA metadata for segments.

    Parameters
    ----------
    stats_engine:
        The inferential statistics engine (used for AA tests and bootstrapping).
    significance_level:
        Alpha threshold passed through from the analyzer.
    """

    def __init__(
        self,
        stats_engine: StatsmodelsABTestEngine,
        significance_level: float,
    ) -> None:
        self.stats_engine = stats_engine
        self.significance_level = significance_level

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def prepare(
        self,
        *,
        segment_name: str,
        df_filtered: pd.DataFrame,
        group_col: str,
        pre_effect_col: Optional[str],
        post_effect_col: str,
        treatment_label,
        control_label,
    ) -> _PreparedSegmentData:
        """Full preparation pipeline: split -> post-only baseline -> pre-align -> fallback."""
        treatment_df, control_df = self._split_groups(
            df_filtered=df_filtered,
            group_col=group_col,
            treatment_label=treatment_label,
            control_label=control_label,
        )
        prepared = self._build_post_only_segment_data(
            segment_name=segment_name,
            treatment_df=treatment_df,
            control_df=control_df,
            post_effect_col=post_effect_col,
            pre_effect_col=pre_effect_col,
            df_filtered=df_filtered,
        )
        self._align_pre_period_data(
            pre_effect_col=pre_effect_col,
            post_effect_col=post_effect_col,
            segment_name=segment_name,
            prepared=prepared,
        )
        self._ensure_post_only_fallback(post_effect_col=post_effect_col, prepared=prepared)
        return prepared

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split_groups(
        *,
        df_filtered: pd.DataFrame,
        group_col: str,
        treatment_label,
        control_label,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the filtered frame into treatment and control groups."""
        treatment_df = df_filtered[df_filtered[group_col] == treatment_label]
        control_df = df_filtered[df_filtered[group_col] == control_label]
        return treatment_df, control_df

    @staticmethod
    def _build_post_only_segment_data(
        *,
        segment_name: str,
        treatment_df: pd.DataFrame,
        control_df: pd.DataFrame,
        post_effect_col: str,
        pre_effect_col: Optional[str],
        df_filtered: pd.DataFrame,
    ) -> _PreparedSegmentData:
        """Construct the baseline post-only analysis payload before pre-period alignment."""
        treatment_post_series = treatment_df[post_effect_col].dropna()
        control_post_series = control_df[post_effect_col].dropna()

        if len(treatment_post_series) < 2 or len(control_post_series) < 2:
            raise ValueError(
                f"Insufficient data for segment '{segment_name}': "
                f"Treatment n={len(treatment_post_series)}, Control n={len(control_post_series)}"
            )

        return _PreparedSegmentData(
            segment_name=segment_name,
            treatment_df=treatment_df,
            control_df=control_df,
            treatment_analysis_df=treatment_df.dropna(subset=[post_effect_col]).copy(),
            control_analysis_df=control_df.dropna(subset=[post_effect_col]).copy(),
            treatment_post_series=treatment_post_series,
            control_post_series=control_post_series,
            treatment_post_aligned=treatment_post_series.to_numpy(),
            control_post_aligned=control_post_series.to_numpy(),
            treatment_pre_aligned=None,
            control_pre_aligned=None,
            treatment_pre_mean=0.0,
            control_pre_mean=0.0,
            aa_test_passed=True,
            aa_p_value=1.0,
            bootstrapping_applied=False,
            original_control_size=len(control_post_series),
            has_pre_effect=(
                pre_effect_col is not None
                and pre_effect_col in df_filtered.columns
            ),
        )

    def _align_pre_period_data(
        self,
        *,
        pre_effect_col: Optional[str],
        post_effect_col: str,
        segment_name: str,
        prepared: _PreparedSegmentData,
    ) -> None:
        """Align pre/post data, run AA checks, and bootstrap control when needed."""
        if not prepared.has_pre_effect or pre_effect_col is None:
            return

        treatment_aligned_df = prepared.treatment_df.dropna(
            subset=[pre_effect_col, post_effect_col]
        )
        control_aligned_df = prepared.control_df.dropna(
            subset=[pre_effect_col, post_effect_col]
        )

        treatment_pre = treatment_aligned_df[pre_effect_col].to_numpy()
        control_pre = control_aligned_df[pre_effect_col].to_numpy()
        if len(treatment_pre) < 2 or len(control_pre) < 2:
            prepared.has_pre_effect = False
            return

        aa_result = self.stats_engine.run_aa_test(treatment_pre, control_pre, segment_name)
        prepared.aa_test_passed = aa_result.is_balanced
        prepared.aa_p_value = aa_result.aa_p_value
        prepared.treatment_pre_mean = aa_result.treatment_pre_mean
        prepared.control_pre_mean = aa_result.control_pre_mean

        control_df_for_analysis = control_aligned_df
        if not prepared.aa_test_passed and len(control_aligned_df) > 0:
            control_df_for_analysis, aa_result = self.stats_engine.bootstrap_balanced_control(
                treatment_pre=treatment_pre,
                control_df=control_aligned_df,
                pre_col=pre_effect_col,
            )
            prepared.aa_test_passed = aa_result.is_balanced
            prepared.aa_p_value = aa_result.aa_p_value
            prepared.bootstrapping_applied = aa_result.bootstrapping_applied
            prepared.original_control_size = aa_result.original_control_size
            prepared.control_pre_mean = aa_result.control_pre_mean

        prepared.treatment_pre_aligned = treatment_aligned_df[pre_effect_col].to_numpy()
        prepared.treatment_post_aligned = treatment_aligned_df[post_effect_col].to_numpy()
        prepared.control_pre_aligned = control_df_for_analysis[pre_effect_col].to_numpy()
        prepared.control_post_aligned = control_df_for_analysis[post_effect_col].to_numpy()
        prepared.treatment_analysis_df = treatment_aligned_df.copy()
        prepared.control_analysis_df = control_df_for_analysis.copy()

        if len(prepared.treatment_pre_aligned) > 0:
            prepared.treatment_pre_mean = float(np.mean(prepared.treatment_pre_aligned))
        if len(prepared.control_pre_aligned) > 0:
            prepared.control_pre_mean = float(np.mean(prepared.control_pre_aligned))

    @staticmethod
    def _ensure_post_only_fallback(
        *,
        post_effect_col: str,
        prepared: _PreparedSegmentData,
    ) -> None:
        """Fallback to post-only arrays when pre/post alignment removes too much data."""
        if len(prepared.treatment_post_aligned) >= 2 and len(prepared.control_post_aligned) >= 2:
            return

        prepared.treatment_post_aligned = prepared.treatment_post_series.to_numpy()
        prepared.control_post_aligned = prepared.control_post_series.to_numpy()
        prepared.treatment_pre_aligned = None
        prepared.control_pre_aligned = None
        prepared.has_pre_effect = False
        prepared.treatment_analysis_df = prepared.treatment_df.dropna(
            subset=[post_effect_col]
        ).copy()
        prepared.control_analysis_df = prepared.control_df.dropna(
            subset=[post_effect_col]
        ).copy()

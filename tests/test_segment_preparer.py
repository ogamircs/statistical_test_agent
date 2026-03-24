"""Tests for the SegmentPreparer extraction from analyzer.py."""

import numpy as np
import pandas as pd
import pytest

from src.statistics.segment_preparer import SegmentPreparer, _PreparedSegmentData
from src.statistics.statsmodels_engine import StatsmodelsABTestEngine


@pytest.fixture
def engine():
    return StatsmodelsABTestEngine(significance_level=0.05, power_threshold=0.8)


@pytest.fixture
def preparer(engine):
    return SegmentPreparer(stats_engine=engine, significance_level=0.05)


def _make_df(n_treatment=50, n_control=50, has_pre=True, seed=42):
    """Create a synthetic experiment DataFrame."""
    rng = np.random.RandomState(seed)
    groups = ["treatment"] * n_treatment + ["control"] * n_control
    post = np.concatenate([
        rng.normal(10.5, 2, n_treatment),
        rng.normal(10.0, 2, n_control),
    ])
    data = {"group": groups, "post_effect": post}
    if has_pre:
        pre = np.concatenate([
            rng.normal(5.0, 1, n_treatment),
            rng.normal(5.0, 1, n_control),
        ])
        data["pre_effect"] = pre
    return pd.DataFrame(data)


class TestBalancedGroups:
    """Balanced treatment/control with pre-period data."""

    def test_prepare_returns_prepared_data(self, preparer):
        df = _make_df()
        result = preparer.prepare(
            segment_name="Overall",
            df_filtered=df,
            group_col="group",
            pre_effect_col="pre_effect",
            post_effect_col="post_effect",
            treatment_label="treatment",
            control_label="control",
        )
        assert isinstance(result, _PreparedSegmentData)
        assert result.segment_name == "Overall"
        assert len(result.treatment_post_aligned) > 0
        assert len(result.control_post_aligned) > 0

    def test_pre_effect_aligned(self, preparer):
        df = _make_df()
        result = preparer.prepare(
            segment_name="Overall",
            df_filtered=df,
            group_col="group",
            pre_effect_col="pre_effect",
            post_effect_col="post_effect",
            treatment_label="treatment",
            control_label="control",
        )
        assert result.has_pre_effect is True
        assert result.treatment_pre_aligned is not None
        assert result.control_pre_aligned is not None

    def test_aa_test_passes_for_balanced(self, preparer):
        df = _make_df(seed=42)
        result = preparer.prepare(
            segment_name="Overall",
            df_filtered=df,
            group_col="group",
            pre_effect_col="pre_effect",
            post_effect_col="post_effect",
            treatment_label="treatment",
            control_label="control",
        )
        # With the same pre-effect distribution the AA test should pass
        assert result.aa_test_passed is True


class TestImbalancedGroups:
    """Imbalanced treatment/control that may trigger bootstrapping."""

    def test_bootstrap_triggered_for_imbalanced_pre(self, preparer):
        rng = np.random.RandomState(99)
        n_t, n_c = 50, 100
        groups = ["treatment"] * n_t + ["control"] * n_c
        # Large mean difference in pre-period to force AA failure
        pre = np.concatenate([
            rng.normal(5.0, 0.5, n_t),
            rng.normal(8.0, 0.5, n_c),
        ])
        post = np.concatenate([
            rng.normal(10.0, 2, n_t),
            rng.normal(10.0, 2, n_c),
        ])
        df = pd.DataFrame({"group": groups, "pre_effect": pre, "post_effect": post})

        result = preparer.prepare(
            segment_name="Imbalanced",
            df_filtered=df,
            group_col="group",
            pre_effect_col="pre_effect",
            post_effect_col="post_effect",
            treatment_label="treatment",
            control_label="control",
        )
        # Bootstrap should have been attempted
        assert result.bootstrapping_applied is True
        assert result.original_control_size > 0


class TestMissingPreEffect:
    """No pre-period column available."""

    def test_post_only_when_no_pre(self, preparer):
        df = _make_df(has_pre=False)
        result = preparer.prepare(
            segment_name="PostOnly",
            df_filtered=df,
            group_col="group",
            pre_effect_col=None,
            post_effect_col="post_effect",
            treatment_label="treatment",
            control_label="control",
        )
        assert result.has_pre_effect is False
        assert result.treatment_pre_aligned is None
        assert result.control_pre_aligned is None
        assert result.aa_test_passed is True  # default when no pre-period

    def test_post_only_when_pre_col_not_in_df(self, preparer):
        df = _make_df(has_pre=False)
        result = preparer.prepare(
            segment_name="PostOnly",
            df_filtered=df,
            group_col="group",
            pre_effect_col="nonexistent_col",
            post_effect_col="post_effect",
            treatment_label="treatment",
            control_label="control",
        )
        assert result.has_pre_effect is False


class TestInsufficientData:
    """Edge case: too few rows to analyze."""

    def test_raises_on_insufficient_data(self, preparer):
        df = pd.DataFrame({
            "group": ["treatment", "control"],
            "post_effect": [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="Insufficient data"):
            preparer.prepare(
                segment_name="Tiny",
                df_filtered=df,
                group_col="group",
                pre_effect_col=None,
                post_effect_col="post_effect",
                treatment_label="treatment",
                control_label="control",
            )

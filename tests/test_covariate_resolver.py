"""Tests for the CovariateResolver extraction from analyzer.py."""

import numpy as np
import pandas as pd
import pytest

from src.statistics.covariate_resolver import CovariateResolver
from src.statistics.statsmodels_engine import StatsmodelsABTestEngine


@pytest.fixture
def engine():
    return StatsmodelsABTestEngine(significance_level=0.05, power_threshold=0.8)


@pytest.fixture
def resolver(engine):
    return CovariateResolver(stats_engine=engine)


class TestParseCovariateColumns:
    """Test the static parse_covariate_columns method."""

    def test_none_returns_empty(self):
        assert CovariateResolver.parse_covariate_columns(None) == []

    def test_string_single(self):
        assert CovariateResolver.parse_covariate_columns("age") == ["age"]

    def test_string_csv(self):
        assert CovariateResolver.parse_covariate_columns("age, income, score") == [
            "age",
            "income",
            "score",
        ]

    def test_string_deduplication(self):
        assert CovariateResolver.parse_covariate_columns("age, age, income") == [
            "age",
            "income",
        ]

    def test_list_input(self):
        assert CovariateResolver.parse_covariate_columns(["age", "income"]) == [
            "age",
            "income",
        ]

    def test_list_with_non_string(self):
        result = CovariateResolver.parse_covariate_columns([1, 2, 3])
        assert result == ["1", "2", "3"]

    def test_unsupported_type_returns_empty(self):
        assert CovariateResolver.parse_covariate_columns(42) == []
        assert CovariateResolver.parse_covariate_columns(True) == []


class TestResolveCovariateColumns:
    """Test resolution against a real DataFrame."""

    def test_numeric_columns_resolved(self, resolver):
        df = pd.DataFrame({
            "group": ["A", "B"] * 10,
            "post_effect": np.random.rand(20),
            "age": np.random.rand(20),
            "income": np.random.rand(20),
        })
        mapping = {"covariates": "age, income", "group": "group", "effect_value": "post_effect"}
        result = resolver.resolve_covariate_columns(
            df=df,
            group_col="group",
            post_effect_col="post_effect",
            pre_effect_col=None,
            column_mapping=mapping,
        )
        assert "age" in result
        assert "income" in result

    def test_non_numeric_excluded(self, resolver):
        df = pd.DataFrame({
            "group": ["A", "B"] * 10,
            "post_effect": np.random.rand(20),
            "category": ["cat_a", "cat_b"] * 10,
        })
        mapping = {"covariates": "category", "group": "group", "effect_value": "post_effect"}
        result = resolver.resolve_covariate_columns(
            df=df,
            group_col="group",
            post_effect_col="post_effect",
            pre_effect_col=None,
            column_mapping=mapping,
        )
        assert "category" not in result

    def test_group_col_excluded(self, resolver):
        df = pd.DataFrame({
            "group": [0, 1] * 10,
            "post_effect": np.random.rand(20),
        })
        mapping = {"covariates": "group", "group": "group", "effect_value": "post_effect"}
        result = resolver.resolve_covariate_columns(
            df=df,
            group_col="group",
            post_effect_col="post_effect",
            pre_effect_col=None,
            column_mapping=mapping,
        )
        assert "group" not in result

    def test_pre_effect_auto_included(self, resolver):
        df = pd.DataFrame({
            "group": ["A", "B"] * 10,
            "post_effect": np.random.rand(20),
            "pre_effect": np.random.rand(20),
        })
        mapping = {"group": "group", "effect_value": "post_effect"}
        result = resolver.resolve_covariate_columns(
            df=df,
            group_col="group",
            post_effect_col="post_effect",
            pre_effect_col="pre_effect",
            column_mapping=mapping,
        )
        assert "pre_effect" in result

    def test_missing_column_skipped(self, resolver):
        df = pd.DataFrame({
            "group": ["A", "B"] * 10,
            "post_effect": np.random.rand(20),
        })
        mapping = {"covariates": "nonexistent", "group": "group", "effect_value": "post_effect"}
        result = resolver.resolve_covariate_columns(
            df=df,
            group_col="group",
            post_effect_col="post_effect",
            pre_effect_col=None,
            column_mapping=mapping,
        )
        assert result == []

    def test_segment_col_excluded(self, resolver):
        df = pd.DataFrame({
            "group": ["A", "B"] * 10,
            "post_effect": np.random.rand(20),
            "segment": np.random.randint(0, 3, 20),
        })
        mapping = {
            "covariates": "segment",
            "group": "group",
            "effect_value": "post_effect",
            "segment": "segment",
        }
        result = resolver.resolve_covariate_columns(
            df=df,
            group_col="group",
            post_effect_col="post_effect",
            pre_effect_col=None,
            column_mapping=mapping,
        )
        assert "segment" not in result

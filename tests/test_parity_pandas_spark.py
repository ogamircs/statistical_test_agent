"""
Golden-dataset parity tests between pandas and PySpark analyzers.

These tests verify:
1) canonical result schema parity
2) key numeric metric parity within tolerances
3) clean skip behavior when Spark is unavailable
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from src.statistics.analyzer import ABTestAnalyzer
from src.statistics.models import ABTestResult, canonical_result_as_dict

pyspark = pytest.importorskip(
    "pyspark",
    reason="PySpark not installed; skipping pandas-vs-Spark parity tests.",
)

from src.statistics.pyspark_analyzer import (  # noqa: E402
    PySparkABTestAnalyzer,
    create_spark_session,
)

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"

CANONICAL_FIELDS = set(ABTestResult.__dataclass_fields__.keys())
PARITY_NUMERIC_FIELDS = (
    "treatment_size",
    "control_size",
    "treatment_mean",
    "control_mean",
    "effect_size",
    "did_effect",
    "treatment_proportion",
    "control_proportion",
    "proportion_diff",
)


def _create_spark_or_skip(**kwargs: Any):
    try:
        return create_spark_session(**kwargs)
    except Exception as exc:
        pytest.skip(f"Spark runtime unavailable; skipping parity tests. Details: {exc}")


def _assert_numeric_close(left: float, right: float, *, rel: float = 1e-6, abs_: float = 1e-6) -> None:
    assert math.isfinite(left), f"left value is not finite: {left}"
    assert math.isfinite(right), f"right value is not finite: {right}"
    assert right == pytest.approx(left, rel=rel, abs=abs_)


def _result_by_segment(results: List[Any]) -> Dict[str, Dict[str, Any]]:
    return {result.segment: canonical_result_as_dict(result) for result in results}


def _run_pandas_analysis(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    labels: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    analyzer = ABTestAnalyzer(significance_level=0.05, power_threshold=0.8)
    analyzer.set_dataframe(df.copy())
    analyzer.set_column_mapping(mapping)
    analyzer.set_group_labels(labels["treatment"], labels["control"])

    overall = canonical_result_as_dict(analyzer.run_ab_test())
    segmented = _result_by_segment(analyzer.run_segmented_analysis())
    return overall, segmented


def _run_spark_analysis(
    spark: Any,
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    labels: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    analyzer = PySparkABTestAnalyzer(spark=spark, significance_level=0.05, power_threshold=0.8)
    analyzer.set_dataframe(spark.createDataFrame(df))
    analyzer.set_column_mapping(mapping)
    analyzer.set_group_labels(labels["treatment"], labels["control"])

    overall = canonical_result_as_dict(analyzer.run_ab_test())
    segmented = _result_by_segment(analyzer.run_segmented_analysis())
    return overall, segmented


@pytest.fixture(scope="module")
def spark_session():
    spark = _create_spark_or_skip(
        app_name="PandasSparkParityGoldenTests",
        master="local[2]",
        config={
            "spark.sql.shuffle.partitions": "2",
            "spark.ui.enabled": "false",
            "spark.driver.memory": "1g",
        },
    )
    yield spark
    try:
        spark.stop()
    except Exception:
        pass


@pytest.fixture(
    params=[
        pytest.param(
            {
                "dataset": "sample_ab_data.csv",
                "usecols": ["experiment_group", "customer_segment", "post_effect"],
                "mapping": {
                    "group": "experiment_group",
                    "segment": "customer_segment",
                    "effect_value": "post_effect",
                    "post_effect": "post_effect",
                },
                "labels": {"treatment": "treatment", "control": "control"},
            },
            id="sample_ab_data",
        ),
        pytest.param(
            {
                "dataset": "sample_ab_data_alt.csv",
                "usecols": ["ab_variant", "tier", "revenue"],
                "mapping": {
                    "group": "ab_variant",
                    "segment": "tier",
                    "effect_value": "revenue",
                    "post_effect": "revenue",
                },
                "labels": {"treatment": "test", "control": "ctrl"},
            },
            id="sample_ab_data_alt",
        ),
    ]
)
def golden_dataset_case(request):
    case = request.param
    dataset_path = DATA_DIR / case["dataset"]
    df = pd.read_csv(dataset_path, usecols=case["usecols"])
    return df, case["mapping"], case["labels"], case["dataset"]


class TestGoldenDatasetParity:
    def test_canonical_schema_parity(
        self,
        spark_session,
        golden_dataset_case,
    ):
        df, mapping, labels, _ = golden_dataset_case

        pandas_overall, pandas_segmented = _run_pandas_analysis(df, mapping, labels)
        spark_overall, spark_segmented = _run_spark_analysis(spark_session, df, mapping, labels)

        assert set(pandas_overall.keys()) == CANONICAL_FIELDS
        assert set(spark_overall.keys()) == CANONICAL_FIELDS
        assert set(pandas_overall.keys()) == set(spark_overall.keys())

        assert set(pandas_segmented.keys()) == set(spark_segmented.keys())
        for segment in pandas_segmented:
            pandas_payload = pandas_segmented[segment]
            spark_payload = spark_segmented[segment]
            assert set(pandas_payload.keys()) == CANONICAL_FIELDS
            assert set(spark_payload.keys()) == CANONICAL_FIELDS
            assert set(pandas_payload.keys()) == set(spark_payload.keys())

    def test_key_numeric_parity_with_tolerances(
        self,
        spark_session,
        golden_dataset_case,
    ):
        df, mapping, labels, dataset_name = golden_dataset_case

        pandas_overall, pandas_segmented = _run_pandas_analysis(df, mapping, labels)
        spark_overall, spark_segmented = _run_spark_analysis(spark_session, df, mapping, labels)

        for field_name in PARITY_NUMERIC_FIELDS:
            _assert_numeric_close(
                float(pandas_overall[field_name]),
                float(spark_overall[field_name]),
                rel=1e-6,
                abs_=1e-6,
            )

        for segment in pandas_segmented:
            pandas_payload = pandas_segmented[segment]
            spark_payload = spark_segmented[segment]
            for field_name in PARITY_NUMERIC_FIELDS:
                _assert_numeric_close(
                    float(pandas_payload[field_name]),
                    float(spark_payload[field_name]),
                    rel=1e-6,
                    abs_=1e-6,
                )

        # Extra guardrail: counts should match exactly on each dataset
        for segment in pandas_segmented:
            assert pandas_segmented[segment]["treatment_size"] == spark_segmented[segment]["treatment_size"], (
                f"{dataset_name} segment={segment} treatment_size mismatch"
            )
            assert pandas_segmented[segment]["control_size"] == spark_segmented[segment]["control_size"], (
                f"{dataset_name} segment={segment} control_size mismatch"
            )

    def test_covariate_adjustment_parity(self, spark_session) -> None:
        """Spark covariate-adjusted effect should match pandas within tolerance.

        Builds a synthetic dataset where the post metric is driven by both a
        treatment uplift and a numeric covariate. Both backends should report
        the same covariate_adjusted_effect when the same covariate is mapped.
        """
        rng = np.random.default_rng(0)
        n_per_arm = 600
        treatment_pre = rng.normal(50.0, 10.0, n_per_arm)
        control_pre = rng.normal(50.0, 10.0, n_per_arm)
        treatment_post = 0.7 * treatment_pre + rng.normal(2.5, 1.0, n_per_arm)
        control_post = 0.7 * control_pre + rng.normal(0.0, 1.0, n_per_arm)
        df = pd.DataFrame(
            {
                "experiment_group": ["treatment"] * n_per_arm + ["control"] * n_per_arm,
                "pre_effect": np.concatenate([treatment_pre, control_pre]),
                "post_effect": np.concatenate([treatment_post, control_post]),
            }
        )
        mapping = {
            "group": "experiment_group",
            "effect_value": "post_effect",
            "post_effect": "post_effect",
            "pre_effect": "pre_effect",
            "covariates": ["pre_effect"],
        }
        labels = {"treatment": "treatment", "control": "control"}

        pandas_analyzer = ABTestAnalyzer(significance_level=0.05, power_threshold=0.8)
        pandas_analyzer.set_dataframe(df.copy())
        pandas_analyzer.set_column_mapping(mapping)
        pandas_analyzer.set_group_labels(labels["treatment"], labels["control"])
        pandas_result = canonical_result_as_dict(pandas_analyzer.run_ab_test())

        spark_analyzer = PySparkABTestAnalyzer(
            spark=spark_session, significance_level=0.05, power_threshold=0.8
        )
        spark_analyzer.set_dataframe(spark_session.createDataFrame(df))
        spark_analyzer.set_column_mapping(mapping)
        spark_analyzer.set_group_labels(labels["treatment"], labels["control"])
        spark_result = canonical_result_as_dict(spark_analyzer.run_ab_test())

        assert pandas_result["covariate_adjustment_applied"] is True
        assert spark_result["covariate_adjustment_applied"] is True
        _assert_numeric_close(
            float(pandas_result["covariate_adjusted_effect"]),
            float(spark_result["covariate_adjusted_effect"]),
            rel=1e-3,
            abs_=1e-3,
        )

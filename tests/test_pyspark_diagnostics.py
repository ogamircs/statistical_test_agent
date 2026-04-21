"""Spark backend diagnostics — driver-only logic, no SparkSession needed."""

from __future__ import annotations

from types import SimpleNamespace

from src.statistics.pyspark_analyzer import PySparkABTestAnalyzer


def _build(stub_attrs: dict | None = None):
    stub_attrs = stub_attrs or {}
    return SimpleNamespace(significance_level=0.05, **stub_attrs)


def test_balanced_split_no_srm_mismatch() -> None:
    payload = PySparkABTestAnalyzer._build_diagnostics_payload(
        _build(),
        n_t=500,
        n_c=500,
        var_t=4.0,
        var_c=4.0,
    )
    srm = payload["experiment_quality"]["srm"]
    assert srm["is_sample_ratio_mismatch"] is False
    assert payload["experiment_quality"]["assumptions"]["equal_variance_passed"] is True


def test_imbalanced_split_flags_srm() -> None:
    payload = PySparkABTestAnalyzer._build_diagnostics_payload(
        _build(),
        n_t=100,
        n_c=900,
        var_t=4.0,
        var_c=4.0,
    )
    srm = payload["experiment_quality"]["srm"]
    assert srm["is_sample_ratio_mismatch"] is True
    assert srm["p_value"] < 0.05


def test_extreme_variance_ratio_flags_unequal_variance() -> None:
    payload = PySparkABTestAnalyzer._build_diagnostics_payload(
        _build(),
        n_t=400,
        n_c=400,
        var_t=1.0,
        var_c=25.0,  # 25x variance ratio
    )
    assumptions = payload["experiment_quality"]["assumptions"]
    assert assumptions["variance_ratio"] == 25.0
    assert assumptions["equal_variance_passed"] is False


def test_zero_variance_marks_equal_variance_unknown() -> None:
    payload = PySparkABTestAnalyzer._build_diagnostics_payload(
        _build(),
        n_t=400,
        n_c=400,
        var_t=0.0,
        var_c=0.0,
    )
    assumptions = payload["experiment_quality"]["assumptions"]
    assert assumptions["equal_variance_passed"] is None


def test_payload_documents_outlier_sensitivity_skip() -> None:
    payload = PySparkABTestAnalyzer._build_diagnostics_payload(
        _build(),
        n_t=400,
        n_c=400,
        var_t=4.0,
        var_c=4.0,
    )
    outlier = payload["experiment_quality"]["outlier_sensitivity"]
    assert outlier["is_applicable"] is False
    assert "spark" in outlier["reason"]

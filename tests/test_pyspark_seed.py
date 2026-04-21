"""Driver-only reproducibility tests for the Spark backend's Monte Carlo step.

The Spark Bayesian sampler runs entirely on the driver after summary stats
are computed in Spark, so we can exercise its seed correctness without an
actual SparkSession by binding the unbound method onto a stub.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.statistics.pyspark_analyzer import PySparkABTestAnalyzer


def _run_with_seed(seed: int) -> dict:
    stub = SimpleNamespace(seed=seed)
    return PySparkABTestAnalyzer._run_bayesian_test_montecarlo(
        stub,
        mean_t=10.5,
        var_t=4.0,
        n_t=200,
        mean_c=10.0,
        var_c=4.0,
        n_c=200,
        n_samples=4_000,
    )


def test_spark_montecarlo_reproducible_for_same_seed() -> None:
    a = _run_with_seed(7)
    b = _run_with_seed(7)
    assert a["prob_treatment_better"] == b["prob_treatment_better"]
    assert a["credible_interval"] == b["credible_interval"]


def test_spark_montecarlo_changes_with_seed() -> None:
    a = _run_with_seed(1)
    b = _run_with_seed(99)
    assert a["credible_interval"] != b["credible_interval"]

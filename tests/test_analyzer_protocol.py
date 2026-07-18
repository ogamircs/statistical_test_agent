"""Verify both analyzer backends satisfy the shared protocol."""
from src.statistics.analyzer import ABTestAnalyzer
from src.statistics.analyzer_protocol import ABAnalyzerProtocol


def test_pandas_analyzer_satisfies_protocol():
    assert isinstance(ABTestAnalyzer(), ABAnalyzerProtocol)


# Conditional test for Spark — module-level import is now always safe
# (pyspark imports are guarded), but the constructor still requires the
# runtime to be present.
from src.statistics.pyspark_analyzer import (  # noqa: E402
    PYSPARK_RUNTIME_AVAILABLE,
    PySparkABTestAnalyzer,
)
from tests.spark_gate import skip_or_fail  # noqa: E402


def test_spark_analyzer_satisfies_protocol():
    if not PYSPARK_RUNTIME_AVAILABLE:
        skip_or_fail("PySpark not installed")
    try:
        analyzer = PySparkABTestAnalyzer()
    except Exception as exc:  # Spark requires a working JVM — skip when unavailable
        skip_or_fail(f"Spark runtime unavailable: {exc}")
    assert isinstance(analyzer, ABAnalyzerProtocol)

"""Verify both analyzer backends satisfy the shared protocol."""
import pytest

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


@pytest.mark.skipif(not PYSPARK_RUNTIME_AVAILABLE, reason="PySpark not installed")
def test_spark_analyzer_satisfies_protocol():
    try:
        analyzer = PySparkABTestAnalyzer()
    except Exception as exc:  # Spark requires a working JVM — skip when unavailable
        pytest.skip(f"Spark runtime unavailable: {exc}")
    assert isinstance(analyzer, ABAnalyzerProtocol)

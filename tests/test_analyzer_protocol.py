"""Verify both analyzer backends satisfy the shared protocol."""
import pytest
from src.statistics.analyzer_protocol import ABAnalyzerProtocol
from src.statistics.analyzer import ABTestAnalyzer


def test_pandas_analyzer_satisfies_protocol():
    assert isinstance(ABTestAnalyzer(), ABAnalyzerProtocol)


# Conditional test for Spark
try:
    from src.statistics.pyspark_analyzer import PySparkABTestAnalyzer
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False


@pytest.mark.skipif(not HAS_SPARK, reason="PySpark not installed")
def test_spark_analyzer_satisfies_protocol():
    assert isinstance(PySparkABTestAnalyzer(), ABAnalyzerProtocol)

"""Verify the analyzer satisfies the agent-facing protocol."""
from src.statistics.analyzer import ABTestAnalyzer
from src.statistics.analyzer_protocol import ABAnalyzerProtocol


def test_pandas_analyzer_satisfies_protocol():
    assert isinstance(ABTestAnalyzer(), ABAnalyzerProtocol)

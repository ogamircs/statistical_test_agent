"""Spark query_data validation paths — exercise without a SparkSession.

The actual SQL execution is gated by Spark availability; only the
input-validation surface is asserted here. End-to-end execution is
covered by the gated parity suite.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent_reporting import AgentUserFacingError
from src.statistics.pyspark_analyzer import PySparkABTestAnalyzer


def _stub(df=None) -> SimpleNamespace:
    return SimpleNamespace(df=df, spark=None)


def test_query_rejects_unloaded_data() -> None:
    with pytest.raises(AgentUserFacingError) as exc:
        PySparkABTestAnalyzer.query_data(_stub(df=None), "experiment_group = 'treatment'")
    assert exc.value.code == "DATA_NOT_LOADED"


def test_query_rejects_empty_string() -> None:
    df_marker = SimpleNamespace()
    with pytest.raises(AgentUserFacingError) as exc:
        PySparkABTestAnalyzer.query_data(_stub(df=df_marker), "   ")
    assert exc.value.code == "INVALID_QUERY"


def test_query_blocks_dml_keywords() -> None:
    df_marker = SimpleNamespace()
    for bad in (
        "DROP TABLE ab_test_data",
        "DELETE FROM ab_test_data WHERE 1=1",
        "INSERT INTO ab_test_data VALUES (1)",
        "UPDATE ab_test_data SET x = 1",
    ):
        with pytest.raises(AgentUserFacingError) as exc:
            PySparkABTestAnalyzer.query_data(_stub(df=df_marker), bad)
        assert exc.value.code == "INVALID_QUERY", bad


def test_query_rejects_negative_max_rows() -> None:
    class _FakeView:
        def createOrReplaceTempView(self, name):
            return None

    with pytest.raises(AgentUserFacingError) as exc:
        PySparkABTestAnalyzer.query_data(
            _stub(df=_FakeView()),
            "experiment_group = 'treatment'",
            max_rows=0,
        )
    assert exc.value.code == "INVALID_QUERY"

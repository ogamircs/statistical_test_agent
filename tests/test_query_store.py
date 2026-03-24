"""Tests for the SQLite-backed query store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from pandas.errors import DatabaseError

from src.query_store import SQLiteQueryStore
from src.statistics.models import ABTestResult, ABTestSummary


def _sample_result() -> ABTestResult:
    return ABTestResult(
        segment="Premium",
        treatment_size=50,
        control_size=45,
        treatment_mean=12.5,
        control_mean=10.0,
        effect_size=2.5,
        total_effect=125.0,
        p_value=0.01,
        is_significant=True,
    )


def test_query_store_persists_raw_and_analysis_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "query_store.sqlite"
    store = SQLiteQueryStore(db_path)

    raw_df = pd.DataFrame(
        {
            "experiment_group": ["treatment", "control"],
            "customer_segment": ["Premium", "Premium"],
            "post_effect": [12.5, 10.0],
        }
    )
    summary = ABTestSummary(
        total_segments_analyzed=1,
        significant_segments=1,
        total_treatment_customers=50,
        total_control_customers=45,
        total_effect_size=125.0,
    )

    store.save_raw_dataframe(raw_df)
    store.save_segment_results([_sample_result()])
    store.save_summary(summary)

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"raw_data", "analysis_segment_results", "analysis_summary"}.issubset(tables)

        raw_count = connection.execute("SELECT COUNT(*) FROM raw_data").fetchone()[0]
        result_row = connection.execute(
            "SELECT segment, effect_size, total_effect FROM analysis_segment_results"
        ).fetchone()
        summary_row = connection.execute(
            "SELECT total_segments_analyzed, total_effect_size FROM analysis_summary"
        ).fetchone()

    assert raw_count == 2
    assert result_row == ("Premium", 2.5, 125.0)
    assert summary_row == (1, 125.0)


def test_query_store_exposes_schema_context(tmp_path: Path) -> None:
    store = SQLiteQueryStore(tmp_path / "schema.sqlite")
    store.save_raw_dataframe(
        pd.DataFrame({"segment": ["Premium"], "metric": [1.5]})
    )

    schema = store.describe_schema()

    assert "raw_data" in schema
    assert "segment" in schema
    assert "metric" in schema


def test_execute_query_blocks_mutations(tmp_path: Path) -> None:
    store = SQLiteQueryStore(tmp_path / "readonly.sqlite")
    store.save_raw_dataframe(
        pd.DataFrame({"segment": ["Premium"], "metric": [1.5]})
    )

    with pytest.raises((sqlite3.OperationalError, DatabaseError), match="readonly"):
        store.execute_query("DROP TABLE raw_data")


def test_execute_query_allows_select(tmp_path: Path) -> None:
    store = SQLiteQueryStore(tmp_path / "readonly.sqlite")
    store.save_raw_dataframe(
        pd.DataFrame({"segment": ["Premium", "Basic"], "metric": [1.5, 2.0]})
    )

    result = store.execute_query("SELECT count(*) AS cnt FROM raw_data")

    assert isinstance(result, pd.DataFrame)
    assert result.iloc[0]["cnt"] == 2

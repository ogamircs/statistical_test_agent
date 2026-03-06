"""Tests for safe SQL question answering over the query store."""

from __future__ import annotations

import pandas as pd
import pytest

from src.agent_reporting import AgentUserFacingError
from src.query_store import SQLiteQueryStore
from src.sql_query_service import SQLQueryPlan, SQLQueryService
from src.statistics.models import ABTestResult, ABTestSummary


class _Planner:
    def __init__(self, sql: str):
        self.sql = sql
        self.calls: list[str] = []

    def generate_sql(self, question: str, schema: str) -> SQLQueryPlan:
        self.calls.append(question)
        return SQLQueryPlan(sql=self.sql)


def _build_store(tmp_path):
    store = SQLiteQueryStore(tmp_path / "qa.sqlite")
    store.save_raw_dataframe(
        pd.DataFrame(
            {
                "experiment_group": ["treatment", "control", "treatment"],
                "customer_segment": ["Premium", "Premium", "Standard"],
                "post_effect": [12.5, 10.0, 7.2],
            }
        )
    )
    store.save_segment_results(
        [
            ABTestResult(
                segment="Premium",
                treatment_size=50,
                control_size=45,
                effect_size=2.5,
                total_effect=125.0,
                p_value=0.01,
                is_significant=True,
            )
        ]
    )
    store.save_summary(ABTestSummary(total_segments_analyzed=1))
    return store


def test_sql_query_service_answers_analysis_question(tmp_path) -> None:
    store = _build_store(tmp_path)
    planner = _Planner(
        "SELECT segment, total_effect FROM analysis_segment_results WHERE segment = 'Premium'"
    )
    service = SQLQueryService(query_store=store, sql_planner=planner)

    answer = service.answer_question("What is the total effect size for Premium?")

    assert answer.row_count == 1
    assert "analysis_segment_results" in answer.source_tables
    assert answer.data.iloc[0]["total_effect"] == 125.0


def test_sql_query_service_rejects_unsafe_sql(tmp_path) -> None:
    store = _build_store(tmp_path)
    planner = _Planner("DROP TABLE raw_data")
    service = SQLQueryService(query_store=store, sql_planner=planner)

    with pytest.raises(AgentUserFacingError) as exc_info:
        service.answer_question("Delete the table")

    assert exc_info.value.code == "INVALID_SQL_QUERY"


def test_sql_query_service_limits_rows(tmp_path) -> None:
    store = _build_store(tmp_path)
    planner = _Planner("SELECT experiment_group, customer_segment FROM raw_data")
    service = SQLQueryService(query_store=store, sql_planner=planner, default_limit=2)

    answer = service.answer_question("Show me all rows")

    assert answer.row_count == 2


def test_sql_query_service_preserves_existing_limit(tmp_path) -> None:
    store = _build_store(tmp_path)
    planner = _Planner("SELECT total_effect FROM analysis_segment_results WHERE segment = 'Premium' LIMIT 1")
    service = SQLQueryService(query_store=store, sql_planner=planner, default_limit=20)

    answer = service.answer_question("What is the total effect size for Premium?")

    assert answer.row_count == 1
    assert answer.sql.endswith("LIMIT 1")


def test_sql_query_service_preserves_existing_limit_in_multiline_sql(tmp_path) -> None:
    store = _build_store(tmp_path)
    planner = _Planner(
        "SELECT total_effect\nFROM analysis_segment_results\nWHERE segment = 'Premium'\nLIMIT 1"
    )
    service = SQLQueryService(query_store=store, sql_planner=planner, default_limit=20)

    answer = service.answer_question("What is the total effect size for Premium?")

    assert answer.row_count == 1
    assert answer.sql.endswith("LIMIT 1")

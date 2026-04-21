"""Tests for centralized output truncation helpers."""

from __future__ import annotations

import pandas as pd

from src.agent_reporting import (
    render_column_values_output,
    render_query_data_output,
)
from src.output_truncation import (
    DEFAULT_LLM_ROW_LIMIT,
    truncate_dataframe_for_llm,
    truncate_iterable_for_llm,
)


def test_truncate_dataframe_no_suffix_when_under_limit() -> None:
    df = pd.DataFrame({"a": range(5)})
    head, suffix = truncate_dataframe_for_llm(df, max_rows=10)
    assert len(head) == 5
    assert suffix is None


def test_truncate_dataframe_emits_suffix_when_over_limit() -> None:
    df = pd.DataFrame({"a": range(50)})
    head, suffix = truncate_dataframe_for_llm(df, max_rows=20)
    assert len(head) == 20
    assert suffix == "_showing first 20 of 50 rows_"


def test_truncate_iterable_returns_count_when_over_limit() -> None:
    head, omitted = truncate_iterable_for_llm(range(30), max_items=10)
    assert head == list(range(10))
    assert omitted == 20


def test_render_query_data_includes_truncation_notice() -> None:
    df = pd.DataFrame({"x": range(45)})
    output = render_query_data_output(df)
    assert "Query result (45 rows)" in output
    assert f"_showing first {DEFAULT_LLM_ROW_LIMIT} of 45 rows_" in output


def test_render_query_data_omits_notice_for_small_results() -> None:
    df = pd.DataFrame({"x": range(3)})
    output = render_query_data_output(df)
    assert "Query result (3 rows)" in output
    assert "showing first" not in output


def test_render_column_values_uses_central_limit() -> None:
    counts = pd.Series({f"v{i}": i for i in range(40)})
    output = render_column_values_output("col", values=list(counts.index), value_counts=counts)
    assert f"... and {40 - DEFAULT_LLM_ROW_LIMIT} more values" in output

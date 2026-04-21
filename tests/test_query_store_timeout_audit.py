"""Tests for SQLiteQueryStore wall-clock timeout and per-query audit log."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd
import pytest

from src.query_store import QueryTimeoutError, SQLiteQueryStore


def _make_store(tmp_path: Path, **kwargs) -> SQLiteQueryStore:
    return SQLiteQueryStore(tmp_path / "session.sqlite", **kwargs)


def _audit_rows(store: SQLiteQueryStore) -> list[tuple]:
    with sqlite3.connect(store.db_path) as conn:
        return conn.execute(
            "SELECT id, sql, duration_ms, row_count, error FROM _query_audit ORDER BY id"
        ).fetchall()


def test_audit_table_created_on_init(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    with sqlite3.connect(store.db_path) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "_query_audit" in names


def test_query_completes_within_timeout(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    df = store.execute_query("SELECT 1 AS value")

    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["value"] == 1

    rows = _audit_rows(store)
    assert len(rows) == 1
    _id, sql, duration_ms, row_count, error = rows[0]
    assert "SELECT 1" in sql
    assert duration_ms is not None and duration_ms >= 0
    assert row_count == 1
    assert error is None


def test_query_times_out(tmp_path: Path) -> None:
    store = _make_store(tmp_path, query_timeout_seconds=0.2)
    slow_sql = (
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r) "
        "SELECT COUNT(*) FROM r LIMIT 1"
    )

    start = time.monotonic()
    with pytest.raises(QueryTimeoutError):
        store.execute_query(slow_sql)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0

    rows = _audit_rows(store)
    assert len(rows) == 1
    _id, sql, duration_ms, row_count, error = rows[0]
    assert "RECURSIVE" in sql
    assert error is not None and error.lower().startswith("timeout")
    assert row_count is None


def test_audit_records_failed_query(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    with pytest.raises(Exception):
        store.execute_query("SELEKT * FROM nope")

    rows = _audit_rows(store)
    assert len(rows) == 1
    _id, _sql, _duration_ms, row_count, error = rows[0]
    assert row_count is None
    assert error is not None
    assert "SELEKT" in error or "syntax" in error.lower()


def test_audit_persists_across_calls(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    store.execute_query("SELECT 1")
    store.execute_query("SELECT 2")
    store.execute_query("SELECT 3")

    rows = _audit_rows(store)
    assert [r[0] for r in rows] == [1, 2, 3]


def test_list_tables_hides_audit(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    assert "_query_audit" not in store.list_tables()

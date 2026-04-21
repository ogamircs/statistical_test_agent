"""SQLite-backed persistence for raw uploaded data and analysis outputs."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .statistics.models import (
    canonical_result_as_dict,
    to_ab_test_summary,
    to_segment_analysis_failure,
)

_AUDIT_TABLE = "_query_audit"
_CHAT_HISTORY_TABLE = "_chat_history"
_DEFAULT_QUERY_TIMEOUT_SECONDS = 5.0
_PROGRESS_HANDLER_INTERVAL = 1000


class QueryTimeoutError(Exception):
    """Raised when an execute_query call exceeds its wall-clock deadline."""


def _normalize_sqlite_value(value: Any) -> Any:
    if is_dataclass(value):
        return json.dumps(asdict(value), sort_keys=True)
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _normalize_sqlite_value(value) for key, value in record.items()}


class SQLiteQueryStore:
    """Persist raw data and analysis outputs to a local SQLite database."""

    def __init__(
        self,
        db_path: str | Path,
        query_timeout_seconds: float = _DEFAULT_QUERY_TIMEOUT_SECONDS,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.query_timeout_seconds = float(query_timeout_seconds)
        self._initialize_database()

    def _initialize_database(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_AUDIT_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    executed_at TEXT NOT NULL,
                    sql TEXT NOT NULL,
                    duration_ms REAL,
                    row_count INTEGER,
                    error TEXT
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_CHAT_HISTORY_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL
                )
                """
            )

    def save_chat_message(self, role: str, content: str) -> None:
        """Append one chat message to the persisted history."""
        try:
            with sqlite3.connect(self.db_path) as connection:
                connection.execute(
                    f"INSERT INTO {_CHAT_HISTORY_TABLE} (created_at, role, content) VALUES (?, ?, ?)",
                    (
                        datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                        role,
                        content,
                    ),
                )
        except sqlite3.Error:
            # Persistence must not break the conversation.
            pass

    def load_chat_messages(self) -> List[Dict[str, str]]:
        """Return persisted chat history in insertion order."""
        try:
            with sqlite3.connect(self.db_path) as connection:
                rows = connection.execute(
                    f"SELECT role, content FROM {_CHAT_HISTORY_TABLE} ORDER BY id ASC"
                ).fetchall()
        except sqlite3.Error:
            return []
        return [{"role": row[0], "content": row[1]} for row in rows]

    def clear_chat_messages(self) -> None:
        """Wipe persisted chat history so resume starts fresh."""
        try:
            with sqlite3.connect(self.db_path) as connection:
                connection.execute(f"DELETE FROM {_CHAT_HISTORY_TABLE}")
        except sqlite3.Error:
            pass

    def _record_audit(
        self,
        sql: str,
        duration_ms: Optional[float],
        row_count: Optional[int],
        error: Optional[str],
    ) -> None:
        try:
            with sqlite3.connect(self.db_path) as connection:
                connection.execute(
                    f"INSERT INTO {_AUDIT_TABLE} (executed_at, sql, duration_ms, row_count, error)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                        sql,
                        duration_ms,
                        row_count,
                        error,
                    ),
                )
        except sqlite3.Error:
            # Audit must never break the caller.
            pass

    def save_raw_dataframe(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Raw-data persistence currently requires a pandas DataFrame.")

        normalized = df.copy()
        for column in normalized.columns:
            normalized[column] = normalized[column].map(_normalize_sqlite_value)

        with sqlite3.connect(self.db_path) as connection:
            normalized.to_sql("raw_data", connection, index=False, if_exists="replace")

    def save_segment_results(self, results: Iterable[Any]) -> None:
        rows = [
            _normalize_record(canonical_result_as_dict(result, include_legacy_aliases=True))
            for result in results
        ]
        frame = pd.DataFrame(rows)
        with sqlite3.connect(self.db_path) as connection:
            frame.to_sql(
                "analysis_segment_results",
                connection,
                index=False,
                if_exists="replace",
            )

    def save_summary(self, summary: Any) -> None:
        normalized = to_ab_test_summary(summary)
        summary_payload = asdict(normalized)
        summary_payload.pop("detailed_results", None)
        summary_payload.pop("segment_failures", None)

        with sqlite3.connect(self.db_path) as connection:
            pd.DataFrame([_normalize_record(summary_payload)]).to_sql(
                "analysis_summary",
                connection,
                index=False,
                if_exists="replace",
            )

    def save_segment_failures(self, failures: Iterable[Any]) -> None:
        rows = [
            _normalize_record(asdict(to_segment_analysis_failure(failure)))
            for failure in failures
        ]
        with sqlite3.connect(self.db_path) as connection:
            pd.DataFrame(rows).to_sql(
                "analysis_segment_failures",
                connection,
                index=False,
                if_exists="replace",
            )

    def list_tables(self) -> List[str]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        hidden = {_AUDIT_TABLE, _CHAT_HISTORY_TABLE}
        return [row[0] for row in rows if row[0] not in hidden]

    def describe_schema(self) -> str:
        lines: List[str] = []
        with sqlite3.connect(self.db_path) as connection:
            for table_name in self.list_tables():
                columns = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
                column_bits = ", ".join(f"{column[1]} {column[2]}" for column in columns)
                lines.append(f"{table_name}({column_bits})")
        return "\n".join(lines)

    def execute_query(
        self,
        sql: str,
        timeout_seconds: Optional[float] = None,
    ) -> pd.DataFrame:
        deadline_seconds = (
            float(timeout_seconds) if timeout_seconds is not None else self.query_timeout_seconds
        )
        connection = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        connection.execute("PRAGMA busy_timeout = 5000")

        deadline = time.monotonic() + deadline_seconds
        timed_out = {"value": False}

        def _progress() -> int:
            if time.monotonic() >= deadline:
                timed_out["value"] = True
                return 1
            return 0

        connection.set_progress_handler(_progress, _PROGRESS_HANDLER_INTERVAL)

        start = time.monotonic()
        try:
            df = pd.read_sql_query(sql, connection)
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000.0
            if timed_out["value"]:
                self._record_audit(
                    sql,
                    duration_ms,
                    None,
                    f"timeout after {deadline_seconds:.3f}s",
                )
                raise QueryTimeoutError(
                    f"query exceeded {deadline_seconds:.3f}s deadline"
                ) from exc
            self._record_audit(sql, duration_ms, None, str(exc))
            raise
        finally:
            connection.set_progress_handler(None, 0)
            connection.close()

        duration_ms = (time.monotonic() - start) * 1000.0
        self._record_audit(sql, duration_ms, int(len(df)), None)
        return df

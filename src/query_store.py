"""SQLite-backed persistence for raw uploaded data and analysis outputs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .statistics.models import (
    canonical_result_as_dict,
    to_ab_test_summary,
    to_segment_analysis_failure,
)


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

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _initialize_database(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")

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
        return [row[0] for row in rows]

    def describe_schema(self) -> str:
        lines: List[str] = []
        with sqlite3.connect(self.db_path) as connection:
            for table_name in self.list_tables():
                columns = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
                column_bits = ", ".join(f"{column[1]} {column[2]}" for column in columns)
                lines.append(f"{table_name}({column_bits})")
        return "\n".join(lines)

    def execute_query(self, sql: str) -> pd.DataFrame:
        connection = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            return pd.read_sql_query(sql, connection)
        finally:
            connection.close()

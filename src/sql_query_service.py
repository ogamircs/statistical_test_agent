"""Natural-language question answering via safe SQL over SQLite."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, List, Protocol

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from .agent_reporting import AgentUserFacingError
from .query_store import SQLiteQueryStore


@dataclass(frozen=True)
class SQLQueryPlan:
    sql: str
    rationale: str = ""


@dataclass
class SQLQueryAnswer:
    question: str
    answer_text: str
    sql: str
    source_tables: List[str]
    row_count: int
    data: pd.DataFrame


class SQLPlanner(Protocol):
    def generate_sql(self, question: str, schema: str) -> SQLQueryPlan:
        ...


class OpenAISQLPlanner:
    """Use the configured LLM to turn natural-language questions into SQL."""

    def __init__(self, llm: Any):
        self.llm = llm

    def generate_sql(self, question: str, schema: str) -> SQLQueryPlan:
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You translate user questions into SQLite SQL.\n"
                        "Rules:\n"
                        "- Return strict JSON only.\n"
                        "- JSON shape: {\"sql\": \"...\", \"rationale\": \"...\"}\n"
                        "- Use only SELECT statements or CTEs ending in SELECT.\n"
                        "- Only use tables/columns present in the schema.\n"
                        "- Prefer analysis tables for questions about effect sizes, p-values, power, or summaries.\n"
                        "- Prefer raw_data for row-level counts, filters, or raw metrics.\n"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Schema:\n{schema}\n\n"
                        "Examples:\n"
                        "- 'what is the total effect size for Premium?' -> query analysis_segment_results.total_effect filtered by segment\n"
                        "- 'how many treatment users are in Premium?' -> count rows in raw_data\n\n"
                        f"Question: {question}"
                    )
                ),
            ]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        payload = self._parse_json(content)
        sql = str(payload.get("sql", "")).strip()
        rationale = str(payload.get("rationale", "")).strip()
        if not sql:
            raise AgentUserFacingError(
                "SQL_GENERATION_FAILED",
                "Unable to generate a SQL query for that question.",
            )
        return SQLQueryPlan(sql=sql, rationale=rationale)

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)


class SQLQueryService:
    """Execute safe read-only SQL generated from user questions."""

    _FORBIDDEN_PATTERN = re.compile(
        r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|reindex|vacuum|replace|truncate)\b",
        flags=re.IGNORECASE,
    )
    _TABLE_PATTERN = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", flags=re.IGNORECASE)
    _LIMIT_PATTERN = re.compile(r"\blimit\b", flags=re.IGNORECASE)

    def __init__(
        self,
        *,
        query_store: SQLiteQueryStore,
        sql_planner: SQLPlanner,
        default_limit: int = 20,
    ):
        self.query_store = query_store
        self.sql_planner = sql_planner
        self.default_limit = default_limit

    def answer_question(self, question: str) -> SQLQueryAnswer:
        schema = self.query_store.describe_schema()
        plan = self.sql_planner.generate_sql(question, schema)
        sql = self._validate_and_limit(plan.sql)
        try:
            data = self.query_store.execute_query(sql)
        except Exception as exc:
            raise AgentUserFacingError(
                "SQL_EXECUTION_FAILED",
                f"Unable to answer that question from the current query store: {exc}",
            ) from exc

        source_tables = sorted(set(self._TABLE_PATTERN.findall(sql)))
        answer_text = self._build_answer_text(data)
        return SQLQueryAnswer(
            question=question,
            answer_text=answer_text,
            sql=sql,
            source_tables=source_tables,
            row_count=len(data),
            data=data,
        )

    def _validate_and_limit(self, sql: str) -> str:
        cleaned = sql.strip().rstrip(";")
        if not cleaned:
            raise AgentUserFacingError("INVALID_SQL_QUERY", "The generated SQL query was empty.")
        if not re.match(r"^(select|with)\b", cleaned, flags=re.IGNORECASE):
            raise AgentUserFacingError(
                "INVALID_SQL_QUERY",
                "Only read-only SELECT queries are allowed for data questions.",
            )
        if self._FORBIDDEN_PATTERN.search(cleaned):
            raise AgentUserFacingError(
                "INVALID_SQL_QUERY",
                "The generated SQL query included unsupported write or schema operations.",
            )
        if not self._LIMIT_PATTERN.search(cleaned):
            cleaned = f"{cleaned} LIMIT {self.default_limit}"
        return cleaned

    @staticmethod
    def _build_answer_text(data: pd.DataFrame) -> str:
        if data.empty:
            return "I couldn't find any matching rows for that question."
        if len(data) == 1 and len(data.columns) == 1:
            column = data.columns[0]
            return f"The result is `{data.iloc[0][column]}`."
        if len(data) == 1 and "segment" in data.columns and len(data.columns) >= 2:
            non_segment_columns: Sequence[str] = [col for col in data.columns if col != "segment"]
            if non_segment_columns:
                metric = non_segment_columns[0]
                return (
                    f"For `{data.iloc[0]['segment']}`, `{metric}` is "
                    f"`{data.iloc[0][metric]}`."
                )
        if len(data) == 1:
            return "I found 1 matching row."
        return f"I found {len(data)} matching rows."

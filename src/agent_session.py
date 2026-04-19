"""Session state and persistence helpers for the conversational agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd

from .query_store import SQLiteQueryStore
from .query_store_gc import run_startup_gc
from .sql_query_service import OpenAISQLPlanner, SQLQueryService
from .statistics.models import to_ab_test_summary


@dataclass
class AgentSessionState:
    """Mutable in-memory state for one chat session."""

    chat_history: List[Any] = field(default_factory=list)
    last_charts: Dict[str, Any] = field(default_factory=dict)
    last_results: Any = None
    last_summary: Any = None

    def clear_analysis_state(self) -> None:
        self.last_results = None
        self.last_summary = None
        self.last_charts = {}

    def clear_chat_history(self) -> None:
        self.chat_history = []


class AgentAnalysisSession:
    """Owns query persistence, SQL question answering, and transient analysis state."""

    def __init__(
        self,
        *,
        llm: Any = None,
        state: Optional[AgentSessionState] = None,
        query_store: Any = None,
        query_store_path: Optional[Path] = None,
        sql_planner: Any = None,
        data_question_service: Any = None,
    ) -> None:
        self.state = state or AgentSessionState()
        self.query_store_path = (
            Path(query_store_path)
            if query_store_path is not None
            else Path("output") / "query_store" / f"session-{uuid4().hex}.sqlite"
        )
        run_startup_gc(self.query_store_path.parent)
        self.query_store = query_store or SQLiteQueryStore(self.query_store_path)

        planner = sql_planner
        if planner is None and llm is not None:
            planner = OpenAISQLPlanner(llm)

        self.data_question_service = data_question_service
        if self.data_question_service is None and planner is not None:
            self.data_question_service = SQLQueryService(
                query_store=self.query_store,
                sql_planner=planner,
            )

    def persist_loaded_data(self, analyzer: Any) -> bool:
        """Persist the currently loaded raw dataframe to the query store when possible."""
        df = getattr(analyzer, "df", None)
        if not isinstance(df, pd.DataFrame):
            return False

        self.query_store.save_raw_dataframe(df)
        return True

    def persist_analysis_outputs(self, results: Any, summary: Any) -> None:
        """Persist latest result tables and structured summary."""
        normalized_summary = to_ab_test_summary(summary)
        self.query_store.save_segment_results(results)
        self.query_store.save_summary(normalized_summary)
        if normalized_summary.segment_failures:
            self.query_store.save_segment_failures(normalized_summary.segment_failures)

    def answer_data_question(self, question: str) -> Any:
        """Delegate natural-language data questions to the configured query service."""
        if self.data_question_service is None:
            raise RuntimeError("Data question service is not configured for this session.")
        return self.data_question_service.answer_question(question)

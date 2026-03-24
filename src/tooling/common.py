"""Shared helpers for modular agent tool registration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

from ..agent_reporting import AgentUserFacingError, render_tool_error

logger = logging.getLogger(__name__)


class AgentProtocol(Protocol):
    analyzer: Any
    data_question_service: Any
    visualizer: Any
    _last_results: Any
    _last_summary: Any
    _last_charts: Dict[str, Any]
    FILE_SIZE_THRESHOLD_MB: float

    def _load_data_with_backend(self, filepath: str) -> Tuple[Any, Dict[str, Any], str, float, bool, Optional[str]]:
        ...

    def _normalize_shape(self, info: Dict[str, Any]) -> Tuple[int, int]:
        ...

    def _get_active_analyzer(self) -> Any:
        ...

    def persist_loaded_data(self, analyzer: Any) -> bool:
        ...

    def persist_analysis_outputs(self, results: Any, summary: Any) -> None:
        ...


@dataclass
class ToolContext:
    agent: AgentProtocol

    def active_analyzer(self) -> Any:
        return self.agent._get_active_analyzer()

    @staticmethod
    def backend_label(analyzer: Any) -> str:
        analyzer_name = type(analyzer).__name__.lower()
        return "spark" if "spark" in analyzer_name else "pandas"

    def unsupported(self, operation: str, analyzer: Any) -> AgentUserFacingError:
        return AgentUserFacingError(
            "BACKEND_OPERATION_UNSUPPORTED",
            f"{operation} is not supported for the active backend ({self.backend_label(analyzer)}).",
        )

    def require_pandas_dataframe(self, analyzer: Any, operation: str) -> Any:
        df = getattr(analyzer, "df", None)
        if df is None:
            raise ValueError("No data loaded")
        if hasattr(df, "groupBy") and hasattr(df, "select"):
            raise self.unsupported(operation, analyzer)
        return df

    def handle_tool_error(
        self,
        *,
        operation: str,
        prefix: str,
        error: Exception,
        default_code: str = "TOOL_EXECUTION_FAILED",
        default_message: str = "Unable to complete that operation.",
    ) -> str:
        logger.exception("Tool failure: %s", operation)
        return render_tool_error(
            prefix,
            error,
            default_code=default_code,
            default_message=default_message,
        )

    def remember_analysis(self, results: Any, summary: Any) -> None:
        self.agent.persist_analysis_outputs(results, summary)
        self.agent._last_results = results
        self.agent._last_summary = summary

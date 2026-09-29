"""Server-Sent Events helpers and the tool-progress callback bridge."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

# Human-readable step labels for the progress list in the UI.
TOOL_LABELS: Dict[str, str] = {
    "load_csv": "Loading data",
    "load_and_auto_analyze": "Loading data and running the analysis",
    "set_column_mapping": "Setting column mapping",
    "set_group_labels": "Setting group labels",
    "configure_and_analyze": "Running the analysis",
    "auto_configure_and_analyze": "Auto-configuring and running the analysis",
    "run_ab_test": "Running the A/B test",
    "run_full_analysis": "Running the full analysis",
    "generate_charts": "Building charts",
    "show_distribution_chart": "Building the distribution chart",
    "get_data_summary": "Summarizing the data",
    "get_segment_distribution": "Summarizing segments",
    "get_column_values": "Reading column values",
    "calculate_statistics": "Calculating statistics",
    "query_data": "Filtering the data",
    "answer_data_question": "Querying the data",
    "plan_sample_size": "Planning sample size",
    "compute_ratio_metric": "Computing the ratio metric",
}


def tool_label(name: str) -> str:
    return TOOL_LABELS.get(name, name.replace("_", " ").capitalize())


def format_sse(event: str, data: Any) -> str:
    """Encode one SSE frame. ``data`` is JSON-encoded on a single line."""
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


Emit = Callable[[str, Dict[str, Any]], None]


class ToolProgressHandler(BaseCallbackHandler):
    """Forward tool start/end events of one agent run to ``emit``.

    LangChain calls these on the worker thread running the agent; ``emit``
    must be thread-safe (the API uses ``loop.call_soon_threadsafe``).
    """

    def __init__(self, emit: Emit) -> None:
        self._emit = emit
        self._names: Dict[UUID, str] = {}

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = str((serialized or {}).get("name") or kwargs.get("name") or "tool")
        self._names[run_id] = name
        self._emit("tool_start", {"id": str(run_id), "name": name, "label": tool_label(name)})

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        name = self._names.pop(run_id, "tool")
        text = str(getattr(output, "content", output))
        # Tools report handled failures as text carrying an error code.
        ok = "[error_code=" not in text
        self._emit("tool_end", {"id": str(run_id), "name": name, "ok": ok})

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        name = self._names.pop(run_id, "tool")
        self._emit("tool_end", {"id": str(run_id), "name": name, "ok": False})

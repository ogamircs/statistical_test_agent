"""Serialize Plotly figures for the browser and build charts on demand."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

import plotly.graph_objects as go

from src.statistics.chart_catalog import CHART_DEFINITIONS, build_chart_map, resolve_chart_keys

# Chart picker options exposed to the UI: catalog keys plus the group aliases
# understood by ``resolve_chart_keys``.
CHART_TYPE_OPTIONS: List[Dict[str, str]] = [
    {"key": "dashboard", "label": "Dashboard"},
    {"key": "all", "label": "All charts"},
    {"key": "bayesian", "label": "Bayesian (all)"},
] + [
    {"key": d.key, "label": d.key.replace("_", " ").capitalize()}
    for d in CHART_DEFINITIONS
    if d.key != "dashboard"
]


def chart_title(name: str, figure: go.Figure) -> str:
    title = getattr(getattr(figure.layout, "title", None), "text", None)
    if title:
        # Strip HTML markup Plotly titles sometimes carry (<b>, <br><sup>…).
        plain = title.split("<br>")[0]
        for tag in ("<b>", "</b>", "<i>", "</i>"):
            plain = plain.replace(tag, "")
        if plain.strip():
            return plain.strip()
    return name.replace("_", " ").title()


def serialize_charts(charts: Mapping[str, go.Figure]) -> List[Dict[str, Any]]:
    return [
        {"name": name, "title": chart_title(name, fig), "figure": json.loads(fig.to_json())}
        for name, fig in charts.items()
    ]


class NoAnalysisError(LookupError):
    """Raised when charts are requested before any analysis has run."""


class UnknownChartTypeError(ValueError):
    pass


def build_charts_for_agent(agent: Any, chart_type: str) -> List[Dict[str, Any]]:
    """Build charts straight from the agent's last results (no LLM call)."""
    results = getattr(agent, "_last_results", None)
    summary = getattr(agent, "_last_summary", None)
    if not results or summary is None:
        raise NoAnalysisError("Run an analysis before requesting charts.")
    keys = resolve_chart_keys(chart_type)
    if not keys:
        raise UnknownChartTypeError(f"Unknown chart type '{chart_type}'.")

    kwargs: Dict[str, Any] = {}
    if "distribution" in keys:
        analyzer = agent._get_active_analyzer()
        mapping = getattr(analyzer, "column_mapping", {}) or {}
        kwargs = {
            "df": getattr(analyzer, "df", None),
            "group_col": mapping.get("group"),
            "segment_col": mapping.get("segment"),
        }
    charts = build_chart_map(agent.visualizer, results, summary, keys, **kwargs)
    return serialize_charts(charts)

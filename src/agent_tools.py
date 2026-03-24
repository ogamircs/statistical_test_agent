"""Public tool-registry facade for the conversational agent."""

from __future__ import annotations

from typing import List

from langchain_core.tools import Tool

from .tooling import (
    create_analysis_tools,
    create_loading_tools,
    create_visualization_tools,
)
from .tooling.common import AgentProtocol, ToolContext


def create_agent_tools(agent: AgentProtocol) -> List[Tool]:
    """Create the public tool list while keeping implementation grouped by concern."""
    context = ToolContext(agent)
    return [
        *create_loading_tools(context),
        *create_analysis_tools(context),
        *create_visualization_tools(context),
    ]

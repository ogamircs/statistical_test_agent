"""Live LLM eval: actually invokes the agent and asserts tool calls.

Skipped unless STATAGENT_RUN_LIVE_EVAL=1 AND OPENAI_API_KEY is set.
Wire this into a nightly job, NOT every PR — each task burns tokens.
"""

from __future__ import annotations

import os
from typing import List

import pytest

from tests.eval.golden_tasks import GOLDEN_TASKS

_LIVE_FLAG = os.environ.get("STATAGENT_RUN_LIVE_EVAL", "").lower() in {"1", "true", "yes"}
_API_KEY = bool(os.environ.get("OPENAI_API_KEY"))


pytestmark = pytest.mark.skipif(
    not (_LIVE_FLAG and _API_KEY),
    reason=(
        "Live LLM eval gate. Set STATAGENT_RUN_LIVE_EVAL=1 and "
        "OPENAI_API_KEY to enable."
    ),
)


def _collect_tool_names(messages) -> List[str]:
    tools_called: List[str] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                tools_called.append(name)
    return tools_called


@pytest.mark.parametrize("task", GOLDEN_TASKS, ids=[t.name for t in GOLDEN_TASKS])
def test_agent_routes_to_expected_tool(task) -> None:
    from src.agent import ABTestingAgent

    agent = ABTestingAgent()
    # Drive the underlying graph directly so we can inspect tool_calls.
    state = agent.agent.invoke({"messages": [("human", task.user_message)]})
    called = _collect_tool_names(state.get("messages", []))

    for expected in task.expected_tools:
        assert expected in called, (
            f"expected `{expected}` to be invoked for task '{task.name}', "
            f"got {called}"
        )

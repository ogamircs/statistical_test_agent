"""The agent must request one tool call at a time.

Tools share mutable analyzer state, so parallel tool calls race (a live run
rendered a dashboard from stale results while run_full_analysis re-ran).
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

import src.agent as agent_module


class _RecordingModel(GenericFakeChatModel):
    bound_kwargs: list = []

    def bind_tools(self, tools: Any, **kwargs: Any):  # type: ignore[override]
        type(self).bound_kwargs.append(kwargs)
        return self


@tool
def _noop() -> str:
    """Do nothing."""
    return "ok"


def test_agent_model_calls_disable_parallel_tool_calls() -> None:
    _RecordingModel.bound_kwargs = []
    model = _RecordingModel(messages=iter([AIMessage(content="done")]))
    graph = create_agent(
        model, [_noop], system_prompt="x", middleware=[agent_module._sequential_tool_calls]
    )

    graph.invoke({"messages": [HumanMessage(content="hi")]})

    assert _RecordingModel.bound_kwargs, "model was never bound to tools"
    assert all(kw.get("parallel_tool_calls") is False for kw in _RecordingModel.bound_kwargs)


def test_abtesting_agent_wires_the_middleware(monkeypatch) -> None:
    captured: dict = {}

    def _fake_create_agent(_llm, _tools, system_prompt=None, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(agent_module, "create_agent", _fake_create_agent)
    agent_module.ABTestingAgent()

    assert agent_module._sequential_tool_calls in captured["middleware"]

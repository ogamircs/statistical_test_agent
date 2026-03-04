"""Unit tests for ABTestingAgent orchestration without live OpenAI calls."""

import types

import pytest
from langchain_core.messages import AIMessage

import src.agent as agent_module
from src.agent import ABTestingAgent


class _DummyGraphAgent:
    def invoke(self, _payload):
        return {"messages": [AIMessage(content="dummy response")]}

    async def ainvoke(self, _payload):
        return {"messages": [AIMessage(content="dummy async response")]}


@pytest.fixture
def stubbed_agent(monkeypatch):
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(
        agent_module,
        "create_react_agent",
        lambda _llm, _tools, prompt=None: _DummyGraphAgent(),
    )
    return ABTestingAgent()


def test_agent_run_tracks_history(stubbed_agent):
    response = stubbed_agent.run("hello")

    assert response == "dummy response"
    assert len(stubbed_agent.chat_history) == 2


def test_agent_has_expected_tools(stubbed_agent):
    tools = stubbed_agent._create_tools()
    names = {tool.name for tool in tools}

    expected = {
        "load_csv",
        "load_and_auto_analyze",
        "configure_and_analyze",
        "auto_configure_and_analyze",
        "generate_charts",
    }
    assert expected.issubset(names)


def test_clear_memory(stubbed_agent):
    stubbed_agent.run("hello")
    stubbed_agent.clear_memory()

    assert stubbed_agent.chat_history == []

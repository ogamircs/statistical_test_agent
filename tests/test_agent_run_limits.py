"""Run-loop guardrails: recursion cap, bounded history, LLM error mapping (TODO.md #75, #76, #80)."""

from __future__ import annotations

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

import src.agent as agent_module
from src.agent import ABTestingAgent
from src.config import Config


class _RecordingGraph:
    def __init__(self):
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append((payload, config))
        return {"messages": [AIMessage(content="ok")]}


class _RaisingGraph:
    def __init__(self, error: Exception):
        self.error = error

    def invoke(self, _payload, config=None):
        raise self.error


@pytest.fixture
def llm_kwargs(monkeypatch):
    captured: dict = {}

    def _fake_chat(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent_module, "ChatOpenAI", _fake_chat)
    monkeypatch.setattr(
        agent_module, "create_agent", lambda _llm, _tools, system_prompt=None: _RecordingGraph()
    )
    return captured


def _make_agent(tmp_path, **config_overrides) -> ABTestingAgent:
    return ABTestingAgent(
        config=Config(**config_overrides),
        query_store_path=str(tmp_path / "store.sqlite"),
    )


def test_llm_timeout_and_retries_come_from_config(llm_kwargs, tmp_path):
    _make_agent(tmp_path, llm_request_timeout_seconds=12.5, llm_max_retries=4)

    assert llm_kwargs["timeout"] == 12.5
    assert llm_kwargs["max_retries"] == 4


def test_run_passes_recursion_limit(llm_kwargs, tmp_path):
    agent = _make_agent(tmp_path, agent_recursion_limit=7)

    agent.run("hello")

    _payload, config = agent.agent.calls[-1]
    assert config == {"recursion_limit": 7}


def test_run_bounds_history_sent_to_model_but_keeps_full_history(llm_kwargs, tmp_path):
    agent = _make_agent(tmp_path, max_history_messages=4)
    for i in range(5):
        agent.chat_history.append(HumanMessage(content=f"q{i}"))
        agent.chat_history.append(AIMessage(content=f"a{i}"))

    agent.run("latest")

    payload, _config = agent.agent.calls[-1]
    sent = payload["messages"]
    assert len(sent) <= 4
    assert isinstance(sent[0], HumanMessage)
    assert sent[-1].content == "latest"
    # Full history is retained locally (10 prior + new human + new AI).
    assert len(agent.chat_history) == 12


def test_history_window_never_starts_on_ai_message(llm_kwargs, tmp_path):
    agent = _make_agent(tmp_path, max_history_messages=4)
    agent.chat_history.extend(
        [HumanMessage(content="q0"), AIMessage(content="a0"), HumanMessage(content="q1"),
         AIMessage(content="a1"), HumanMessage(content="q2")]
    )

    window = agent._model_bound_history()

    # The raw last-4 slice starts on "a0"; it must be advanced to "q1".
    assert [m.content for m in window] == ["q1", "a1", "q2"]


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (GraphRecursionError("too many steps"), "AGENT_STEP_LIMIT_REACHED"),
        (
            openai.RateLimitError(
                "slow down", response=httpx.Response(429, request=_request()), body=None
            ),
            "LLM_RATE_LIMITED",
        ),
        (openai.APITimeoutError(request=_request()), "LLM_TIMEOUT"),
        (
            openai.AuthenticationError(
                "bad key", response=httpx.Response(401, request=_request()), body=None
            ),
            "LLM_AUTH_FAILED",
        ),
        (openai.APIConnectionError(request=_request()), "LLM_UNAVAILABLE"),
        (RuntimeError("boom"), "AGENT_EXECUTION_FAILED"),
    ],
)
def test_run_maps_llm_failures_to_distinct_codes(llm_kwargs, tmp_path, error, code):
    agent = _make_agent(tmp_path)
    agent.agent = _RaisingGraph(error)

    result = agent.run("hello")

    assert result.startswith("Error processing request:")
    assert f"[error_code={code}]" in result


def test_config_from_env_reads_new_agent_knobs():
    cfg = Config.from_env(
        {
            "STATAGENT_LLM_TIMEOUT_SECONDS": "30",
            "STATAGENT_LLM_MAX_RETRIES": "0",
            "STATAGENT_AGENT_RECURSION_LIMIT": "40",
            "STATAGENT_MAX_HISTORY_MESSAGES": "10",
        }
    )
    assert cfg.llm_request_timeout_seconds == 30.0
    assert cfg.llm_max_retries == 0
    assert cfg.agent_recursion_limit == 40
    assert cfg.max_history_messages == 10


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_request_timeout_seconds": 0},
        {"llm_max_retries": -1},
        {"agent_recursion_limit": 1},
        {"max_history_messages": 0},
    ],
)
def test_config_validate_rejects_bad_agent_knobs(overrides):
    with pytest.raises(ValueError):
        Config(**overrides).validate()

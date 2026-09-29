"""ABTestingAgent.run(on_token=...) streams model text (TODO.md #86)."""

from __future__ import annotations

import json
import re
from typing import Any, List

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool

import src.agent as agent_module
from src.agent import ABTestingAgent, _chunk_text


class _ToolableFake(GenericFakeChatModel):
    """Fake chat model that streams like a real provider.

    GenericFakeChatModel's stream drops tool calls, so emit the text word by
    word and the tool calls as tool_call_chunks on a final chunk.
    """

    def bind_tools(self, tools: Any, **kwargs: Any):  # type: ignore[override]
        return self

    def _stream(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any):
        message = next(self.messages)
        words = re.split(r"(\s)", str(message.content))
        for word in (w for w in words if w):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word))
            if run_manager:
                run_manager.on_llm_new_token(word, chunk=chunk)
            yield chunk
        if message.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                        for i, c in enumerate(message.tool_calls)
                    ],
                )
            )


@tool
def lookup() -> str:
    """Return a fixed value."""
    return "42"


def _agent_with(monkeypatch, tmp_path, replies: List[AIMessage]) -> ABTestingAgent:
    model = _ToolableFake(messages=iter(replies))
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **_kwargs: model)
    monkeypatch.setattr(
        agent_module,
        "create_agent",
        lambda llm, _tools, system_prompt=None, **kwargs: create_agent(
            llm, [lookup], system_prompt=system_prompt, **kwargs
        ),
    )
    return ABTestingAgent(query_store_path=str(tmp_path / "s.sqlite"))


def test_run_streams_final_answer_tokens_and_returns_full_text(monkeypatch, tmp_path) -> None:
    agent = _agent_with(monkeypatch, tmp_path, [AIMessage(content="Premium had the largest effect.")])
    tokens: List[str] = []

    response = agent.run("which segment?", on_token=tokens.append)

    assert response == "Premium had the largest effect."
    assert len(tokens) > 1, "answer should arrive in several chunks"
    assert "".join(tokens) == response
    # History and persistence are unchanged by streaming.
    assert agent.chat_history[-1].content == response
    assert agent.session.query_store.load_chat_messages()[-1]["content"] == response


def test_tool_calls_and_tool_output_are_not_streamed_as_text(monkeypatch, tmp_path) -> None:
    # A tool turn may carry a short preamble; that text streams (the UI resets
    # its live buffer on tool_start), but tool-call args and tool output never do.
    tool_turn = AIMessage(
        content="Checking.", tool_calls=[{"name": "lookup", "args": {"q": "zz"}, "id": "call_1"}]
    )
    agent = _agent_with(monkeypatch, tmp_path, [tool_turn, AIMessage(content="The answer is 42.")])
    tokens: List[str] = []

    response = agent.run("look it up", on_token=tokens.append)

    assert response == "The answer is 42."
    streamed = "".join(tokens)
    assert streamed == "Checking.The answer is 42."
    assert "zz" not in streamed and "lookup" not in streamed


def test_run_without_on_token_is_unchanged(monkeypatch, tmp_path) -> None:
    agent = _agent_with(monkeypatch, tmp_path, [AIMessage(content="plain")])
    assert agent.run("hi") == "plain"


def test_chunk_text_ignores_tool_call_chunks_and_non_text_blocks() -> None:
    assert _chunk_text(AIMessageChunk(content="hi")) == "hi"
    assert _chunk_text(
        AIMessageChunk(content="", tool_call_chunks=[{"name": "x", "args": "{", "id": "1", "index": 0}])
    ) == ""
    assert _chunk_text(AIMessageChunk(content=[{"type": "text", "text": "a"}, {"type": "reasoning", "text": "b"}])) == "a"
    assert _chunk_text("not a chunk") == ""

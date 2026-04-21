"""Persisted chat history round-trip across SQLiteQueryStore reopen."""

from __future__ import annotations

from pathlib import Path

from src.query_store import SQLiteQueryStore


def test_chat_history_persists_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "session.sqlite"
    store = SQLiteQueryStore(db_path)

    store.save_chat_message("human", "load my CSV")
    store.save_chat_message("ai", "Loaded. 1000 rows, 4 columns.")
    store.save_chat_message("human", "best guess analyze it")

    reopened = SQLiteQueryStore(db_path)
    history = reopened.load_chat_messages()

    assert [m["role"] for m in history] == ["human", "ai", "human"]
    assert history[1]["content"] == "Loaded. 1000 rows, 4 columns."


def test_chat_history_table_hidden_from_list_tables(tmp_path: Path) -> None:
    store = SQLiteQueryStore(tmp_path / "session.sqlite")
    store.save_chat_message("human", "ping")
    assert "_chat_history" not in store.list_tables()


def test_load_chat_messages_empty_returns_empty_list(tmp_path: Path) -> None:
    store = SQLiteQueryStore(tmp_path / "session.sqlite")
    assert store.load_chat_messages() == []


def test_agent_init_rehydrates_chat_history(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    from src.agent import ABTestingAgent

    db_path = str(tmp_path / "session-fixture.sqlite")
    store = SQLiteQueryStore(db_path)
    store.save_chat_message("human", "hello")
    store.save_chat_message("ai", "hi back")

    agent = ABTestingAgent(query_store_path=db_path)
    assert len(agent.chat_history) == 2
    assert agent.chat_history[0].content == "hello"
    assert agent.chat_history[1].content == "hi back"

"""Analysis state survives a server restart (TODO.md #104).

A "restart" is a fresh ABTestingAgent / API app over the same session store:
the dataframe, mapping, labels, results and latest charts must come back
without re-uploading, and only chat text used to.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.agent as agent_module
from src.agent import ABTestingAgent
from src.api import create_app
from src.config import Config

SAMPLE_CSV = str(Path(__file__).resolve().parent.parent / "data" / "sample_ab_data.csv")


class _NoGraph:
    def invoke(self, *_args, **_kwargs):
        raise AssertionError("no LLM calls in restart tests")


@pytest.fixture(autouse=True)
def _offline_agent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(agent_module, "create_agent", lambda *_args, **_kwargs: _NoGraph())


def _tool(agent: ABTestingAgent, name: str):
    return next(tool for tool in agent._create_tools() if tool.name == name)


def _analyze(agent: ABTestingAgent, ratio: float | None = 0.6) -> None:
    _tool(agent, "load_and_auto_analyze").func(SAMPLE_CSV)
    if ratio is not None:
        _tool(agent, "set_column_mapping").func(expected_treatment_ratio=ratio)
        _tool(agent, "run_full_analysis").func("")


def test_agent_restores_data_mapping_labels_and_results(tmp_path) -> None:
    store = str(tmp_path / "session.sqlite")
    before = ABTestingAgent(query_store_path=store)
    _analyze(before)
    expected = [(r.segment, r.is_significant) for r in before._last_results or []]

    after = ABTestingAgent(query_store_path=store)  # simulated restart

    analyzer = after._get_active_analyzer()
    assert analyzer.df is not None and len(analyzer.df) == 5000
    assert analyzer.column_mapping["expected_treatment_ratio"] == pytest.approx(0.6)
    assert (analyzer.treatment_label, analyzer.control_label) == ("treatment", "control")
    assert [(r.segment, r.is_significant) for r in after._last_results or []] == expected
    assert after._last_summary is not None


def test_new_upload_supersedes_pending_restore(tmp_path) -> None:
    store = str(tmp_path / "session.sqlite")
    _analyze(ABTestingAgent(query_store_path=store))

    after = ABTestingAgent(query_store_path=store)
    after._load_data(SAMPLE_CSV)  # user loads data before anything triggered the restore

    assert after._pending_analysis_state is None
    assert after._last_results is None  # nothing stale resurrected


def test_corrupt_state_falls_back_to_chat_only(tmp_path) -> None:
    store = str(tmp_path / "session.sqlite")
    _analyze(ABTestingAgent(query_store_path=store))
    with closing(sqlite3.connect(store)) as conn, conn:
        conn.execute("UPDATE _session_state SET value = '{not json' WHERE key = 'analysis'")

    after = ABTestingAgent(query_store_path=store)

    assert after._last_results is None
    assert after._get_active_analyzer().df is None


def test_state_table_is_hidden_from_the_sql_planner(tmp_path) -> None:
    agent = ABTestingAgent(query_store_path=str(tmp_path / "session.sqlite"))
    _analyze(agent, ratio=None)
    assert "_session_state" not in agent.session.query_store.list_tables()
    assert "raw_data" in agent.session.query_store.list_tables()


def _app(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(
            config=Config(),
            store_dir=tmp_path / "store",
            uploads_dir=tmp_path / "uploads",
            frontend_dist=tmp_path / "no-dist",
        )
    )


def test_api_restart_restores_charts_and_analysis(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("STATAGENT_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("STATAGENT_AUTH_PASSWORD", raising=False)
    first = _app(tmp_path)
    session_id = first.post("/api/sessions").json()["id"]
    agent = first.app.state.registry.get(session_id).agent  # type: ignore[attr-defined]
    agent.session.query_store.save_chat_message("human", "analyze it")
    _analyze(agent)
    chosen = first.get(f"/api/sessions/{session_id}/charts", params={"type": "bayesian"})
    assert chosen.status_code == 200

    second = _app(tmp_path)  # simulated server restart

    reopened = second.get(f"/api/sessions/{session_id}/messages").json()
    assert [c["name"] for c in reopened["charts"]] == [c["name"] for c in chosen.json()["charts"]]
    dashboard = second.get(f"/api/sessions/{session_id}/charts", params={"type": "dashboard"})
    assert dashboard.status_code == 200, dashboard.text
    assert "T-test sig: 3/5" in dashboard.json()["charts"][0]["figure"]["layout"]["title"]["text"]


# -- PR #10 review round 2 ----------------------------------------------------

ALT_CSV = str(Path(__file__).resolve().parent.parent / "data" / "sample_ab_data_alt.csv")


def test_load_csv_invalidates_persisted_analysis(tmp_path) -> None:
    # An inspection-only load replaces raw_data; a restart must not combine
    # the new dataframe with the previous dataset's mapping and labels.
    store = str(tmp_path / "session.sqlite")
    before = ABTestingAgent(query_store_path=store)
    _analyze(before)
    version = before.analysis_version
    _tool(before, "load_csv").func(ALT_CSV)

    assert before.analysis_version > version
    assert before._last_results is None
    assert before.session.query_store.load_state("analysis") is None

    after = ABTestingAgent(query_store_path=store)
    assert after._last_results is None
    assert after._get_active_analyzer().df is None  # chat-only, nothing mixed


def test_restart_replays_a_single_segment_analysis(tmp_path) -> None:
    store = str(tmp_path / "session.sqlite")
    before = ABTestingAgent(query_store_path=store)
    _analyze(before, ratio=None)
    _tool(before, "run_ab_test").func(segment="Premium")
    [expected] = before._last_results or []

    after = ABTestingAgent(query_store_path=store)
    restored = after._last_results or []

    assert [r.segment for r in restored] == ["Premium"]
    assert restored[0].p_value == pytest.approx(expected.p_value)


def test_restart_replays_an_overall_only_analysis(tmp_path) -> None:
    store = str(tmp_path / "session.sqlite")
    before = ABTestingAgent(query_store_path=store)
    _analyze(before, ratio=None)
    _tool(before, "run_ab_test").func()

    after = ABTestingAgent(query_store_path=store)

    assert [r.segment for r in after._last_results or []] == ["Overall"]


def test_new_results_invalidate_pending_charts(tmp_path) -> None:
    agent = ABTestingAgent(query_store_path=str(tmp_path / "session.sqlite"))
    _analyze(agent, ratio=None)
    _tool(agent, "generate_charts").func("dashboard")
    assert agent.get_charts()
    version = agent.analysis_version

    _tool(agent, "run_full_analysis").func("")

    assert agent.analysis_version > version
    assert agent.get_charts() == {}  # charts of the replaced results are gone


def test_agent_can_load_uploads_from_a_custom_uploads_dir(monkeypatch, tmp_path) -> None:
    # Self-review (PR #10): uploads stored in a non-default uploads_dir were
    # outside the agent's allowed data roots, so the agent could not load them.
    monkeypatch.setattr("src.data_paths.tempfile.gettempdir", lambda: str(tmp_path / "sys-tmp"))
    monkeypatch.chdir(tmp_path)
    client = TestClient(
        create_app(
            config=Config(),
            store_dir=tmp_path / "store",
            uploads_dir=tmp_path / "custom-uploads",
            frontend_dist=tmp_path / "no-dist",
        )
    )
    session_id = client.post("/api/sessions").json()["id"]
    with open(SAMPLE_CSV, "rb") as handle:
        file_id = client.post(
            f"/api/sessions/{session_id}/upload", files={"file": ("exp.csv", handle, "text/csv")}
        ).json()["file_id"]

    registry = client.app.state.registry  # type: ignore[attr-defined]
    agent = registry.get(session_id).agent
    _, info, _ = agent._load_data(str(registry.resolve_upload(session_id, file_id)))

    assert info["shape"][0] == 5000


# -- PR #10 review round 3 ---------------------------------------------------


def test_invalidation_also_drops_the_persisted_chart_snapshot(tmp_path) -> None:
    from src.query_store import LATEST_CHARTS_STATE_KEY

    store = str(tmp_path / "session.sqlite")
    agent = ABTestingAgent(query_store_path=store)
    _analyze(agent)
    agent.session.query_store.save_state(LATEST_CHARTS_STATE_KEY, [{"name": "dashboard"}])

    agent._load_data(SAMPLE_CSV)  # new data; process could die before charts finalize

    restarted = ABTestingAgent(query_store_path=store)
    assert restarted.session.query_store.load_state(LATEST_CHARTS_STATE_KEY) is None


def test_replay_state_is_not_saved_when_raw_data_persistence_fails(tmp_path, monkeypatch) -> None:
    store = str(tmp_path / "session.sqlite")
    _analyze(ABTestingAgent(query_store_path=store))  # store now holds dataset #1 + its state

    agent = ABTestingAgent(query_store_path=store)
    monkeypatch.setattr(
        agent.session, "persist_loaded_data", lambda _analyzer: (_ for _ in ()).throw(RuntimeError("too wide"))
    )
    _tool(agent, "load_and_auto_analyze").func(SAMPLE_CSV)  # analysis succeeds in memory

    assert agent._last_results
    # The stale raw_data table must not be paired with this analysis on restart.
    restarted = ABTestingAgent(query_store_path=store)
    assert restarted._pending_analysis_state is None

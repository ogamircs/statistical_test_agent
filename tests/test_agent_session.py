"""Tests for the extracted agent session abstractions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.agent_session import AgentAnalysisSession, AgentSessionState
from src.statistics.models import ABTestSummary, SegmentAnalysisFailure


class _FakeStore:
    def __init__(self):
        self.raw_frames = []
        self.segment_results = []
        self.summaries = []
        self.failures = []

    def save_raw_dataframe(self, df):
        self.raw_frames.append(df.copy())

    def save_segment_results(self, results):
        self.segment_results.append(results)

    def save_summary(self, summary):
        self.summaries.append(summary)

    def save_segment_failures(self, failures):
        self.failures.append(failures)


class _FakeQuestionService:
    def __init__(self):
        self.questions = []

    def answer_question(self, question):
        self.questions.append(question)
        return {"answer_text": "ok"}


class _FakeAnalyzer:
    def __init__(self, df):
        self.df = df


def test_session_state_tracks_and_clears_chart_state() -> None:
    state = AgentSessionState()

    state.last_results = ["result"]
    state.last_summary = {"summary": True}
    state.last_charts["dashboard"] = "figure"

    state.clear_analysis_state()

    assert state.last_results is None
    assert state.last_summary is None
    assert state.last_charts == {}


def test_agent_analysis_session_persists_raw_dataframe() -> None:
    store = _FakeStore()
    session = AgentAnalysisSession(query_store=store, data_question_service=_FakeQuestionService())
    analyzer = _FakeAnalyzer(pd.DataFrame({"value": [1, 2, 3]}))

    persisted = session.persist_loaded_data(analyzer)

    assert persisted is True
    assert len(store.raw_frames) == 1


def test_agent_analysis_session_persists_results_summary_and_failures() -> None:
    store = _FakeStore()
    session = AgentAnalysisSession(query_store=store, data_question_service=_FakeQuestionService())
    summary = ABTestSummary(
        total_segments_analyzed=1,
        segment_failures=[
            SegmentAnalysisFailure(segment="Premium", error="boom"),
        ],
    )

    session.persist_analysis_outputs(["result"], summary)

    assert store.segment_results == [["result"]]
    assert len(store.summaries) == 1
    assert len(store.failures) == 1


def test_agent_analysis_session_delegates_data_questions() -> None:
    question_service = _FakeQuestionService()
    session = AgentAnalysisSession(query_store=_FakeStore(), data_question_service=question_service)

    result = session.answer_data_question("What is the total effect size?")

    assert result == {"answer_text": "ok"}
    assert question_service.questions == ["What is the total effect size?"]


def test_agent_analysis_session_uses_default_query_store_path(tmp_path: Path) -> None:
    session = AgentAnalysisSession(query_store_path=tmp_path / "session.sqlite", sql_planner=None)

    assert session.query_store is not None
    assert session.query_store_path == tmp_path / "session.sqlite"

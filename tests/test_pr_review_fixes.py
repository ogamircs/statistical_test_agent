"""Regression tests for PR #5 review feedback."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.config import Config
from src.query_store import SQLiteQueryStore
from src.statistics.ratio_metric import delta_method_ratio_test

# ---------------------------------------------------------------------------
# P1: joint null mask on numerator+denominator
# ---------------------------------------------------------------------------

def test_ratio_tool_drops_rows_with_null_in_either_column(tmp_path) -> None:
    os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

    from src.statistics.analyzer import ABTestAnalyzer
    from src.tooling.analysis import _ratio_metric_impl
    from src.tooling.common import ToolContext

    df = pd.DataFrame(
        {
            "experiment_group": ["treatment"] * 6 + ["control"] * 6,
            "revenue": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 2.0, np.nan, 6.0, 8.0, 10.0, 12.0],
            "sessions": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, np.nan, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    )
    bound_analyzer = ABTestAnalyzer()
    bound_analyzer.set_dataframe(df)
    bound_analyzer.set_column_mapping({"group": "experiment_group", "effect_value": "revenue"})
    bound_analyzer.set_group_labels("treatment", "control")

    class _StubAgent:
        analyzer = bound_analyzer
        data_question_service = None
        visualizer = None
        _last_results = None
        _last_summary = None
        _last_charts = {}
        FILE_SIZE_THRESHOLD_MB = 2.0

        def _load_data_with_backend(self, filepath): raise NotImplementedError
        def _normalize_shape(self, info): raise NotImplementedError
        def _get_active_analyzer(self): return bound_analyzer
        def persist_loaded_data(self, x): return False
        def persist_analysis_outputs(self, r, s): return None

    ctx = ToolContext(_StubAgent())
    output = _ratio_metric_impl(
        ctx, json.dumps({"numerator": "revenue", "denominator": "sessions"})
    )
    # Treatment: 6 rows, 2 have a null in revenue and 1 has a null in
    # sessions (disjoint rows). Joint mask drops 3 rows → 3 remain.
    assert "(n=3)" in output
    # Control: 6 rows, 1 null in revenue (idx 1) and 1 null in sessions
    # (idx 0) on different rows. Joint mask drops both → 4 remain.
    assert "(n=4)" in output


# ---------------------------------------------------------------------------
# P1: one-arm zero variance still yields a valid test
# ---------------------------------------------------------------------------

def test_one_arm_zero_variance_still_runs_test() -> None:
    # Control arm has zero variance (constant values); treatment has signal.
    # Historically this returned p=1 via a short-circuit; test must still run.
    out = delta_method_ratio_test(
        treatment_numerator=np.array([10.0, 12.0, 8.0, 14.0, 11.0]),
        treatment_denominator=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        control_numerator=np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
        control_denominator=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    assert out.standard_error > 0
    assert out.reason is None or out.reason != "zero_variance_arm"


def test_both_arms_defined_but_combined_se_zero_bails() -> None:
    # Genuinely degenerate: both arms identical constants.
    out = delta_method_ratio_test(
        treatment_numerator=np.array([5.0, 5.0, 5.0]),
        treatment_denominator=np.array([1.0, 1.0, 1.0]),
        control_numerator=np.array([5.0, 5.0, 5.0]),
        control_denominator=np.array([1.0, 1.0, 1.0]),
    )
    assert out.is_significant is False


def test_undefined_ratio_when_denominator_mean_zero() -> None:
    out = delta_method_ratio_test(
        treatment_numerator=np.array([1.0, 2.0, 3.0]),
        treatment_denominator=np.zeros(3),
        control_numerator=np.array([1.0, 2.0, 3.0]),
        control_denominator=np.array([1.0, 1.0, 1.0]),
    )
    assert out.reason == "undefined_ratio"
    assert out.is_significant is False


# ---------------------------------------------------------------------------
# P1: plan_sample_size labels treatment/control separately
# ---------------------------------------------------------------------------

def test_plan_sample_size_reports_both_arms_for_asymmetric_allocation() -> None:
    from src.tooling.analysis import _plan_sample_size_impl

    payload = json.dumps(
        {
            "metric_type": "continuous",
            "baseline_mean": 100.0,
            "baseline_std": 20.0,
            "mde": 5.0,
            "ratio": 2.0,
        }
    )
    output = _plan_sample_size_impl(payload)
    assert "Treatment arm" in output
    assert "Control arm" in output
    # Control arm under 2:1 allocation should be roughly 2x the treatment
    # arm count; the previous "per arm" phrasing masked this.
    assert "ratio=2" in output


# ---------------------------------------------------------------------------
# P2: Config values thread into SQLiteQueryStore / SQLQueryService
# ---------------------------------------------------------------------------

def test_agent_config_threads_query_timeout_and_sql_limit(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    from src.agent import ABTestingAgent

    config = Config(
        query_timeout_seconds=0.75,
        sql_default_row_limit=7,
    )
    agent = ABTestingAgent(
        config=config,
        query_store_path=str(tmp_path / "session.sqlite"),
    )
    assert agent.session.query_store.query_timeout_seconds == 0.75
    assert agent.session.data_question_service is None or (
        getattr(agent.session.data_question_service, "default_limit", None) == 7
    )


# ---------------------------------------------------------------------------
# P2: clear_memory wipes persisted chat history
# ---------------------------------------------------------------------------

def test_clear_memory_wipes_persisted_chat(tmp_path) -> None:
    db = tmp_path / "session.sqlite"
    store = SQLiteQueryStore(db)
    store.save_chat_message("human", "hello")
    store.save_chat_message("ai", "hi")
    assert len(store.load_chat_messages()) == 2

    store.clear_chat_messages()
    assert store.load_chat_messages() == []


def test_agent_clear_memory_also_clears_persisted_history(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    from src.agent import ABTestingAgent

    db_path = str(tmp_path / "session.sqlite")
    store = SQLiteQueryStore(db_path)
    store.save_chat_message("human", "hello")
    store.save_chat_message("ai", "hi")

    agent = ABTestingAgent(query_store_path=db_path)
    assert len(agent.chat_history) == 2
    agent.clear_memory()
    assert agent.chat_history == []

    # Reload fresh agent from same path — persisted history should be gone.
    reloaded = ABTestingAgent(query_store_path=db_path)
    assert reloaded.chat_history == []

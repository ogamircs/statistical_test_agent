"""Tests for the plan_sample_size agent tool."""

from __future__ import annotations

import json
import math

from src.agent_tools import create_agent_tools
from src.statistics.power_analysis import calculate_required_sample_size
from src.tooling.analysis import _plan_sample_size_impl


def test_plan_sample_size_proportion_returns_markdown() -> None:
    payload = json.dumps(
        {
            "metric_type": "proportion",
            "baseline_rate": 0.10,
            "mde": 0.02,
        }
    )

    output = _plan_sample_size_impl(payload)

    p1, p2 = 0.10, 0.12
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    expected = calculate_required_sample_size(
        effect_size=h,
        ratio=1.0,
        power_threshold=0.8,
        significance_level=0.05,
    )

    assert "Required sample size per arm" in output
    assert (str(expected) in output) or (f"{expected:,}" in output)


def test_plan_sample_size_continuous_returns_markdown() -> None:
    payload = json.dumps(
        {
            "metric_type": "continuous",
            "baseline_mean": 100.0,
            "baseline_std": 20.0,
            "mde": 5.0,
        }
    )

    output = _plan_sample_size_impl(payload)

    expected = calculate_required_sample_size(
        effect_size=5.0 / 20.0,
        ratio=1.0,
        power_threshold=0.8,
        significance_level=0.05,
    )

    assert "Required sample size per arm" in output
    assert (str(expected) in output) or (f"{expected:,}" in output)


def test_plan_sample_size_missing_required_returns_structured_error() -> None:
    payload = json.dumps({"baseline_rate": 0.1, "mde": 0.02})

    output = _plan_sample_size_impl(payload)

    assert "error_code=" in output
    assert "metric_type" in output


def test_plan_sample_size_invalid_json_returns_structured_error() -> None:
    output = _plan_sample_size_impl("not-json")
    assert "error_code=" in output


def test_tool_registered() -> None:
    class _StubAgent:
        analyzer = None
        data_question_service = None
        visualizer = None
        _last_results = None
        _last_summary = None
        _last_charts: dict = {}
        FILE_SIZE_THRESHOLD_MB = 2.0

        def _load_data_with_backend(self, filepath):
            raise NotImplementedError

        def _normalize_shape(self, info):
            raise NotImplementedError

        def _get_active_analyzer(self):
            return None

        def persist_loaded_data(self, analyzer):
            return False

        def persist_analysis_outputs(self, results, summary):
            return None

    tools = create_agent_tools(_StubAgent())
    assert any(getattr(tool, "name", None) == "plan_sample_size" for tool in tools)

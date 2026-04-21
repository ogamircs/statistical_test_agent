"""Delta-method ratio metric tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.statistics.ratio_metric import delta_method_ratio_test


def _draw_arm(rng, n: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    denominator = rng.poisson(lam=8.0, size=n) + 1  # always >= 1 to avoid div-by-zero
    base_mean = ratio * denominator.astype(float)
    noise = rng.normal(0.0, 0.5, size=n)
    numerator = np.maximum(base_mean + noise, 0.0)
    return numerator, denominator


def test_no_signal_does_not_reject() -> None:
    rng = np.random.default_rng(42)
    n_t, n_c = (1000, 1000)
    treatment_num, treatment_den = _draw_arm(rng, n_t, ratio=2.0)
    control_num, control_den = _draw_arm(rng, n_c, ratio=2.0)

    out = delta_method_ratio_test(
        treatment_numerator=treatment_num,
        treatment_denominator=treatment_den,
        control_numerator=control_num,
        control_denominator=control_den,
    )

    assert out.is_significant is False
    assert out.p_value > 0.05
    assert abs(out.absolute_diff) < 0.1


def test_real_signal_is_detected() -> None:
    rng = np.random.default_rng(7)
    treatment_num, treatment_den = _draw_arm(rng, 2000, ratio=2.5)
    control_num, control_den = _draw_arm(rng, 2000, ratio=2.0)

    out = delta_method_ratio_test(
        treatment_numerator=treatment_num,
        treatment_denominator=treatment_den,
        control_numerator=control_num,
        control_denominator=control_den,
    )

    assert out.is_significant is True
    assert out.p_value < 0.001
    assert out.absolute_diff > 0.4
    assert out.confidence_interval[0] > 0  # CI does not cross zero


def test_insufficient_samples_marks_unusable() -> None:
    out = delta_method_ratio_test(
        treatment_numerator=np.array([1.0]),
        treatment_denominator=np.array([1.0]),
        control_numerator=np.array([1.0, 2.0]),
        control_denominator=np.array([1.0, 1.0]),
    )
    assert out.is_significant is False
    assert out.reason == "insufficient_sample_size"


def test_zero_denominator_returns_zero_ratio_safely() -> None:
    out = delta_method_ratio_test(
        treatment_numerator=np.array([1.0, 2.0, 3.0]),
        treatment_denominator=np.zeros(3),
        control_numerator=np.array([1.0, 2.0, 3.0]),
        control_denominator=np.array([1.0, 1.0, 1.0]),
    )
    assert out.treatment_ratio == 0.0
    assert out.is_significant is False


def test_compute_ratio_metric_tool_returns_markdown() -> None:
    import os

    os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

    from src.statistics.analyzer import ABTestAnalyzer
    from src.tooling.analysis import _ratio_metric_impl
    from src.tooling.common import ToolContext

    rng = np.random.default_rng(1)
    treatment_num, treatment_den = _draw_arm(rng, 1500, ratio=3.0)
    control_num, control_den = _draw_arm(rng, 1500, ratio=2.5)
    df = pd.DataFrame(
        {
            "experiment_group": ["treatment"] * 1500 + ["control"] * 1500,
            "revenue": np.concatenate([treatment_num, control_num]),
            "sessions": np.concatenate([treatment_den, control_den]),
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

        def _load_data_with_backend(self, filepath):
            raise NotImplementedError

        def _normalize_shape(self, info):
            raise NotImplementedError

        def _get_active_analyzer(self):
            return bound_analyzer

        def persist_loaded_data(self, x):
            return False

        def persist_analysis_outputs(self, results, summary):
            return None

    ctx = ToolContext(_StubAgent())
    payload = json.dumps({"numerator": "revenue", "denominator": "sessions"})
    output = _ratio_metric_impl(ctx, payload)

    assert "Ratio Metric" in output
    assert "Treatment ratio" in output
    assert "Significant (p < 0.05): YES" in output


def test_ratio_metric_tool_invalid_input_returns_structured_error() -> None:
    from src.tooling.analysis import _ratio_metric_impl
    from src.tooling.common import ToolContext

    class _NoopAgent:
        analyzer = None
        data_question_service = None
        visualizer = None
        _last_results = None
        _last_summary = None
        _last_charts = {}
        FILE_SIZE_THRESHOLD_MB = 2.0

        def _load_data_with_backend(self, filepath):
            raise NotImplementedError

        def _normalize_shape(self, info):
            raise NotImplementedError

        def _get_active_analyzer(self):
            return None

        def persist_loaded_data(self, x):
            return False

        def persist_analysis_outputs(self, results, summary):
            return None

    ctx = ToolContext(_NoopAgent())
    output = _ratio_metric_impl(ctx, "not-json")
    assert "error_code=" in output


def test_ratio_metric_tool_registered_in_agent_tools() -> None:
    import os

    os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
    from src.agent_tools import create_agent_tools

    class _StubAgent:
        analyzer = None
        data_question_service = None
        visualizer = None
        _last_results = None
        _last_summary = None
        _last_charts = {}
        FILE_SIZE_THRESHOLD_MB = 2.0

        def _load_data_with_backend(self, filepath):
            raise NotImplementedError

        def _normalize_shape(self, info):
            raise NotImplementedError

        def _get_active_analyzer(self):
            return None

        def persist_loaded_data(self, x):
            return False

        def persist_analysis_outputs(self, results, summary):
            return None

    tools = create_agent_tools(_StubAgent())
    assert any(t.name == "compute_ratio_metric" for t in tools)

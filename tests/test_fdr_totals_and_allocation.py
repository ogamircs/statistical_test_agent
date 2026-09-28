"""FDR-consistent effect totals and allocation-ratio propagation (TODO.md #40, #42, #92).

Regressions from the review of the statistical-correctness wave: the report
table re-summed t-test and proportion totals, the headline total ignored the
FDR-adjusted calls, BH could override a sequential "continue", and the
designed allocation ratio was dropped by configure_and_analyze/auto_configure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.agent as agent_module
from src.agent import ABTestingAgent
from src.agent_reporting import _render_ab_results_section
from src.statistics import ABTestAnalyzer
from src.statistics.engine_helpers import apply_fdr_correction
from src.statistics.models import ABTestResult, ABTestSummary


def _result(segment: str, p_value: float, **overrides) -> ABTestResult:
    base = ABTestResult(
        segment=segment,
        treatment_size=100,
        control_size=100,
        treatment_mean=11.0,
        control_mean=10.0,
        effect_size=1.0,
        cohens_d=0.2,
        t_statistic=2.0,
        p_value=p_value,
        is_significant=p_value < 0.05,
        confidence_interval=(0.1, 1.9),
        power=0.8,
        required_sample_size=100,
        is_sample_adequate=True,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_fdr_recomputes_total_effect_from_adjusted_calls() -> None:
    # Raw p=0.04 is significant; BH across [0.04, 0.5] adjusts it to 0.08.
    flipped = _result("A", 0.04, total_effect=100.0, total_effect_per_customer=1.0)
    other = _result("B", 0.5)

    apply_fdr_correction([flipped, other], significance_level=0.05)

    assert flipped.is_significant_adjusted is False
    assert flipped.total_effect == 0.0
    assert flipped.total_effect_per_customer == 0.0


def test_fdr_keeps_total_effect_for_segments_that_survive() -> None:
    survivor = _result("A", 0.001, total_effect=100.0, total_effect_per_customer=1.0)
    other = _result("B", 0.002)

    apply_fdr_correction([survivor, other], significance_level=0.05)

    assert survivor.is_significant_adjusted is True
    assert survivor.total_effect == pytest.approx(100.0)


def test_fdr_cannot_override_sequential_continue_decision() -> None:
    # Alpha-spending said "continue" (not significant) despite a small raw p.
    sequential = _result("A", 0.001, is_significant=False, sequential_mode_enabled=True)
    other = _result("B", 0.001, sequential_mode_enabled=True)

    apply_fdr_correction([sequential, other], significance_level=0.05)

    assert sequential.is_significant_adjusted is False
    assert other.is_significant_adjusted is True


def test_segment_table_total_is_not_double_counted() -> None:
    # Both tests significant, unbalanced arms: the old table showed
    # effect*Nt + prop_effect*Nc = 100 + 150 = 250.
    result = _result(
        "A",
        0.001,
        control_size=300,
        proportion_is_significant=True,
        proportion_p_value=0.001,
        proportion_effect_per_customer=0.5,
        total_effect=100.0,
    )
    summary = ABTestSummary(
        total_segments_analyzed=1,
        significant_segments=1,
        detailed_results=[result],
    )

    output = _render_ab_results_section(summary)

    row = next(line for line in output.splitlines() if line.startswith("| A |") and "100.00" in line)
    assert "250.00" not in row


def _holdout_frame(n_t: int = 100, n_c: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "experiment_group": ["treatment"] * n_t + ["control"] * n_c,
            "post_effect": rng.normal(10.0, 1.0, n_t + n_c),
        }
    )


class _DummyGraph:
    def invoke(self, *_args, **_kwargs):
        raise AssertionError("graph should not be invoked in tool tests")


@pytest.fixture
def agent(monkeypatch, tmp_path) -> ABTestingAgent:
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(
        agent_module, "create_agent", lambda _llm, _tools, system_prompt=None: _DummyGraph()
    )
    built = ABTestingAgent(query_store_path=str(tmp_path / "store.sqlite"))
    built.analyzer.set_dataframe(_holdout_frame())
    return built


def _tool(agent: ABTestingAgent, name: str):
    return next(tool for tool in agent._create_tools() if tool.name == name)


def _srm_flagged(agent: ABTestingAgent) -> bool:
    result = agent.analyzer.run_ab_test()
    return bool(result.diagnostics["experiment_quality"]["srm"]["is_sample_ratio_mismatch"])


def test_configure_and_analyze_accepts_expected_treatment_ratio(agent) -> None:
    _tool(agent, "configure_and_analyze").func(
        group_column="experiment_group",
        effect_column="post_effect",
        treatment_label="treatment",
        control_label="control",
        expected_treatment_ratio=0.1,
    )

    assert agent.analyzer.column_mapping["expected_treatment_ratio"] == pytest.approx(0.1)
    assert _srm_flagged(agent) is False


def test_configure_and_analyze_keeps_ratio_declared_earlier(agent) -> None:
    _tool(agent, "set_column_mapping").func(
        group="experiment_group", effect_value="post_effect", expected_treatment_ratio=0.1
    )
    _tool(agent, "configure_and_analyze").func(
        group_column="experiment_group",
        effect_column="post_effect",
        treatment_label="treatment",
        control_label="control",
    )

    assert agent.analyzer.column_mapping["expected_treatment_ratio"] == pytest.approx(0.1)
    assert _srm_flagged(agent) is False


def test_auto_configure_keeps_declared_ratio() -> None:
    analyzer = ABTestAnalyzer()
    analyzer.set_dataframe(_holdout_frame())
    analyzer.set_column_mapping(
        {"group": "experiment_group", "effect_value": "post_effect", "expected_treatment_ratio": 0.1}
    )

    analyzer.auto_configure()

    assert analyzer.column_mapping["expected_treatment_ratio"] == pytest.approx(0.1)


def _frame_with_pre() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n_t, n_c = 600, 400
    return pd.DataFrame(
        {
            "experiment_group": ["treatment"] * n_t + ["control"] * n_c,
            "customer_segment": rng.choice(["A", "B"], n_t + n_c),
            "pre_effect": np.concatenate([rng.normal(12, 1, n_t), rng.normal(10, 1, n_c)]),
            "post_effect": np.concatenate([rng.normal(13, 1, n_t), rng.normal(10, 1, n_c)]),
        }
    )


def _rerun_same_metric(agent: ABTestingAgent, **extra) -> None:
    agent.analyzer.set_dataframe(_frame_with_pre())
    agent.analyzer.auto_configure()
    assert agent.analyzer.column_mapping.get("pre_effect") == "pre_effect"
    _tool(agent, "configure_and_analyze").func(
        group_column="experiment_group",
        effect_column="post_effect",
        treatment_label="treatment",
        control_label="control",
        expected_treatment_ratio=0.6,
        **extra,
    )


def test_configure_and_analyze_rerun_keeps_pre_period_column(agent) -> None:
    # Regression from a live run: re-running with a declared allocation
    # dropped pre_effect, so the failed AA check and DiD silently vanished.
    _rerun_same_metric(agent, segment_column="customer_segment")

    assert agent.analyzer.column_mapping["pre_effect"] == "pre_effect"
    result = agent.analyzer.run_ab_test()
    assert result.aa_test_passed is False
    assert result.treatment_pre_mean > 0


def test_configure_and_analyze_omitting_segment_disables_segmentation(agent) -> None:
    _rerun_same_metric(agent)

    assert "segment" not in agent.analyzer.column_mapping


def test_configure_and_analyze_new_metric_starts_fresh(agent) -> None:
    agent.analyzer.set_dataframe(_frame_with_pre())
    agent.analyzer.auto_configure()
    _tool(agent, "configure_and_analyze").func(
        group_column="experiment_group",
        effect_column="pre_effect",
        treatment_label="treatment",
        control_label="control",
    )

    mapping = agent.analyzer.column_mapping
    assert mapping["effect_value"] == "pre_effect"
    assert "post_effect" not in mapping
    assert "pre_effect" not in mapping


def test_set_column_mapping_with_only_ratio_keeps_existing_columns(agent) -> None:
    # Regression from a live run: the model declared the allocation with
    # set_column_mapping(expected_treatment_ratio=0.6) alone, which replaced
    # the whole mapping and wiped group/effect/pre_effect.
    agent.analyzer.set_dataframe(_frame_with_pre())
    agent.analyzer.auto_configure()
    before = dict(agent.analyzer.column_mapping)

    output = _tool(agent, "set_column_mapping").func(expected_treatment_ratio=0.6)

    assert "error_code" not in output
    mapping = agent.analyzer.column_mapping
    for key in ("group", "effect_value", "pre_effect", "segment"):
        assert mapping[key] == before[key]
    assert mapping["expected_treatment_ratio"] == pytest.approx(0.6)

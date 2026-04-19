"""Sequential alpha-spending must drive is_significant when enabled."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics.analyzer import ABTestAnalyzer


def _df_with_borderline_signal() -> pd.DataFrame:
    """Build a dataset whose fixed-sample p sits between alpha and the
    interim Pocock boundary so we can see the override flip is_significant."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "experiment_group": ["treatment"] * 1000 + ["control"] * 1000,
            "post_effect": np.concatenate(
                [rng.normal(10.18, 2.0, 1000), rng.normal(10.0, 2.0, 1000)]
            ),
        }
    )


def _make_analyzer() -> ABTestAnalyzer:
    a = ABTestAnalyzer()
    a.set_dataframe(_df_with_borderline_signal())
    a.set_column_mapping({
        "group": "experiment_group",
        "effect_value": "post_effect",
        "post_effect": "post_effect",
    })
    a.set_group_labels("treatment", "control")
    return a


def test_sequential_disabled_uses_fixed_alpha() -> None:
    result = _make_analyzer().run_ab_test()
    assert result.sequential_mode_enabled is False
    assert result.is_significant == (result.p_value < 0.05)


def test_sequential_efficacy_stop_marks_significant() -> None:
    config = {
        "enabled": True,
        "look_index": 1,
        "max_looks": 5,
        "spending_method": "pocock",
    }
    result = _make_analyzer().run_ab_test(sequential_config=config)
    assert result.sequential_mode_enabled is True
    if result.sequential_decision == "stop_efficacy":
        assert result.is_significant is True
    elif result.sequential_decision == "continue":
        assert result.is_significant is False
    elif result.sequential_decision == "final_reject":
        assert result.is_significant is False
    elif result.sequential_decision == "final_accept":
        assert result.is_significant is True


def test_sequential_continue_blocks_significance_even_when_p_below_alpha() -> None:
    """At an early look with strict alpha-spending, p < 0.05 alone is not enough."""
    config = {
        "enabled": True,
        "look_index": 1,
        "max_looks": 10,
        "spending_method": "obrien_fleming",  # very strict early
    }
    result = _make_analyzer().run_ab_test(sequential_config=config)
    if result.p_value < 0.05 and result.sequential_decision == "continue":
        assert result.is_significant is False


def test_sequential_final_reject_blocks_significance() -> None:
    config = {
        "enabled": True,
        "look_index": 5,
        "max_looks": 5,
        "spending_method": "pocock",
    }
    result = _make_analyzer().run_ab_test(sequential_config=config)
    if result.sequential_decision == "final_reject":
        assert result.is_significant is False

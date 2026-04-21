"""Tests for configurable RNG seed in analyzer + bayesian + experiment_design."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics.analyzer import ABTestAnalyzer
from src.statistics.bayesian import run_bayesian_test
from src.statistics.statsmodels_engine import StatsmodelsABTestEngine


def _payload(samples: int = 200) -> dict:
    rng = np.random.default_rng(123)
    return {
        "treatment_post": rng.normal(10.5, 2.0, samples),
        "control_post": rng.normal(10.0, 2.0, samples),
        "n_samples": 5_000,
    }


def test_run_bayesian_test_seed_reproducible() -> None:
    a = run_bayesian_test(seed=7, **_payload())
    b = run_bayesian_test(seed=7, **_payload())
    assert a["prob_treatment_better"] == b["prob_treatment_better"]
    assert a["credible_interval"] == b["credible_interval"]


def test_run_bayesian_test_different_seed_yields_different_samples() -> None:
    a = run_bayesian_test(seed=1, **_payload())
    b = run_bayesian_test(seed=99, **_payload())
    assert a["credible_interval"] != b["credible_interval"]


def test_engine_threads_seed_into_bayesian() -> None:
    e1 = StatsmodelsABTestEngine(seed=1)
    e2 = StatsmodelsABTestEngine(seed=2)
    payload = _payload()
    r1 = e1.run_bayesian_test(payload["treatment_post"], payload["control_post"])
    r2 = e2.run_bayesian_test(payload["treatment_post"], payload["control_post"])
    assert r1["credible_interval"] != r2["credible_interval"]


def _df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "experiment_group": ["treatment"] * 200 + ["control"] * 200,
            "post_effect": np.concatenate(
                [rng.normal(10.5, 2.0, 200), rng.normal(10.0, 2.0, 200)]
            ),
        }
    )


def test_analyzer_seed_propagates_to_bayesian_results() -> None:
    df = _df()

    def _run(seed: int):
        a = ABTestAnalyzer(seed=seed)
        a.set_dataframe(df.copy())
        a.set_column_mapping({
            "group": "experiment_group",
            "effect_value": "post_effect",
            "post_effect": "post_effect",
        })
        a.set_group_labels("treatment", "control")
        return a.run_ab_test()

    same_a = _run(11)
    same_b = _run(11)
    different = _run(99)

    assert same_a.bayesian_credible_interval == same_b.bayesian_credible_interval
    assert same_a.bayesian_credible_interval != different.bayesian_credible_interval

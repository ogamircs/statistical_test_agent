"""Direct unit tests for metric-family detection."""

from __future__ import annotations

import numpy as np

from src.statistics.model_families import (
    infer_metric_type,
    is_binary_metric,
    is_count_metric,
    is_heavy_tail_metric,
)


VARIANCE_EPSILON = 1e-12


def test_is_binary_metric_zero_one() -> None:
    assert is_binary_metric(np.array([0, 1, 0, 1])) is True


def test_is_binary_metric_rejects_continuous() -> None:
    assert is_binary_metric(np.array([0.1, 0.5, 0.9])) is False


def test_is_count_metric_nonneg_integers() -> None:
    assert is_count_metric(np.array([0, 1, 5, 10])) is True
    assert is_count_metric(np.array([0, 1, 5, 10, 2, 3])) is True


def test_is_count_metric_rejects_negatives() -> None:
    assert is_count_metric(np.array([-1, 2, 3])) is False


def test_is_heavy_tail_metric_detects_skewed_distribution() -> None:
    rng = np.random.default_rng(0)
    heavy = rng.lognormal(mean=0.0, sigma=2.0, size=500)
    assert is_heavy_tail_metric(heavy, variance_epsilon=VARIANCE_EPSILON) is True


def test_is_heavy_tail_metric_rejects_normal() -> None:
    rng = np.random.default_rng(1)
    # Mean far from zero so the p99/p50 ratio rule cannot accidentally fire.
    normal = rng.normal(50.0, 1.0, size=500)
    assert is_heavy_tail_metric(normal, variance_epsilon=VARIANCE_EPSILON) is False


def test_is_heavy_tail_metric_too_small_returns_false() -> None:
    assert is_heavy_tail_metric(np.array([1.0, 2.0, 3.0]), variance_epsilon=VARIANCE_EPSILON) is False


def test_infer_metric_type_routes_to_binary() -> None:
    assert (
        infer_metric_type(np.array([0, 1, 0, 1]), variance_epsilon=VARIANCE_EPSILON)
        == "binary"
    )


def test_infer_metric_type_routes_to_count() -> None:
    assert (
        infer_metric_type(
            np.array([0, 1, 5, 10, 2, 3]), variance_epsilon=VARIANCE_EPSILON
        )
        == "count"
    )


def test_infer_metric_type_routes_to_heavy_tail() -> None:
    rng = np.random.default_rng(2)
    heavy = rng.lognormal(mean=0.0, sigma=2.5, size=500)
    assert infer_metric_type(heavy, variance_epsilon=VARIANCE_EPSILON) == "heavy_tail"


def test_infer_metric_type_falls_through_to_continuous() -> None:
    rng = np.random.default_rng(3)
    normal = rng.normal(50.0, 10.0, size=500)
    assert infer_metric_type(normal, variance_epsilon=VARIANCE_EPSILON) == "continuous"


def test_infer_metric_type_respects_explicit_request() -> None:
    assert (
        infer_metric_type(
            np.array([0, 1, 0, 1]),
            requested_metric_type="continuous",
            variance_epsilon=VARIANCE_EPSILON,
        )
        == "continuous"
    )

"""Direct unit tests for label_inference."""

from __future__ import annotations

from src.statistics.label_inference import infer_group_labels


def test_infers_treatment_and_control_from_canonical_tokens() -> None:
    out = infer_group_labels(["treatment", "control"])
    assert out["treatment"] == "treatment"
    assert out["control"] == "control"
    assert out["confidence"] == "high"


def test_infers_test_vs_ctrl_synonyms() -> None:
    out = infer_group_labels(["test", "ctrl"])
    assert out["treatment"] == "test"
    assert out["control"] == "ctrl"


def test_handles_binary_zero_one_groups() -> None:
    out = infer_group_labels([0, 1])
    assert out["treatment"] == 1
    assert out["control"] == 0


def test_falls_back_to_sorted_order_when_ambiguous() -> None:
    out = infer_group_labels(["alpha", "beta"])
    assert out["confidence"] == "low"
    assert out["treatment"] in {"alpha", "beta"}
    assert out["control"] in {"alpha", "beta"}
    assert out["treatment"] != out["control"]
    assert out["warnings"]


def test_strips_none_values() -> None:
    out = infer_group_labels([None, "treatment", "control"])
    assert out["treatment"] == "treatment"
    assert out["control"] == "control"

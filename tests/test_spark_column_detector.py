"""SparkColumnDetector — pattern-only role detection, no SparkSession needed."""

from __future__ import annotations

from src.statistics.pyspark_analyzer import SparkColumnDetector


def _detect(columns, numeric_columns):
    return SparkColumnDetector(columns=columns, numeric_columns=numeric_columns).detect()


def test_detect_full_canonical_schema() -> None:
    columns = [
        "customer_id",
        "experiment_group",
        "customer_segment",
        "pre_effect",
        "post_effect",
        "experiment_duration_days",
    ]
    numeric = ["pre_effect", "post_effect", "experiment_duration_days"]
    out = _detect(columns, numeric)

    # "customer_segment" matches both customer-id and segment patterns by
    # design — preserved from the legacy detect_columns behavior.
    assert "customer_id" in out["customer_id"]
    assert "customer_segment" in out["customer_id"]
    assert out["group"] == ["experiment_group"]
    assert "customer_segment" in out["segment"]
    assert out["pre_effect"] == ["pre_effect"]
    assert out["post_effect"] == ["post_effect"]
    assert "experiment_duration_days" in out["duration"]


def test_post_effect_requires_numeric_dtype() -> None:
    out = _detect(["post_effect", "post_effect_label"], numeric_columns=["post_effect"])
    assert out["post_effect"] == ["post_effect"]
    # post_effect_label is non-numeric → not surfaced as post_effect
    assert "post_effect_label" not in out["post_effect"]


def test_effect_value_falls_back_to_unclassified_numerics() -> None:
    out = _detect(
        ["customer_id", "experiment_group", "revenue_amount"],
        numeric_columns=["revenue_amount"],
    )
    # revenue_amount matches the effect pattern -> classified directly
    assert "revenue_amount" in out["effect_value"]


def test_effect_value_falls_back_when_no_pattern_matches() -> None:
    out = _detect(
        ["customer_id", "group", "x_metric_1", "x_metric_2"],
        numeric_columns=["x_metric_1", "x_metric_2"],
    )
    assert sorted(out["effect_value"]) == ["x_metric_1", "x_metric_2"]


def test_segment_does_not_collide_with_group() -> None:
    out = _detect(["experiment_group"], numeric_columns=[])
    assert "experiment_group" not in out["segment"]


def test_alt_naming_revenue_alias() -> None:
    out = _detect(
        ["ab_variant", "tier", "revenue"],
        numeric_columns=["revenue"],
    )
    assert "ab_variant" in out["group"]
    assert "tier" in out["segment"]
    assert "revenue" in out["effect_value"]

"""Shared chart registry used by the visualizer and agent tool layer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .models import to_ab_test_summary


@dataclass(frozen=True)
class ChartDefinition:
    key: str
    aliases: Tuple[str, ...] = ()


CHART_DEFINITIONS: Tuple[ChartDefinition, ...] = (
    ChartDefinition("statistical_summary", ("summary", "stats", "table")),
    ChartDefinition("dashboard", ()),
    ChartDefinition("treatment_vs_control", ("treatment_control", "comparison", "means")),
    ChartDefinition("effect_sizes", ("effect", "effect_size", "effects")),
    ChartDefinition("combined_effects", ("combined",)),
    ChartDefinition("proportion_comparison", ("proportion", "conversion")),
    ChartDefinition("p_values", ("pvalue", "p_value", "significance")),
    ChartDefinition("sample_sizes", ("sample", "sample_size", "samples")),
    ChartDefinition("power_analysis", ("power",)),
    ChartDefinition("cohens_d", ("cohen", "effect_magnitude")),
    ChartDefinition("effect_waterfall", ("waterfall", "contribution")),
    ChartDefinition("bayesian_probability", ("probability",)),
    ChartDefinition("bayesian_credible_intervals", ("bayesian_credible", "credible_interval", "credible")),
    ChartDefinition("bayesian_expected_loss", ("bayesian_loss", "expected_loss", "loss")),
    ChartDefinition("distribution", ()),
)

_BAYESIAN_GROUP = [
    "bayesian_probability",
    "bayesian_credible_intervals",
    "bayesian_expected_loss",
]


def all_chart_keys() -> List[str]:
    return [definition.key for definition in CHART_DEFINITIONS]


def core_chart_keys() -> List[str]:
    return [key for key in all_chart_keys() if key != "distribution"]


def resolve_chart_keys(selection: str) -> List[str]:
    normalized = (selection or "all").strip().lower()
    if normalized == "all":
        return core_chart_keys()
    if normalized == "bayesian":
        return list(_BAYESIAN_GROUP)

    for definition in CHART_DEFINITIONS:
        if normalized == definition.key or normalized in definition.aliases:
            return [definition.key]
    return []


def contains_chart(chart_keys: Iterable[str], key: str) -> bool:
    return key in set(chart_keys)


def build_chart_map(
    visualizer: Any,
    results: Any,
    summary: Any,
    selected_keys: Optional[Iterable[str]] = None,
    df: Any = None,
    group_col: Optional[str] = None,
    segment_col: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_summary = to_ab_test_summary(summary)
    chart_factories: Dict[str, Any] = {
        "statistical_summary": lambda: visualizer.plot_statistical_summary(results),
        "dashboard": lambda: visualizer.plot_summary_dashboard(results, normalized_summary),
        "treatment_vs_control": lambda: visualizer.plot_treatment_vs_control(results),
        "effect_sizes": lambda: visualizer.plot_effect_sizes(results),
        "combined_effects": lambda: visualizer.plot_combined_effects(results),
        "proportion_comparison": lambda: visualizer.plot_proportion_comparison(results),
        "p_values": lambda: visualizer.plot_p_values(results),
        "sample_sizes": lambda: visualizer.plot_sample_sizes(results),
        "power_analysis": lambda: visualizer.plot_power_analysis(results),
        "cohens_d": lambda: visualizer.plot_cohens_d(results),
        "effect_waterfall": lambda: visualizer.plot_effect_waterfall(results),
        "bayesian_probability": lambda: visualizer.plot_bayesian_probability(results),
        "bayesian_credible_intervals": lambda: visualizer.plot_bayesian_credible_intervals(results),
        "bayesian_expected_loss": lambda: visualizer.plot_bayesian_expected_loss(results),
    }

    if df is not None and group_col is not None:
        chart_factories["distribution"] = lambda: visualizer.plot_segment_distribution(
            df, group_col, segment_col
        )

    keys = list(selected_keys) if selected_keys is not None else core_chart_keys()
    # Only build charts that have a registered factory
    return {key: chart_factories[key]() for key in keys if key in chart_factories}

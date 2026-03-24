"""
Sequential Testing Configuration and Decision Evaluation

Resolves optional sequential testing configuration from user input
and evaluates stopping decisions via the statistics engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .statsmodels_engine import StatsmodelsABTestEngine


@dataclass(frozen=True)
class SequentialConfig:
    """Resolved sequential testing configuration."""

    enabled: bool
    look_index: int = 0
    max_looks: int = 0
    spending_method: str = "none"
    futility_min_information_fraction: float = 0.75
    futility_p_value_threshold: float = 0.5


_DISABLED_SEQUENTIAL_RESULTS: Dict[str, Any] = {
    "enabled": False,
    "method": "none",
    "look_index": 0,
    "max_looks": 0,
    "information_fraction": 0.0,
    "alpha_spent": 0.0,
    "stop_recommended": False,
    "decision": "not_requested",
    "rationale": "",
    "thresholds": {},
}


def _coerce_int(value: Any, *, default: int, minimum: int = 1) -> int:
    """Best-effort integer coercion with lower-bound clamping."""
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = default
    return max(numeric, minimum)


def _coerce_float(
    value: Any,
    *,
    default: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """Best-effort float coercion with bounded range."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return float(min(max(numeric, minimum), maximum))


def resolve_sequential_config(
    sequential_config: Optional[Mapping[str, Any]],
    column_mapping: Dict[str, Any],
    stats_engine: StatsmodelsABTestEngine,
) -> SequentialConfig:
    """
    Resolve optional sequential testing configuration.

    Sequential mode is opt-in through either:
    - run_ab_test(..., sequential_config={...})
    - column_mapping["sequential"] = {...}
    """
    raw_config: Any = (
        sequential_config
        if sequential_config is not None
        else column_mapping.get("sequential")
    )

    if raw_config in (None, False):
        return SequentialConfig(enabled=False)

    if raw_config is True:
        raw_dict: Dict[str, Any] = {}
    elif isinstance(raw_config, Mapping):
        raw_dict = dict(raw_config)
    else:
        return SequentialConfig(enabled=False)

    enabled = bool(raw_dict.get("enabled", True))
    if not enabled:
        return SequentialConfig(enabled=False)

    look_index = _coerce_int(
        raw_dict.get(
            "look_index",
            raw_dict.get("current_look", raw_dict.get("interim_look", 1)),
        ),
        default=1,
        minimum=1,
    )
    max_looks = _coerce_int(
        raw_dict.get(
            "max_looks",
            raw_dict.get("total_looks", raw_dict.get("planned_looks", look_index)),
        ),
        default=look_index,
        minimum=1,
    )
    look_index = min(look_index, max_looks)

    method = str(
        raw_dict.get(
            "spending_method",
            raw_dict.get("method", stats_engine.DEFAULT_SEQUENTIAL_METHOD),
        )
    ).strip().lower()

    return SequentialConfig(
        enabled=True,
        look_index=look_index,
        max_looks=max_looks,
        spending_method=method,
        futility_min_information_fraction=_coerce_float(
            raw_dict.get(
                "futility_min_information_fraction",
                stats_engine.DEFAULT_FUTILITY_MIN_INFORMATION_FRACTION,
            ),
            default=stats_engine.DEFAULT_FUTILITY_MIN_INFORMATION_FRACTION,
            minimum=0.0,
            maximum=1.0,
        ),
        futility_p_value_threshold=_coerce_float(
            raw_dict.get(
                "futility_p_value_threshold",
                stats_engine.DEFAULT_FUTILITY_P_VALUE_THRESHOLD,
            ),
            default=stats_engine.DEFAULT_FUTILITY_P_VALUE_THRESHOLD,
            minimum=0.0,
            maximum=1.0,
        ),
    )


def evaluate_sequential_decision(
    *,
    sequential_config: Optional[Mapping[str, Any]],
    column_mapping: Dict[str, Any],
    stats_engine: StatsmodelsABTestEngine,
    p_value: float,
    effect_size: float,
    confidence_interval: tuple[float, float],
) -> Dict[str, Any]:
    """Evaluate opt-in sequential stopping logic for the current result."""
    config = resolve_sequential_config(sequential_config, column_mapping, stats_engine)
    if not config.enabled:
        return dict(_DISABLED_SEQUENTIAL_RESULTS)

    return stats_engine.evaluate_sequential_decision(
        p_value=p_value,
        effect_size=effect_size,
        confidence_interval=confidence_interval,
        look_index=config.look_index,
        max_looks=config.max_looks,
        method=config.spending_method,
        futility_min_information_fraction=config.futility_min_information_fraction,
        futility_p_value_threshold=config.futility_p_value_threshold,
    )

"""Shared heuristics for inferring treatment/control labels."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


_TREATMENT_TOKENS = {
    "treatment",
    "treat",
    "test",
    "experiment",
    "exposed",
    "true",
    "yes",
    "1",
}

_CONTROL_TOKENS = {
    "control",
    "ctrl",
    "baseline",
    "placebo",
    "unexposed",
    "false",
    "no",
    "0",
}


def _tokenize_label(value: Any) -> List[str]:
    normalized = str(value).strip().lower()
    if not normalized:
        return []
    return re.findall(r"[a-z0-9]+", normalized)


def infer_group_labels(unique_values: Sequence[Any]) -> Dict[str, Any]:
    """
    Infer treatment/control labels from a list of unique group values.

    Strong matches use exact normalized tokens such as "treatment", "control",
    "test", or "ctrl". Ambiguous labels fall back to deterministic sorted order
    and emit a low-confidence warning.
    """
    values = [value for value in unique_values if value is not None]
    warnings: List[str] = []

    treatment_label = None
    control_label = None

    for value in values:
        tokens = set(_tokenize_label(value))
        if not tokens:
            continue

        is_treatment = bool(tokens & _TREATMENT_TOKENS)
        is_control = bool(tokens & _CONTROL_TOKENS)

        if is_treatment and not is_control and treatment_label is None:
            treatment_label = value
        elif is_control and not is_treatment and control_label is None:
            control_label = value

    if len(values) == 2:
        if treatment_label is not None and control_label is None:
            control_label = next(value for value in values if value != treatment_label)
            warnings.append(
                "Low-confidence control guess based on the remaining group label. "
                "Please confirm the treatment/control mapping."
            )
        elif control_label is not None and treatment_label is None:
            treatment_label = next(value for value in values if value != control_label)
            warnings.append(
                "Low-confidence treatment guess based on the remaining group label. "
                "Please confirm the treatment/control mapping."
            )

    if treatment_label is None or control_label is None:
        if len(values) >= 2:
            sorted_values = sorted(values, key=lambda value: str(value).lower())
            control_label = sorted_values[0]
            treatment_label = sorted_values[1]
            warnings.append(
                "Low-confidence treatment/control guess based on sorted group label order. "
                "Please confirm the mapping."
            )

    return {
        "treatment": treatment_label,
        "control": control_label,
        "warnings": warnings,
        "confidence": "high" if not warnings else "low",
    }

"""Centralized helpers for truncating LLM-bound payloads.

Lives at the top level (not under ``tooling/``) so both ``agent_reporting``
and ``tooling.common`` can import from it without creating an import cycle.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

import pandas as pd

DEFAULT_LLM_ROW_LIMIT = 20


def truncate_dataframe_for_llm(
    frame: pd.DataFrame,
    max_rows: int = DEFAULT_LLM_ROW_LIMIT,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Return ``(head, suffix)`` where suffix is the "showing N of M" notice.

    When the frame is at or below ``max_rows`` the suffix is None so callers
    can omit the trailing line cleanly.
    """
    total = int(len(frame))
    if total <= max_rows:
        return frame, None
    return frame.head(max_rows), f"_showing first {max_rows:,} of {total:,} rows_"


def truncate_iterable_for_llm(
    items: Iterable[Any],
    max_items: int = DEFAULT_LLM_ROW_LIMIT,
) -> Tuple[List[Any], int]:
    """Return ``(head_list, omitted_count)``."""
    materialized = list(items)
    if len(materialized) <= max_items:
        return materialized, 0
    return materialized[:max_items], len(materialized) - max_items

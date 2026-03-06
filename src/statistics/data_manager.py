"""
Data access and schema management for A/B test analysis.

This module isolates dataframe lifecycle, column detection, auto-configuration,
and basic data queries from statistical computation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .label_inference import infer_group_labels

logger = logging.getLogger(__name__)


class DataQueryError(ValueError):
    """User-facing query validation error with stable metadata."""

    def __init__(self, code: str, message: str, *, query: Optional[str] = None) -> None:
        self.code = code
        self.user_message = message
        self.query = query
        super().__init__(message)


class ABTestDataManager:
    """Manage experiment dataframe state, schema detection, and mappings."""

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.column_mapping: Dict[str, str] = {}
        self.treatment_label: Optional[Any] = None
        self.control_label: Optional[Any] = None

    def load_data(self, filepath: str) -> Dict[str, Any]:
        """Load CSV data and return quick metadata."""
        self.df = pd.read_csv(filepath)
        return {
            "columns": list(self.df.columns),
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.to_dict(),
            "sample": self.df.head(3).to_dict(),
        }

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Set dataframe directly."""
        self.df = df

    def set_column_mapping(self, mapping: Dict[str, str]) -> None:
        """Set column mapping for analysis."""
        self.column_mapping = mapping

    def set_group_labels(self, treatment_label: Any, control_label: Any) -> None:
        """Set treatment and control labels used in group column."""
        self.treatment_label = treatment_label
        self.control_label = control_label

    def get_group_values(self) -> Dict[str, List[Any]]:
        """Return unique values for configured group column."""
        if self.df is None or "group" not in self.column_mapping:
            raise ValueError("Data not loaded or group column not set")

        group_col = self.column_mapping["group"]
        unique_values = self.df[group_col].dropna().unique().tolist()
        return {"group_column": group_col, "unique_values": unique_values}

    def detect_columns(self) -> Dict[str, List[str]]:
        """
        Auto-detect likely columns based on naming patterns.

        Returns candidate column lists for each required/optional role.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        columns = [col.lower() for col in self.df.columns]
        original_columns = list(self.df.columns)

        suggestions = {
            "customer_id": [],
            "group": [],
            "pre_effect": [],
            "post_effect": [],
            "effect_value": [],
            "segment": [],
            "duration": [],
        }

        numeric_mask = {
            col: pd.api.types.is_numeric_dtype(self.df[col]) for col in original_columns
        }

        id_patterns = ["customer_id", "customerid", "user_id", "userid", "id", "customer"]
        group_patterns = [
            "group",
            "treatment",
            "control",
            "variant",
            "test_group",
            "experiment_group",
            "ab_group",
        ]
        pre_effect_patterns = ["pre_effect", "pre_value", "pre_revenue", "baseline", "before"]
        post_effect_patterns = ["post_effect", "post_value", "effect_value", "revenue", "amount", "score"]
        effect_patterns = [
            "effect",
            "value",
            "metric",
            "outcome",
            "result",
            "conversion",
            "revenue",
            "amount",
            "score",
        ]
        segment_patterns = ["segment", "category", "tier", "type", "cohort", "cluster", "group_name"]
        duration_patterns = ["duration", "days", "period", "time", "length", "exposure"]

        for i, col_lower in enumerate(columns):
            col = original_columns[i]

            if any(pattern in col_lower for pattern in id_patterns):
                suggestions["customer_id"].append(col)

            if any(pattern in col_lower for pattern in group_patterns):
                suggestions["group"].append(col)

            if any(pattern in col_lower for pattern in pre_effect_patterns) and numeric_mask[col]:
                suggestions["pre_effect"].append(col)

            if ("post_" in col_lower or col_lower == "post_effect") and numeric_mask[col]:
                suggestions["post_effect"].append(col)

            if any(pattern in col_lower for pattern in effect_patterns) and numeric_mask[col]:
                if col not in suggestions["pre_effect"]:
                    suggestions["effect_value"].append(col)

            if any(pattern in col_lower for pattern in segment_patterns):
                if col_lower not in [c.lower() for c in suggestions["group"]]:
                    suggestions["segment"].append(col)

            if any(pattern in col_lower for pattern in duration_patterns):
                suggestions["duration"].append(col)

        # Secondary post-effect pass (kept for backward-compatible behavior)
        for i, col_lower in enumerate(columns):
            col = original_columns[i]
            if any(pattern in col_lower for pattern in post_effect_patterns) and numeric_mask[col]:
                # Do not let pre-period metrics (e.g., pre_revenue) leak into
                # post-effect suggestions.
                if col in suggestions["pre_effect"] or any(
                    pattern in col_lower for pattern in pre_effect_patterns
                ):
                    continue
                if col not in suggestions["post_effect"]:
                    suggestions["post_effect"].append(col)

        return suggestions

    def auto_configure(self) -> Dict[str, Any]:
        """
        Auto-configure mapping and group labels using best effort heuristics.
        """
        if self.df is None:
            return {"success": False, "error": "No data loaded"}

        config: Dict[str, Any] = {"success": True, "warnings": [], "mapping": {}, "labels": {}}
        suggestions = self.detect_columns()

        if suggestions["group"]:
            config["mapping"]["group"] = suggestions["group"][0]
        else:
            for col in self.df.columns:
                if self.df[col].nunique(dropna=True) == 2:
                    config["mapping"]["group"] = col
                    config["warnings"].append(
                        f"Guessed '{col}' as group column (has 2 unique values)"
                    )
                    break

        if "group" not in config["mapping"]:
            return {"success": False, "error": "Could not detect group column"}

        if suggestions["pre_effect"]:
            config["mapping"]["pre_effect"] = suggestions["pre_effect"][0]

        if suggestions["post_effect"]:
            config["mapping"]["post_effect"] = suggestions["post_effect"][0]
            config["mapping"]["effect_value"] = suggestions["post_effect"][0]
        elif suggestions["effect_value"]:
            config["mapping"]["effect_value"] = suggestions["effect_value"][0]
            config["mapping"]["post_effect"] = suggestions["effect_value"][0]
        else:
            numeric_cols = list(self.df.select_dtypes(include=["number"]).columns)
            for col in numeric_cols:
                if col != config["mapping"].get("group") and col not in suggestions.get("pre_effect", []):
                    config["mapping"]["effect_value"] = col
                    config["mapping"]["post_effect"] = col
                    config["warnings"].append(
                        f"Guessed '{col}' as effect value column (numeric)"
                    )
                    break

        if "effect_value" not in config["mapping"]:
            return {"success": False, "error": "Could not detect effect value column"}

        if suggestions["segment"]:
            config["mapping"]["segment"] = suggestions["segment"][0]
        if suggestions["customer_id"]:
            config["mapping"]["customer_id"] = suggestions["customer_id"][0]

        self.set_column_mapping(config["mapping"])

        group_col = config["mapping"]["group"]
        unique_values = self.df[group_col].dropna().unique().tolist()

        label_guess = infer_group_labels(unique_values)
        treatment_label = label_guess["treatment"]
        control_label = label_guess["control"]
        config["warnings"].extend(label_guess["warnings"])

        if treatment_label is None or control_label is None:
            return {"success": False, "error": "Could not detect treatment/control labels"}

        config["labels"]["treatment"] = treatment_label
        config["labels"]["control"] = control_label
        self.set_group_labels(treatment_label, control_label)

        return config

    def query_data(self, query: str) -> pd.DataFrame:
        """Execute pandas query expression against active dataframe."""
        if self.df is None:
            raise DataQueryError("DATA_NOT_LOADED", "No data loaded")

        normalized_query = (query or "").strip()
        if not normalized_query:
            raise DataQueryError(
                "QUERY_EMPTY",
                "Query cannot be empty. Please provide a filter expression.",
            )

        broad_patterns = {
            "*",
            "all",
            "true",
            "1 == 1",
            "1==1",
            "index == index",
            "index==index",
        }
        if normalized_query.lower() in broad_patterns:
            raise DataQueryError(
                "QUERY_TOO_BROAD",
                "Query is too broad. Please provide a specific filter condition.",
                query=normalized_query,
            )

        try:
            return self.df.query(normalized_query)
        except Exception as error:
            logger.warning(
                "Invalid dataframe query rejected: %s",
                normalized_query,
                exc_info=error,
            )
            raise DataQueryError(
                "INVALID_QUERY",
                "Invalid query syntax. Use pandas query syntax, e.g. segment == 'Premium'.",
                query=normalized_query,
            ) from error

    def get_data_summary(self) -> Dict[str, Any]:
        """Return high-level data profiling summary."""
        if self.df is None:
            raise ValueError("No data loaded")

        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_summary": self.df.describe().to_dict(),
            "sample_rows": self.df.head(5).to_dict(),
        }

    def get_segment_distribution(self) -> Dict[str, Any]:
        """Return group and segment distribution tables."""
        if self.df is None:
            raise ValueError("No data loaded")

        result: Dict[str, Any] = {"columns_used": self.column_mapping}

        if "group" in self.column_mapping:
            group_col = self.column_mapping["group"]
            result["group_distribution"] = self.df[group_col].value_counts().to_dict()

        if "segment" in self.column_mapping:
            segment_col = self.column_mapping["segment"]
            result["segment_distribution"] = self.df[segment_col].value_counts().to_dict()

            if "group" in self.column_mapping:
                group_col = self.column_mapping["group"]
                cross_tab = pd.crosstab(self.df[segment_col], self.df[group_col]).to_dict()
                result["segment_by_group"] = cross_tab

        return result

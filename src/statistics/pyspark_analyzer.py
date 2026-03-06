"""
PySpark-based A/B Test Statistical Analyzer

High-performance distributed A/B testing analysis using PySpark and MLlib.
Designed for large-scale experiments with millions of customers.

Key Features:
- Distributed statistical computations using Spark DataFrame operations
- MLlib integration for statistical tests and power analysis
- Efficient aggregations avoiding UDFs where possible
- Support for AA tests, frequentist tests, Bayesian analysis, and DiD
- Handles massive datasets through partitioning and caching strategies

Performance Notes:
- Use broadcast variables for small lookup tables
- Leverage Spark's built-in statistical functions
- Minimize shuffles through appropriate partitioning
- Cache intermediate results for iterative operations
"""

import logging

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, BooleanType, ArrayType
from pyspark.ml.stat import Correlation, ChiSquareTest
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.stat import Statistics
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json

from ..agent_reporting import AgentUserFacingError
from .label_inference import infer_group_labels
from .models import ABTestResult
from .summary_builder import ABTestSummaryBuilder


logger = logging.getLogger(__name__)


@dataclass
class SparkABTestResult(ABTestResult):
    """
    Spark result type compatible with canonical ABTestResult schema.

    Backward-compatibility aliases (`*_lower` / `*_upper`) are exposed as
    properties while canonical storage uses tuple intervals.
    """

    pooled_std: float = 0.0

    @property
    def confidence_interval_lower(self) -> float:
        return float(self.confidence_interval[0])

    @confidence_interval_lower.setter
    def confidence_interval_lower(self, value: float) -> None:
        self.confidence_interval = (float(value), float(self.confidence_interval[1]))

    @property
    def confidence_interval_upper(self) -> float:
        return float(self.confidence_interval[1])

    @confidence_interval_upper.setter
    def confidence_interval_upper(self, value: float) -> None:
        self.confidence_interval = (float(self.confidence_interval[0]), float(value))

    @property
    def bayesian_credible_interval_lower(self) -> float:
        return float(self.bayesian_credible_interval[0])

    @bayesian_credible_interval_lower.setter
    def bayesian_credible_interval_lower(self, value: float) -> None:
        self.bayesian_credible_interval = (
            float(value),
            float(self.bayesian_credible_interval[1]),
        )

    @property
    def bayesian_credible_interval_upper(self) -> float:
        return float(self.bayesian_credible_interval[1])

    @bayesian_credible_interval_upper.setter
    def bayesian_credible_interval_upper(self, value: float) -> None:
        self.bayesian_credible_interval = (
            float(self.bayesian_credible_interval[0]),
            float(value),
        )

    def to_serializable_dict(self, include_legacy_aliases: bool = True) -> Dict[str, Any]:
        """Serialize result for persistence layers."""
        payload = asdict(self)
        payload.pop("confidence_interval", None)
        payload.pop("bayesian_credible_interval", None)

        # Preserve flattened interval columns for downstream compatibility.
        payload["confidence_interval_lower"] = self.confidence_interval_lower
        payload["confidence_interval_upper"] = self.confidence_interval_upper
        payload["bayesian_credible_interval_lower"] = self.bayesian_credible_interval_lower
        payload["bayesian_credible_interval_upper"] = self.bayesian_credible_interval_upper

        if include_legacy_aliases:
            payload["ci_lower"] = self.confidence_interval_lower
            payload["ci_upper"] = self.confidence_interval_upper
            payload["bayesian_credible_lower"] = self.bayesian_credible_interval_lower
            payload["bayesian_credible_upper"] = self.bayesian_credible_interval_upper
        return payload


class PySparkABTestAnalyzer:
    """
    Distributed A/B Test Analyzer using PySpark

    Optimized for large-scale data processing with:
    - Minimal data movement through broadcast joins
    - Cached intermediate results for multi-pass analysis
    - Native Spark statistical functions
    - Efficient aggregations and window functions
    """

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        significance_level: float = 0.05,
        power_threshold: float = 0.8
    ):
        """
        Initialize the PySpark analyzer

        Args:
            spark: Active SparkSession (optional, auto-created when omitted)
            significance_level: Alpha for hypothesis tests (default: 0.05)
            power_threshold: Minimum statistical power (default: 0.8)
        """
        self.spark = spark or SparkSession.builder.appName("ABTestingAnalyzer").getOrCreate()
        self.significance_level = significance_level
        self.power_threshold = power_threshold
        self.df: Optional[DataFrame] = None
        self.column_mapping: Dict[str, str] = {}
        self.treatment_label: Optional[str] = None
        self.control_label: Optional[str] = None
        self.summary_builder = ABTestSummaryBuilder()
        self.segment_failures: List[Dict[str, str]] = []

    def load_data(self, path: str, format: str = "csv", **options) -> Dict[str, Any]:
        """
        Load data into Spark DataFrame

        Args:
            path: Path to data (supports S3, HDFS, local, etc.)
            format: Data format (csv, parquet, delta, etc.)
            **options: Additional Spark read options (header, inferSchema, etc.)

        Returns:
            Dictionary with data information
        """
        default_options = {"header": "true", "inferSchema": "true"}
        default_options.update(options)

        self.df = self.spark.read.format(format).options(**default_options).load(path)

        # Cache for repeated access
        self.df.cache()

        # Collect summary stats
        row_count = self.df.count()
        columns = self.df.columns
        shape = (row_count, len(columns))
        dtypes = dict(self.df.dtypes)

        return {
            "columns": columns,
            "shape": shape,
            "dtypes": dtypes,
            "row_count": row_count,
            "partitions": self.df.rdd.getNumPartitions(),
            "schema": self.df.schema.json()
        }

    def set_dataframe(self, df: DataFrame) -> None:
        """Set DataFrame directly and cache it"""
        self.df = df
        self.df.cache()

    def detect_columns(self) -> Dict[str, List[str]]:
        """
        Auto-detect column types based on naming patterns
        Uses Spark native operations for schema inspection
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        columns = self.df.columns
        columns_lower = [col.lower() for col in columns]
        schema = self.df.schema

        suggestions = {
            "customer_id": [],
            "group": [],
            "pre_effect": [],
            "post_effect": [],
            "effect_value": [],
            "segment": [],
            "duration": []
        }

        # Get numeric columns
        numeric_types = ["int", "bigint", "long", "double", "float", "decimal"]
        numeric_columns = [
            field.name for field in schema.fields
            if any(t in str(field.dataType).lower() for t in numeric_types)
        ]

        # Customer ID patterns
        id_patterns = ['customer_id', 'customerid', 'user_id', 'userid', 'id', 'customer']
        for i, col in enumerate(columns_lower):
            if any(pattern in col for pattern in id_patterns):
                suggestions["customer_id"].append(columns[i])

        # Group patterns
        group_patterns = ['group', 'treatment', 'control', 'variant', 'test_group', 'experiment_group', 'ab_group']
        for i, col in enumerate(columns_lower):
            if any(pattern in col for pattern in group_patterns):
                suggestions["group"].append(columns[i])

        # Pre-effect patterns
        pre_patterns = ['pre_effect', 'pre_value', 'pre_revenue', 'baseline', 'before']
        for i, col in enumerate(columns_lower):
            if any(pattern in col for pattern in pre_patterns):
                if columns[i] in numeric_columns:
                    suggestions["pre_effect"].append(columns[i])

        # Post-effect patterns
        post_patterns = ['post_effect', 'post_value', 'effect_value', 'revenue', 'amount', 'score']
        for i, col in enumerate(columns_lower):
            if 'post_' in col or col == 'post_effect':
                if columns[i] in numeric_columns:
                    suggestions["post_effect"].append(columns[i])

        # Legacy effect patterns
        effect_patterns = ['effect', 'value', 'metric', 'outcome', 'result', 'conversion', 'revenue', 'amount', 'score']
        for i, col in enumerate(columns_lower):
            if any(pattern in col for pattern in effect_patterns):
                if columns[i] in numeric_columns and columns[i] not in suggestions["pre_effect"]:
                    suggestions["effect_value"].append(columns[i])

        # Segment patterns
        segment_patterns = ['segment', 'category', 'tier', 'type', 'cohort', 'cluster', 'group_name']
        for i, col in enumerate(columns_lower):
            if any(pattern in col for pattern in segment_patterns):
                if col not in [c.lower() for c in suggestions["group"]]:
                    suggestions["segment"].append(columns[i])

        # Duration patterns
        duration_patterns = ['duration', 'days', 'period', 'time', 'length', 'exposure']
        for i, col in enumerate(columns_lower):
            if any(pattern in col for pattern in duration_patterns):
                suggestions["duration"].append(columns[i])

        return suggestions

    def set_column_mapping(self, mapping: Dict[str, str]) -> None:
        """Set column mapping for analysis"""
        self.column_mapping = mapping

    def set_group_labels(self, treatment_label: str, control_label: str) -> None:
        """Set treatment and control group labels"""
        self.treatment_label = treatment_label
        self.control_label = control_label

    def get_group_values(self) -> Dict[str, List[Any]]:
        """Return unique values for the configured group column."""
        if self.df is None or "group" not in self.column_mapping:
            raise ValueError("Data not loaded or group column not set")

        group_col = self.column_mapping["group"]
        values = [row[0] for row in self.df.select(group_col).distinct().collect()]
        return {"group_column": group_col, "unique_values": values}

    def query_data(self, query: str):
        """Spark backend does not support pandas-query semantics."""
        raise AgentUserFacingError(
            "BACKEND_OPERATION_UNSUPPORTED",
            "Querying data is not supported for the active backend (spark).",
        )

    def get_data_summary(self) -> Dict[str, Any]:
        """Return high-level profiling summary for the active Spark DataFrame."""
        if self.df is None:
            raise ValueError("No data loaded")

        row_count = self.df.count()
        columns = list(self.df.columns)
        dtypes = {name: dtype for name, dtype in self.df.dtypes}
        missing_values = (
            self.df.select(
                [
                    F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
                    for column in columns
                ]
            )
            .first()
            .asDict()
        )

        numeric_types = {"int", "bigint", "long", "double", "float", "decimal"}
        numeric_columns = [
            name
            for name, dtype in self.df.dtypes
            if any(token in dtype.lower() for token in numeric_types)
        ]

        numeric_summary: Dict[str, Dict[str, Any]] = {}
        if numeric_columns:
            described_rows = self.df.select(*numeric_columns).describe().collect()
            described = {row["summary"]: row.asDict() for row in described_rows}
            for metric in ("mean", "stddev", "min", "max"):
                metric_name = "std" if metric == "stddev" else metric
                numeric_summary[metric_name] = {}
                metric_row = described.get(metric, {})
                for column in numeric_columns:
                    value = metric_row.get(column)
                    if value is None:
                        continue
                    try:
                        numeric_summary[metric_name][column] = float(value)
                    except (TypeError, ValueError):
                        numeric_summary[metric_name][column] = value

        sample_rows = self.df.limit(5).toPandas().to_dict()

        return {
            "shape": (row_count, len(columns)),
            "columns": columns,
            "dtypes": dtypes,
            "missing_values": missing_values,
            "numeric_summary": numeric_summary,
            "sample_rows": sample_rows,
        }

    def get_segment_distribution(self) -> Dict[str, Any]:
        """Return group and segment distribution tables."""
        if self.df is None:
            raise ValueError("No data loaded")

        result: Dict[str, Any] = {"columns_used": dict(self.column_mapping)}

        if "group" in self.column_mapping:
            group_col = self.column_mapping["group"]
            group_counts = self.df.groupBy(group_col).count().collect()
            result["group_distribution"] = {
                row[group_col]: int(row["count"]) for row in group_counts
            }

        if "segment" in self.column_mapping:
            segment_col = self.column_mapping["segment"]
            segment_counts = self.df.groupBy(segment_col).count().collect()
            result["segment_distribution"] = {
                row[segment_col]: int(row["count"]) for row in segment_counts
            }

            if "group" in self.column_mapping:
                group_col = self.column_mapping["group"]
                cross_rows = self.df.groupBy(group_col, segment_col).count().collect()
                segment_by_group: Dict[Any, Dict[Any, int]] = {}
                for row in cross_rows:
                    group_value = row[group_col]
                    segment_value = row[segment_col]
                    segment_by_group.setdefault(group_value, {})[segment_value] = int(row["count"])
                result["segment_by_group"] = segment_by_group

        return result

    def auto_configure(self) -> Dict[str, Any]:
        """
        Automatically detect and configure columns and labels
        Uses Spark operations to identify group values
        """
        if self.df is None:
            return {"success": False, "error": "No data loaded"}

        config = {"success": True, "warnings": [], "mapping": {}, "labels": {}}
        suggestions = self.detect_columns()

        # Set group column
        if suggestions["group"]:
            config["mapping"]["group"] = suggestions["group"][0]
        else:
            # Find column with exactly 2 distinct values
            for col in self.df.columns:
                distinct_count = self.df.select(col).distinct().count()
                if distinct_count == 2:
                    config["mapping"]["group"] = col
                    config["warnings"].append(f"Guessed '{col}' as group column (has 2 unique values)")
                    break

        if "group" not in config["mapping"]:
            return {"success": False, "error": "Could not detect group column"}

        # Set pre_effect if available
        if suggestions["pre_effect"]:
            config["mapping"]["pre_effect"] = suggestions["pre_effect"][0]

        # Set post_effect (required)
        if suggestions["post_effect"]:
            config["mapping"]["post_effect"] = suggestions["post_effect"][0]
            config["mapping"]["effect_value"] = suggestions["post_effect"][0]
        elif suggestions["effect_value"]:
            config["mapping"]["effect_value"] = suggestions["effect_value"][0]
            config["mapping"]["post_effect"] = suggestions["effect_value"][0]
        else:
            return {"success": False, "error": "Could not detect effect value column"}

        # Set optional columns
        if suggestions["segment"]:
            config["mapping"]["segment"] = suggestions["segment"][0]
        if suggestions["customer_id"]:
            config["mapping"]["customer_id"] = suggestions["customer_id"][0]

        self.set_column_mapping(config["mapping"])

        # Auto-detect treatment/control labels using Spark collect
        group_col = config["mapping"]["group"]
        unique_values = [row[0] for row in self.df.select(group_col).distinct().collect()]

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

    def _calculate_segment_statistics(
        self,
        segment_filter: Optional[str] = None
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Calculate statistics for treatment and control groups using Spark aggregations
        Returns (treatment_stats_df, control_stats_df)

        Each stats DataFrame contains:
        - segment, n, mean, variance, std, sum, sum_sq (for post_effect)
        - pre_mean, pre_variance, pre_std (if pre_effect exists)
        """
        if self.df is None:
            raise ValueError("No data loaded")

        group_col = self.column_mapping["group"]
        post_col = self.column_mapping.get("post_effect", self.column_mapping["effect_value"])
        pre_col = self.column_mapping.get("pre_effect")
        segment_col = self.column_mapping.get("segment")

        # Filter by segment if specified
        df_filtered = self.df
        segment_name = "Overall"

        if segment_filter and segment_col:
            df_filtered = df_filtered.filter(F.col(segment_col) == segment_filter)
            segment_name = segment_filter

        # Add segment column if not exists
        if not segment_col:
            df_filtered = df_filtered.withColumn("_segment", F.lit(segment_name))
            segment_col = "_segment"

        # Split into treatment and control
        treatment_df = df_filtered.filter(F.col(group_col) == self.treatment_label).dropna(subset=[post_col])
        control_df = df_filtered.filter(F.col(group_col) == self.control_label).dropna(subset=[post_col])

        # Build aggregation expressions for post-effect
        agg_exprs = [
            F.count(post_col).alias("n"),
            F.mean(post_col).alias("mean"),
            F.variance(post_col).alias("variance"),
            F.stddev(post_col).alias("std"),
            F.sum(post_col).alias("sum"),
            F.sum(F.col(post_col) * F.col(post_col)).alias("sum_sq"),
            # Proportion calculations (non-zero count)
            F.sum(F.when(F.col(post_col) != 0, 1).otherwise(0)).alias("conversions"),
        ]

        # Add pre-effect stats if available
        if pre_col and pre_col in df_filtered.columns:
            agg_exprs.extend([
                F.mean(pre_col).alias("pre_mean"),
                F.variance(pre_col).alias("pre_variance"),
                F.stddev(pre_col).alias("pre_std")
            ])

        # Group by segment and calculate stats
        group_by_col = segment_col if segment_filter is None else F.lit(segment_name).alias("segment")

        treatment_stats = treatment_df.groupBy(group_by_col).agg(*agg_exprs)
        control_stats = control_df.groupBy(group_by_col).agg(*agg_exprs)

        return treatment_stats, control_stats

    def _calculate_cohens_d(self, mean1: float, mean2: float, var1: float, var2: float, n1: int, n2: int) -> float:
        """Calculate Cohen's d effect size"""
        if n1 < 2 or n2 < 2:
            return 0.0

        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1e-10

        return (mean1 - mean2) / pooled_std

    def _calculate_t_test(
        self,
        mean1: float, mean2: float,
        var1: float, var2: float,
        n1: int, n2: int
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Welch's t-test for unequal variances
        Returns (t_statistic, p_value, (ci_lower, ci_upper))
        """
        if n1 < 2 or n2 < 2:
            return 0.0, 1.0, (0.0, 0.0)

        # Standard error of difference
        se = np.sqrt(var1 / n1 + var2 / n2)

        if se == 0:
            return 0.0, 1.0, (0.0, 0.0)

        # T-statistic
        t_stat = (mean1 - mean2) / se

        # Welch-Satterthwaite degrees of freedom
        df = ((var1 / n1 + var2 / n2) ** 2) / (
            (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        )

        # P-value (two-tailed)
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        # Confidence interval
        t_critical = stats.t.ppf(1 - self.significance_level / 2, df)
        ci_lower = (mean1 - mean2) - t_critical * se
        ci_upper = (mean1 - mean2) + t_critical * se

        return float(t_stat), float(p_value), (float(ci_lower), float(ci_upper))

    def _calculate_proportion_test(
        self,
        conversions1: int, n1: int,
        conversions2: int, n2: int
    ) -> Tuple[float, float, float, float]:
        """
        Two-proportion z-test using statsmodels test_proportions_2indep
        Returns (z_stat, p_value, prop_diff, pooled_proportion)
        """
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0, 0.0, 0.0

        p1 = conversions1 / n1
        p2 = conversions2 / n2
        p_diff = p1 - p2

        # Pooled proportion
        p_pooled = (conversions1 + conversions2) / (n1 + n2)

        # Use statsmodels test_proportions_2indep for two-proportion z-test
        try:
            from statsmodels.stats.proportion import test_proportions_2indep

            # Returns (z_stat, p_value) tuple
            z_stat, p_value = test_proportions_2indep(
                conversions1, n1,
                conversions2, n2,
                method='wald',
                alternative='two-sided'
            )

            return float(z_stat), float(p_value), float(p_diff), float(p_pooled)
        except Exception:
            # Fallback to manual calculation if statsmodels not available
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

            if se == 0:
                return 0.0, 1.0, p_diff, p_pooled

            z_stat = p_diff / se

            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            return float(z_stat), float(p_value), float(p_diff), float(p_pooled)

    def _calculate_power(self, effect_size: float, n1: int, n2: int) -> float:
        """Calculate statistical power"""
        if effect_size == 0 or n1 < 2 or n2 < 2:
            return 0.0

        from statsmodels.stats.power import TTestIndPower

        power_analysis = TTestIndPower()
        ratio = n2 / n1 if n1 > 0 else 1

        try:
            power = power_analysis.solve_power(
                effect_size=abs(effect_size),
                nobs1=n1,
                ratio=ratio,
                alpha=self.significance_level
            )
            return min(float(power), 1.0)
        except Exception:
            return 0.0

    def _calculate_required_sample_size(self, effect_size: float) -> int:
        """Calculate required sample size per group"""
        if effect_size == 0:
            return 999999

        from statsmodels.stats.power import TTestIndPower

        power_analysis = TTestIndPower()

        try:
            n = power_analysis.solve_power(
                effect_size=abs(effect_size),
                power=self.power_threshold,
                ratio=1.0,
                alpha=self.significance_level
            )
            return int(np.ceil(n))
        except Exception:
            return 999999

    def _run_bayesian_test_montecarlo(
        self,
        mean_t: float, var_t: float, n_t: int,
        mean_c: float, var_c: float, n_c: int,
        n_samples: int = 10000
    ) -> Dict[str, float]:
        """
        Bayesian A/B test using Monte Carlo simulation
        Uses t-distribution for small samples, normal for large
        """
        if n_t < 2 or n_c < 2:
            return {
                "prob_treatment_better": 0.5,
                "expected_loss_treatment": 0.0,
                "expected_loss_control": 0.0,
                "credible_interval": (0.0, 0.0),
                "relative_uplift": 0.0,
                "total_effect": 0.0
            }

        # Standard errors
        se_t = np.sqrt(var_t / n_t)
        se_c = np.sqrt(var_c / n_c)

        np.random.seed(42)

        # Draw samples from posterior
        if n_t > 30 and n_c > 30:
            treatment_samples = np.random.normal(mean_t, se_t, n_samples)
            control_samples = np.random.normal(mean_c, se_c, n_samples)
        else:
            treatment_samples = mean_t + se_t * np.random.standard_t(n_t - 1, n_samples)
            control_samples = mean_c + se_c * np.random.standard_t(n_c - 1, n_samples)

        diff_samples = treatment_samples - control_samples

        # Metrics
        prob_treatment_better = float(np.mean(diff_samples > 0))

        treatment_worse = diff_samples < 0
        expected_loss_treatment = float(np.mean(np.abs(diff_samples) * treatment_worse))

        control_worse = diff_samples > 0
        expected_loss_control = float(np.mean(np.abs(diff_samples) * control_worse))

        ci_lower = float(np.percentile(diff_samples, 2.5))
        ci_upper = float(np.percentile(diff_samples, 97.5))

        mean_diff = float(np.mean(diff_samples))
        relative_uplift = mean_diff / mean_c if mean_c != 0 else 0.0
        total_effect = mean_diff * n_t

        return {
            "prob_treatment_better": prob_treatment_better,
            "expected_loss_treatment": expected_loss_treatment,
            "expected_loss_control": expected_loss_control,
            "credible_interval": (ci_lower, ci_upper),
            "relative_uplift": float(relative_uplift),
            "total_effect": float(total_effect)
        }

    def run_ab_test(self, segment_filter: Optional[str] = None) -> SparkABTestResult:
        """
        Run comprehensive A/B test for a segment using Spark aggregations

        Performs:
        - Summary statistics via Spark groupBy aggregations
        - T-test for continuous metrics
        - Proportion test for conversion rates
        - Cohen's d effect size
        - Power analysis
        - Bayesian test with Monte Carlo
        - DiD if pre-effect available

        Args:
            segment_filter: Specific segment to analyze (None for overall)

        Returns:
            SparkABTestResult with comprehensive metrics
        """
        # Get aggregated statistics using Spark
        treatment_stats, control_stats = self._calculate_segment_statistics(segment_filter)

        # Collect to driver (small aggregated data)
        t_row = treatment_stats.first()
        c_row = control_stats.first()

        if t_row is None or c_row is None:
            raise ValueError(f"Insufficient data for segment '{segment_filter}'")

        segment_name = segment_filter if segment_filter else "Overall"

        # Extract statistics
        n_t = t_row["n"]
        n_c = c_row["n"]

        if n_t < 2 or n_c < 2:
            raise ValueError(f"Insufficient samples: treatment={n_t}, control={n_c}")

        # Post-effect stats
        mean_t = t_row["mean"]
        mean_c = c_row["mean"]
        var_t = t_row["variance"]
        var_c = c_row["variance"]
        conversions_t = t_row["conversions"]
        conversions_c = c_row["conversions"]

        # Pre-effect stats (if available)
        has_pre = "pre_mean" in t_row.asDict()
        if has_pre:
            pre_mean_t = t_row["pre_mean"]
            pre_mean_c = c_row["pre_mean"]
            pre_var_t = t_row["pre_variance"]
            pre_var_c = c_row["pre_variance"]
        else:
            pre_mean_t = pre_mean_c = 0.0
            pre_var_t = pre_var_c = 0.0

        # === T-TEST ===
        effect_size = mean_t - mean_c
        cohens_d = self._calculate_cohens_d(mean_t, mean_c, var_t, var_c, n_t, n_c)
        t_stat, p_value, (ci_lower, ci_upper) = self._calculate_t_test(
            mean_t, mean_c, var_t, var_c, n_t, n_c
        )
        is_significant = p_value < self.significance_level

        # === POWER ANALYSIS ===
        power = self._calculate_power(cohens_d, n_t, n_c)
        required_n = self._calculate_required_sample_size(cohens_d)
        is_sample_adequate = power >= self.power_threshold

        # === AA TEST (on pre-effect if available) ===
        aa_test_passed = True
        aa_p_value = 1.0
        if has_pre:
            _, aa_p_value, _ = self._calculate_t_test(
                pre_mean_t, pre_mean_c, pre_var_t, pre_var_c, n_t, n_c
            )
            aa_test_passed = aa_p_value > self.significance_level

        # === DIFFERENCE-IN-DIFFERENCES ===
        if has_pre:
            did_treatment_change = mean_t - pre_mean_t
            did_control_change = mean_c - pre_mean_c
            did_effect = did_treatment_change - did_control_change
        else:
            did_treatment_change = did_control_change = 0.0
            did_effect = effect_size

        # === PROPORTION TEST ===
        z_stat, prop_p_value, prop_diff, _ = self._calculate_proportion_test(
            conversions_t, n_t, conversions_c, n_c
        )
        prop_is_significant = prop_p_value < self.significance_level

        p_t = conversions_t / n_t if n_t > 0 else 0.0
        p_c = conversions_c / n_c if n_c > 0 else 0.0

        if prop_is_significant and prop_diff > 0:
            prop_effect_per_customer = prop_diff * mean_c
            prop_effect = prop_effect_per_customer * n_t
        else:
            prop_effect_per_customer = prop_effect = 0.0

        # === COMBINED EFFECT ===
        t_test_total = effect_size * n_t if is_significant else 0.0
        total_effect = t_test_total + prop_effect
        total_effect_per_customer = (effect_size if is_significant else 0.0) + prop_effect_per_customer

        # === BAYESIAN TEST ===
        bayesian_results = self._run_bayesian_test_montecarlo(
            mean_t, var_t, n_t, mean_c, var_c, n_c
        )
        bayesian_is_significant = (
            bayesian_results["prob_treatment_better"] > 0.95 or
            bayesian_results["prob_treatment_better"] < 0.05
        )

        pooled_std = np.sqrt(((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2))

        return SparkABTestResult(
            segment=segment_name,
            treatment_size=n_t,
            control_size=n_c,
            treatment_pre_mean=pre_mean_t,
            treatment_post_mean=mean_t,
            control_pre_mean=pre_mean_c,
            control_post_mean=mean_c,
            treatment_mean=mean_t,
            control_mean=mean_c,
            effect_size=effect_size,
            cohens_d=cohens_d,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=(ci_lower, ci_upper),
            pooled_std=pooled_std,
            power=power,
            required_sample_size=required_n,
            is_sample_adequate=is_sample_adequate,
            did_treatment_change=did_treatment_change,
            did_control_change=did_control_change,
            did_effect=did_effect,
            aa_test_passed=aa_test_passed,
            aa_p_value=aa_p_value,
            bootstrapping_applied=False,
            original_control_size=n_c,
            treatment_proportion=p_t,
            control_proportion=p_c,
            proportion_diff=prop_diff,
            proportion_z_stat=z_stat,
            proportion_p_value=prop_p_value,
            proportion_is_significant=prop_is_significant,
            proportion_effect=prop_effect,
            proportion_effect_per_customer=prop_effect_per_customer,
            total_effect=total_effect,
            total_effect_per_customer=total_effect_per_customer,
            bayesian_prob_treatment_better=bayesian_results["prob_treatment_better"],
            bayesian_expected_loss_treatment=bayesian_results["expected_loss_treatment"],
            bayesian_expected_loss_control=bayesian_results["expected_loss_control"],
            bayesian_credible_interval=(
                bayesian_results["credible_interval"][0],
                bayesian_results["credible_interval"][1],
            ),
            bayesian_relative_uplift=bayesian_results["relative_uplift"],
            bayesian_is_significant=bayesian_is_significant,
            bayesian_total_effect=bayesian_results["total_effect"],
            bayesian_total_effect_per_customer=bayesian_results["total_effect"] / n_t if n_t > 0 else 0.0
        )

    def run_segmented_analysis(self) -> List[SparkABTestResult]:
        """
        Run A/B tests for all segments in parallel using Spark

        Strategy:
        1. Get unique segments using Spark distinct()
        2. Process each segment (Spark handles parallelization)
        3. Return consolidated results

        Returns:
            List of SparkABTestResult for each segment
        """
        self.segment_failures = []

        if "segment" not in self.column_mapping:
            # No segmentation - run overall analysis
            return [self.run_ab_test()]

        segment_col = self.column_mapping["segment"]

        # Get unique segments using Spark
        segments = [row[0] for row in self.df.select(segment_col).distinct().collect()]

        results = []
        for segment in segments:
            try:
                result = self.run_ab_test(segment_filter=segment)
                results.append(result)
            except ValueError as e:
                failure = {
                    "segment": str(segment),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                self.segment_failures.append(failure)
                logger.warning(
                    "Skipping segment during Spark segmented analysis",
                    extra={"segment": str(segment), "error": str(e)},
                )

        return results

    def generate_summary(self, results: List[SparkABTestResult]) -> Dict[str, Any]:
        """
        Generate summary statistics using canonical summary builder.

        Args:
            results: List of SparkABTestResult objects

        Returns:
            Dictionary with aggregated metrics and recommendations
        """
        return self.summary_builder.generate_summary(
            results,
            segment_failures=self.segment_failures,
        )

    def _generate_recommendations(self, results: List[SparkABTestResult]) -> List[str]:
        """Backward-compatible recommendation helper."""
        return self.summary_builder._generate_recommendations(results)

    def save_results_to_parquet(self, results: List[SparkABTestResult], path: str):
        """
        Save results to Parquet format using Spark

        Args:
            results: List of analysis results
            path: Output path (supports S3, HDFS, etc.)
        """
        results_dicts = [r.to_serializable_dict() for r in results]
        results_df = self.spark.createDataFrame(results_dicts)
        results_df.write.mode("overwrite").parquet(path)
        logger.info("Saved Spark results to Parquet (path=%s, rows=%d)", path, len(results_dicts))

    def save_results_to_delta(self, results: List[SparkABTestResult], path: str, partition_by: Optional[List[str]] = None):
        """
        Save results to Delta Lake format for ACID transactions and time travel

        Args:
            results: List of analysis results
            path: Output path
            partition_by: Optional columns to partition by (e.g., ["segment"])
        """
        results_dicts = [r.to_serializable_dict() for r in results]
        results_df = self.spark.createDataFrame(results_dicts)

        writer = results_df.write.format("delta").mode("overwrite")

        if partition_by:
            writer = writer.partitionBy(*partition_by)

        writer.save(path)
        logger.info(
            "Saved Spark results to Delta (path=%s, rows=%d, partition_by=%s)",
            path,
            len(results_dicts),
            partition_by or [],
        )


def create_spark_session(
    app_name: str = "ABTestingAnalyzer",
    master: str = "local[*]",
    config: Optional[Dict[str, str]] = None
) -> SparkSession:
    """
    Create and configure a SparkSession for A/B testing analysis

    Args:
        app_name: Name of the Spark application
        master: Spark master URL (local[*], yarn, etc.)
        config: Additional Spark configuration options

    Returns:
        Configured SparkSession

    Example:
        >>> spark = create_spark_session(
        ...     app_name="MyABTest",
        ...     config={
        ...         "spark.sql.shuffle.partitions": "200",
        ...         "spark.executor.memory": "4g"
        ...     }
        ... )
    """
    builder = SparkSession.builder.appName(app_name).master(master)

    # Default optimizations for A/B testing workloads
    default_config = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
    }

    if config:
        default_config.update(config)

    for key, value in default_config.items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


# Example usage
if __name__ == "__main__":
    """
    Example: Running distributed A/B test analysis with PySpark
    """

    # Create Spark session
    spark = create_spark_session(
        app_name="ABTestExample",
        config={
            "spark.sql.shuffle.partitions": "100",
            "spark.executor.memory": "8g",
            "spark.driver.memory": "4g"
        }
    )

    # Initialize analyzer
    analyzer = PySparkABTestAnalyzer(spark)

    # Load data (supports CSV, Parquet, Delta, S3, HDFS, etc.)
    analyzer.load_data(
        "data/sample_ab_data.csv",
        format="csv",
        header=True,
        inferSchema=True
    )

    # Auto-configure
    config = analyzer.auto_configure()
    logger.info("Auto-configuration: %s", json.dumps(config, indent=2))

    # Run segmented analysis (distributed across Spark cluster)
    results = analyzer.run_segmented_analysis()

    # Generate summary
    summary = analyzer.generate_summary(results)
    logger.info(
        "Spark example summary: segments=%s significant=%s combined_effect=%.2f",
        summary["total_segments_analyzed"],
        summary["t_test_significant_segments"],
        summary["combined_total_effect"],
    )

    # Save results to Parquet for downstream analysis
    analyzer.save_results_to_parquet(results, "output/ab_test_results.parquet")

    # Optional: Save to Delta Lake for ACID properties
    # analyzer.save_results_to_delta(results, "output/ab_test_results_delta", partition_by=["segment"])

    # Stop Spark session
    spark.stop()

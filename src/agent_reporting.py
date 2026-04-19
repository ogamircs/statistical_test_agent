"""Report rendering helpers for ABTestingAgent tool outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.statistics.models import to_ab_test_summary


@dataclass(frozen=True)
class StructuredAgentError:
    """Normalized user-facing error payload."""

    code: str
    message: str


class AgentUserFacingError(Exception):
    """Exception with an explicit safe user message + machine-readable code."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.user_message = message
        super().__init__(message)


def classify_agent_error(
    error: Exception,
    *,
    default_code: str = "INTERNAL_ERROR",
    default_message: str = "An unexpected error occurred.",
) -> StructuredAgentError:
    """Classify runtime exceptions into stable codes/messages for the UI."""
    if isinstance(error, AgentUserFacingError):
        return StructuredAgentError(code=error.code, message=error.user_message)

    # Support external/user-defined errors that expose the same shape.
    code = getattr(error, "code", None)
    user_message = getattr(error, "user_message", None)
    if isinstance(code, str) and isinstance(user_message, str):
        return StructuredAgentError(code=code, message=user_message)

    if isinstance(error, FileNotFoundError):
        return StructuredAgentError(
            code="FILE_NOT_FOUND",
            message="File not found. Please verify the file path and try again.",
        )
    if isinstance(error, PermissionError):
        return StructuredAgentError(
            code="FILE_ACCESS_DENIED",
            message="Unable to access the file due to permissions.",
        )
    if isinstance(error, TimeoutError):
        return StructuredAgentError(
            code="TIMEOUT",
            message="The operation timed out. Please try again.",
        )

    if isinstance(error, ValueError):
        message = str(error).strip() or default_message
        normalized = message.lower()

        if "no data loaded" in normalized:
            code = "DATA_NOT_LOADED"
        elif "column mapping" in normalized:
            code = "COLUMN_MAPPING_NOT_SET"
        elif "treatment/control labels not set" in normalized or "group labels not set" in normalized:
            code = "GROUP_LABELS_NOT_SET"
        elif "insufficient data" in normalized:
            code = "INSUFFICIENT_DATA"
        elif "query" in normalized:
            code = "INVALID_QUERY"
        else:
            code = "INVALID_INPUT"

        return StructuredAgentError(code=code, message=message)

    return StructuredAgentError(code=default_code, message=default_message)


def render_tool_error(
    prefix: str,
    error: Exception,
    *,
    default_code: str = "INTERNAL_ERROR",
    default_message: str = "An unexpected error occurred.",
) -> str:
    """Render stable/safe user-facing tool errors."""
    structured = classify_agent_error(
        error,
        default_code=default_code,
        default_message=default_message,
    )
    return f"{prefix}: {structured.message} [error_code={structured.code}]"


def render_load_csv_success(
    *,
    filepath: str,
    file_size_mb: float,
    backend: str,
    file_size_threshold_mb: float,
    spark_selected: bool,
    fallback_note: Optional[str],
    shape: Tuple[int, int],
    columns: Sequence[str],
    suggestions: Mapping[str, Sequence[str]],
) -> str:
    """Render successful load_csv output."""
    result = f"File size: {file_size_mb:.2f} MB\n"

    if backend == "spark":
        result += (
            f"[LARGE FILE DETECTED] Using PySpark for distributed processing "
            f"(file size > {file_size_threshold_mb}MB)\n\n"
        )
    else:
        if spark_selected:
            result += (
                f"[LARGE FILE DETECTED] PySpark requested for files > "
                f"{file_size_threshold_mb}MB\n"
            )
        result += "Using pandas for in-memory processing\n"
        if fallback_note:
            result += f"{fallback_note}\n"
        result += "\n"

    result += f"Successfully loaded data from '{filepath}'\n"
    result += f"Shape: {shape[0]:,} rows, {shape[1]} columns\n\n"
    result += f"Columns found: {', '.join(columns)}\n\n"
    result += "Column suggestions based on naming patterns:\n"

    for col_type, suggested in suggestions.items():
        if suggested:
            result += f"  - {col_type}: {', '.join(suggested)}\n"
        else:
            result += f"  - {col_type}: No automatic match found\n"

    result += "\nPlease confirm or specify the correct column mappings."
    return result


def _render_ab_results_section(summary: Any) -> str:
    """Render the shared markdown A/B results section."""
    normalized = to_ab_test_summary(summary)
    output = "## A/B Test Results\n\n"

    output += "### Overview\n"
    output += f"- **Segments Analyzed:** {normalized.total_segments_analyzed}\n"
    output += f"- **AA Test Passed:** {normalized.aa_test_passed_segments}\n"
    output += f"- **AA Test Failed:** {normalized.aa_test_failed_segments}\n"
    output += f"- **Bootstrapped Segments:** {normalized.bootstrapped_segments}\n"
    output += (
        f"- **T-test Significant:** {normalized.t_test_significant_segments} "
        f"({normalized.t_test_significance_rate:.1%})\n"
    )
    output += (
        f"- **Proportion Test Significant:** {normalized.prop_test_significant_segments} "
        f"({normalized.prop_test_significance_rate:.1%})\n"
    )
    output += (
        f"- **Bayesian Significant:** {normalized.bayesian_significant_segments} "
        f"({normalized.bayesian_significance_rate:.1%})\n\n"
    )

    output += "### Sample Information\n"
    output += f"- **Total Treatment:** {normalized.total_treatment_customers:,}\n"
    output += f"- **Total Control:** {normalized.total_control_customers:,}\n\n"

    output += "### Effect Summary\n"
    output += f"- **DiD Avg Effect:** {normalized.did_avg_effect:.4f}\n"
    output += f"- **DiD Total Effect:** {normalized.did_total_effect:.2f}\n"
    output += f"- **T-test Effect:** {normalized.t_test_effect_calculation}\n"
    output += f"- **Proportion Effect:** {normalized.prop_test_effect_calculation}\n"
    output += f"- **Combined Total Effect:** {normalized.combined_effect_calculation}\n"
    output += f"- **Bayesian Total Effect:** {normalized.bayesian_total_effect:.2f}\n"
    output += (
        f"- **Avg P(Treatment Better):** "
        f"{normalized.bayesian_avg_prob_treatment_better:.1%}\n"
    )
    output += f"- **Avg Expected Loss:** {normalized.bayesian_avg_expected_loss:.4f}\n\n"

    if normalized.analysis_warnings:
        output += "### Analysis Warnings\n"
        for warning in normalized.analysis_warnings:
            output += f"- {warning}\n"
        output += "\n"

    if normalized.segment_failures:
        output += "### Skipped Segments\n\n"
        output += "| Segment | Reason |\n"
        output += "|---------|--------|\n"
        for failure in normalized.segment_failures:
            output += f"| {failure.segment} | {failure.error} |\n"
        output += "\n"

    output += "### AA Test & Pre/Post Analysis\n\n"
    output += "| Segment | AA Pass | Boot | Pre Treat | Pre Ctrl | Post Treat | Post Ctrl | DiD Effect |\n"
    output += "|---------|---------|------|-----------|----------|------------|-----------|------------|\n"

    for result in normalized.detailed_results:
        aa_pass = "Yes" if result.aa_test_passed else "No"
        boot = "Yes" if result.bootstrapping_applied else "No"
        output += (
            f"| {result.segment} | {aa_pass} | {boot} | "
            f"{result.treatment_pre_mean:.2f} | {result.control_pre_mean:.2f} | "
            f"{result.treatment_post_mean:.2f} | {result.control_post_mean:.2f} | "
            f"{result.did_effect:.4f} |\n"
        )

    output += "\n"

    output += "### Frequentist Results\n\n"
    output += "| Segment | Treat N | Ctrl N | T-test p-val | T-test Effect | Prop p-val | Prop Effect | Total Effect |\n"
    output += "|---------|---------|--------|--------------|---------------|------------|-------------|-------------|\n"

    for result in normalized.detailed_results:
        t_effect = result.effect_size
        t_pval = result.p_value
        prop_pval = result.proportion_p_value
        prop_effect_per_cust = result.proportion_effect_per_customer

        t_total = t_effect * result.treatment_size if result.is_significant else 0
        prop_total = (
            prop_effect_per_cust * result.control_size
            if result.proportion_is_significant
            else 0
        )
        total_effect = t_total + prop_total

        t_sig_marker = "*" if result.is_significant else ""
        p_sig_marker = "*" if result.proportion_is_significant else ""

        output += (
            f"| {result.segment} | {result.treatment_size:,} | {result.control_size:,} | "
            f"{t_pval:.4f}{t_sig_marker} | {t_effect:.4f} | "
            f"{prop_pval:.4f}{p_sig_marker} | {prop_effect_per_cust:.4f} | {total_effect:.2f} |\n"
        )

    output += "\n*\\* indicates statistical significance (p < 0.05)*\n\n"

    output += "### Bayesian Results (with DiD Total Effect)\n\n"
    output += "| Segment | P(Treat>Ctrl) | 95% Credible Interval | Expected Loss | Bayesian Total Effect |\n"
    output += "|---------|---------------|----------------------|---------------|----------------------|\n"

    for result in normalized.detailed_results:
        expected_loss = min(
            result.bayesian_expected_loss_treatment,
            result.bayesian_expected_loss_control,
        )
        b_sig_marker = "*" if result.bayesian_is_significant else ""

        output += (
            f"| {result.segment} | {result.bayesian_prob_treatment_better:.1%}{b_sig_marker} | "
            f"[{result.bayesian_credible_interval[0]:.4f}, {result.bayesian_credible_interval[1]:.4f}] | "
            f"{expected_loss:.4f} | {result.bayesian_total_effect:.2f} |\n"
        )

    output += "\n*\\* indicates Bayesian significance (P > 95% or P < 5%)*\n"

    diagnostic_rows = []
    for result in normalized.detailed_results:
        findings = _format_segment_diagnostics(result)
        if findings:
            diagnostic_rows.append((result.segment, findings))

    if diagnostic_rows:
        output += "\n### Diagnostics\n\n"
        output += "Failed assumption checks per segment. Consider robust or non-parametric methods when these fire.\n\n"
        output += "| Segment | Findings |\n"
        output += "|---------|----------|\n"
        for segment, findings in diagnostic_rows:
            output += f"| {segment} | {' '.join(findings)} |\n"
        output += "\n"

    output += "\n### Recommendations\n\n"
    for i, rec in enumerate(normalized.recommendations, 1):
        output += f"{i}. {rec}\n"

    output += "\n---\n*Would you like to see the visualizations?*"

    return output


def render_load_and_auto_analyze_report(
    *,
    filepath: str,
    file_size_mb: float,
    shape: Tuple[int, int],
    backend: str,
    fallback_note: Optional[str],
    config: Dict[str, Any],
    summary: Any,
) -> str:
    """Render load_and_auto_analyze output."""
    output = "## Best Guess Mode - Analysis Complete\n\n"

    if backend == "spark":
        output += "**Backend:** PySpark (distributed processing for large files)\n"
    else:
        output += "**Backend:** pandas (in-memory processing)\n"
    if fallback_note:
        output += f"**Backend Note:** {fallback_note}\n"

    output += f"**File:** {filepath.split('/')[-1].split(chr(92))[-1]}\n"
    output += f"**File Size:** {file_size_mb:.2f} MB\n"
    output += f"**Shape:** {shape[0]:,} rows × {shape[1]} columns\n\n"

    output += "### Auto-Detected Configuration\n"
    output += "| Setting | Value |\n|---------|-------|\n"
    for key, value in config["mapping"].items():
        output += f"| {key} | `{value}` |\n"
    output += f"| Treatment Label | `{config['labels'].get('treatment', 'N/A')}` |\n"
    output += f"| Control Label | `{config['labels'].get('control', 'N/A')}` |\n\n"

    if config["warnings"]:
        output += "**Warnings:**\n"
        for warning in config["warnings"]:
            output += f"- {warning}\n"
        output += "\n"

    output += "---\n\n"
    output += _render_ab_results_section(summary)
    return output


def render_configure_and_analyze_report(
    *,
    group_column: str,
    effect_column: str,
    treatment_label: str,
    control_label: str,
    segment_column: Optional[str],
    summary: Any,
) -> str:
    """Render configure_and_analyze output."""
    output = "## Configuration Applied\n\n"
    output += "| Setting | Value |\n|---------|-------|\n"
    output += f"| Group Column | `{group_column}` |\n"
    output += f"| Effect Column | `{effect_column}` |\n"
    output += f"| Treatment Label | `{treatment_label}` |\n"
    output += f"| Control Label | `{control_label}` |\n"
    if segment_column:
        output += f"| Segment Column | `{segment_column}` |\n"
    output += "\n---\n\n"

    output += _render_ab_results_section(summary)
    return output


def render_auto_configure_and_analyze_report(config: Dict[str, Any], summary: Any) -> str:
    """Render auto_configure_and_analyze output."""
    output = "## Best Guess Mode - Analysis Complete\n\n"

    output += "### Auto-Detected Configuration\n"
    output += "| Setting | Value |\n|---------|-------|\n"
    for key, value in config["mapping"].items():
        output += f"| {key} | `{value}` |\n"
    output += f"| Treatment Label | `{config['labels'].get('treatment', 'N/A')}` |\n"
    output += f"| Control Label | `{config['labels'].get('control', 'N/A')}` |\n\n"

    if config["warnings"]:
        output += "**Warnings:**\n"
        for warning in config["warnings"]:
            output += f"- {warning}\n"
        output += "\n"

    output += "---\n\n"
    output += _render_ab_results_section(summary)
    return output


def render_set_column_mapping_success(
    mapping: Mapping[str, str], group: str, group_info: Mapping[str, Any]
) -> str:
    """Render successful set_column_mapping output."""
    result = "Column mapping set successfully:\n"
    for key, value in mapping.items():
        result += f"  - {key}: {value}\n"

    if group_info.get("unique_values"):
        result += f"\nUnique values in '{group}' column: {group_info['unique_values']}\n"
        result += "Please specify which value represents 'treatment' and which represents 'control'."

    return result


def _format_segment_diagnostics(result: Any) -> List[str]:
    """Return one human-readable line per failed assumption check.

    Pulls from result.diagnostics["experiment_quality"]["assumptions"] and
    ["outlier_sensitivity"]. Returns an empty list when no checks fired or
    none failed — callers can use that to skip rendering.
    """
    findings: List[str] = []
    diagnostics = getattr(result, "diagnostics", None) or {}
    quality = diagnostics.get("experiment_quality", {}) if isinstance(diagnostics, dict) else {}
    assumptions = quality.get("assumptions", {}) if isinstance(quality, dict) else {}
    outlier = quality.get("outlier_sensitivity", {}) if isinstance(quality, dict) else {}

    if assumptions:
        treat_norm = assumptions.get("treatment_normality_passed")
        ctrl_norm = assumptions.get("control_normality_passed")
        if treat_norm is False or ctrl_norm is False:
            sides = []
            if treat_norm is False:
                sides.append("treatment")
            if ctrl_norm is False:
                sides.append("control")
            findings.append(
                f"Normality violated for {' and '.join(sides)}; "
                "prefer non-parametric or robust tests."
            )

        eq_var = assumptions.get("equal_variance_passed")
        if eq_var is False:
            findings.append(
                "Equal-variance assumption rejected (Levene); "
                "Welch's t-test is recommended (already used by default)."
            )

    if outlier and outlier.get("is_sensitive"):
        sens = outlier.get("sensitivity_score")
        sens_str = f"{sens:.2%}" if isinstance(sens, (int, float)) else "high"
        findings.append(
            f"Effect is sensitive to outliers (sensitivity={sens_str}); "
            "consider winsorized or trimmed estimates."
        )

    return findings


def render_run_ab_test_output(result: Any) -> str:
    """Render run_ab_test output."""
    output = f"\n{'=' * 60}\n"
    output += f"A/B TEST RESULTS - {result.segment}\n"
    output += f"{'=' * 60}\n\n"

    output += "Sample Sizes:\n"
    output += f"  Treatment: {result.treatment_size}\n"
    output += f"  Control: {result.control_size}\n\n"

    output += "Means:\n"
    output += f"  Treatment: {result.treatment_mean:.4f}\n"
    output += f"  Control: {result.control_mean:.4f}\n\n"

    output += "Effect Size:\n"
    output += f"  Absolute Difference: {result.effect_size:.4f}\n"
    output += f"  Cohen's d: {result.cohens_d:.4f}\n"
    output += f"  95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]\n\n"

    output += "Statistical Test:\n"
    output += f"  t-statistic: {result.t_statistic:.4f}\n"
    output += f"  p-value: {result.p_value:.6f}\n"
    output += f"  Significant (p < 0.05): {'YES' if result.is_significant else 'NO'}\n\n"

    output += "Bayesian Test:\n"
    output += f"  P(Treatment > Control): {result.bayesian_prob_treatment_better:.1%}\n"
    output += (
        f"  95% Credible Interval: "
        f"[{result.bayesian_credible_interval[0]:.4f}, "
        f"{result.bayesian_credible_interval[1]:.4f}]\n"
    )
    output += (
        f"  Bayesian Significant (P > 95% or P < 5%): "
        f"{'YES' if result.bayesian_is_significant else 'NO'}\n\n"
    )

    output += "Power Analysis:\n"
    output += f"  Statistical Power: {result.power:.2%}\n"
    output += f"  Required Sample Size (per group): {result.required_sample_size}\n"
    output += f"  Sample Adequate: {'YES' if result.is_sample_adequate else 'NO'}\n"

    diagnostics_lines = _format_segment_diagnostics(result)
    if diagnostics_lines:
        output += "\nDiagnostics:\n"
        for line in diagnostics_lines:
            output += f"  - {line}\n"

    return output


def render_full_analysis_output(summary: Any) -> str:
    """Render run_full_analysis output."""
    normalized = to_ab_test_summary(summary)
    output = f"\n{'=' * 60}\n"
    output += "FULL A/B TEST ANALYSIS SUMMARY\n"
    output += f"{'=' * 60}\n\n"

    output += "OVERVIEW:\n"
    output += f"  Segments Analyzed: {normalized.total_segments_analyzed}\n"
    output += f"  Significant Results: {normalized.significant_segments}\n"
    output += f"  Non-Significant: {normalized.non_significant_segments}\n"
    output += f"  Significance Rate: {normalized.significance_rate:.1%}\n\n"

    output += "SAMPLE INFORMATION:\n"
    output += f"  Total Treatment Customers: {normalized.total_treatment_customers}\n"
    output += f"  Total Control Customers: {normalized.total_control_customers}\n"
    if normalized.treatment_control_ratio:
        output += f"  Treatment/Control Ratio: {normalized.treatment_control_ratio:.2f}\n\n"

    output += "EFFECT SIZE SUMMARY:\n"
    output += f"  Average Significant Effect: {normalized.average_significant_effect:.4f}\n"
    output += (
        f"  Treatment in Significant Segments: "
        f"{normalized.total_treatment_in_significant_segments}\n"
    )
    output += f"  Total Effect Size: {normalized.effect_calculation}\n\n"

    output += "POWER ANALYSIS:\n"
    output += f"  Segments with Adequate Power: {normalized.segments_with_adequate_power}\n"
    output += (
        f"  Segments with Inadequate Power: {normalized.segments_with_inadequate_power}\n"
    )
    output += f"  Power Adequacy Rate: {normalized.power_adequacy_rate:.1%}\n\n"

    output += "SEGMENT DETAILS:\n"
    output += "-" * 100 + "\n"
    output += f"{'Segment':<20} {'Treat N':<10} {'Ctrl N':<10} {'Effect':<12} {'p-value':<12} {'Sig?':<6} {'Power':<8} {'Adequate?':<10}\n"
    output += "-" * 100 + "\n"

    for result in normalized.detailed_results:
        sig = "YES" if result.is_significant else "NO"
        adeq = "YES" if result.is_sample_adequate else "NO"
        output += (
            f"{result.segment:<20} {result.treatment_size:<10} {result.control_size:<10} "
            f"{result.effect_size:<12.4f} {result.p_value:<12.6f} {sig:<6} "
            f"{result.power:<8.2%} {adeq:<10}\n"
        )

    output += "-" * 100 + "\n\n"

    diagnostic_rows = []
    for result in normalized.detailed_results:
        findings = _format_segment_diagnostics(result)
        if findings:
            diagnostic_rows.append((result.segment, findings))

    if diagnostic_rows:
        output += "### Diagnostics\n"
        output += (
            "Failed assumption checks per segment. Consider robust or "
            "non-parametric methods when these fire.\n"
        )
        for segment, findings in diagnostic_rows:
            output += f"  - {segment}: {' '.join(findings)}\n"
        output += "\n"

    output += "RECOMMENDATIONS:\n"
    for i, rec in enumerate(normalized.recommendations, 1):
        output += f"  {i}. {rec}\n\n"

    return output


def render_query_data_output(result: Any) -> str:
    """Render query_data output."""
    return f"Query result ({len(result)} rows):\n{result.head(20).to_string()}\n\n(Showing first 20 rows)"


def render_data_question_output(answer: Any) -> str:
    """Render formatted output for natural-language data questions."""
    if isinstance(answer, Mapping):
        answer_text = str(answer.get("answer_text", "I found a result."))
        source_tables = answer.get("source_tables", [])
        sql = str(answer.get("sql", "")).strip()
        data = answer.get("data")
    else:
        answer_text = str(getattr(answer, "answer_text", "I found a result."))
        source_tables = getattr(answer, "source_tables", [])
        sql = str(getattr(answer, "sql", "")).strip()
        data = getattr(answer, "data", None)

    output = "## Data Answer\n\n"
    output += f"{answer_text}\n\n"

    if source_tables:
        output += f"**Source Tables:** {', '.join(f'`{table}`' for table in source_tables)}\n\n"

    if isinstance(data, pd.DataFrame) and not data.empty:
        output += "### Results\n\n"
        output += data.head(20).to_markdown(index=False)
        output += "\n\n"
    elif isinstance(data, pd.DataFrame):
        output += "_No matching rows returned._\n\n"

    if sql:
        output += "### SQL Used\n\n```sql\n"
        output += sql
        output += "\n```\n"

    return output


def render_data_summary_output(summary: Dict[str, Any]) -> str:
    """Render get_data_summary output."""
    output = "DATA SUMMARY\n"
    output += f"{'=' * 50}\n\n"
    output += f"Shape: {summary['shape'][0]} rows x {summary['shape'][1]} columns\n\n"

    output += "Columns and Types:\n"
    for col, dtype in summary["dtypes"].items():
        missing = summary["missing_values"].get(col, 0)
        output += f"  - {col}: {dtype} (missing: {missing})\n"

    output += "\nNumeric Column Statistics:\n"
    numeric_stats = summary["numeric_summary"]
    if numeric_stats:
        for col in list(numeric_stats.get("mean", {}).keys())[:5]:
            output += f"\n  {col}:\n"
            output += f"    Mean: {numeric_stats['mean'].get(col, 'N/A'):.2f}\n"
            output += f"    Std: {numeric_stats['std'].get(col, 'N/A'):.2f}\n"
            output += f"    Min: {numeric_stats['min'].get(col, 'N/A'):.2f}\n"
            output += f"    Max: {numeric_stats['max'].get(col, 'N/A'):.2f}\n"

    return output


def render_segment_distribution_output(dist: Dict[str, Any]) -> str:
    """Render get_segment_distribution output."""
    output = "SEGMENT & GROUP DISTRIBUTION\n"
    output += f"{'=' * 50}\n\n"

    if "group_distribution" in dist:
        output += "Group Distribution:\n"
        for group, count in dist["group_distribution"].items():
            output += f"  - {group}: {count}\n"
        output += "\n"

    if "segment_distribution" in dist:
        output += "Segment Distribution:\n"
        for segment, count in dist["segment_distribution"].items():
            output += f"  - {segment}: {count}\n"
        output += "\n"

    if "segment_by_group" in dist:
        output += "Segment x Group Cross-tabulation:\n"
        for group, segments in dist["segment_by_group"].items():
            output += f"\n  {group}:\n"
            for segment, count in segments.items():
                output += f"    - {segment}: {count}\n"

    return output


def render_column_values_output(column_name: str, values: Sequence[Any], value_counts: Any) -> str:
    """Render get_column_values output."""
    output = f"Unique values in '{column_name}' ({len(values)} unique):\n"
    for val, count in value_counts.head(20).items():
        output += f"  - {val}: {count}\n"

    if len(values) > 20:
        output += f"  ... and {len(values) - 20} more values"

    return output


def render_calculate_stats_output(column_name: str, col: Any) -> str:
    """Render calculate_statistics output."""
    output = f"Statistics for '{column_name}':\n"
    output += f"  Count: {col.count()}\n"
    output += f"  Mean: {col.mean():.4f}\n"
    output += f"  Median: {col.median():.4f}\n"
    output += f"  Std Dev: {col.std():.4f}\n"
    output += f"  Min: {col.min():.4f}\n"
    output += f"  Max: {col.max():.4f}\n"
    output += f"  25th Percentile: {col.quantile(0.25):.4f}\n"
    output += f"  75th Percentile: {col.quantile(0.75):.4f}\n"
    output += f"  Missing Values: {col.isna().sum()}\n"

    return output


def render_generate_charts_output(chart_names: Sequence[str]) -> str:
    """Render generate_charts output."""
    return f"Generated {len(chart_names)} chart(s): {', '.join(chart_names)}. The charts are now displayed in the UI."

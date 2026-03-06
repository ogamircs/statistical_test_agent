"""Tool handlers for ABTestingAgent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple

from langchain_core.tools import StructuredTool, Tool

from .agent_reporting import (
    AgentUserFacingError,
    render_auto_configure_and_analyze_report,
    render_calculate_stats_output,
    render_column_values_output,
    render_configure_and_analyze_report,
    render_data_question_output,
    render_data_summary_output,
    render_full_analysis_output,
    render_generate_charts_output,
    render_load_and_auto_analyze_report,
    render_load_csv_success,
    render_query_data_output,
    render_run_ab_test_output,
    render_segment_distribution_output,
    render_set_column_mapping_success,
    render_tool_error,
)
from .statistics.models import to_ab_test_summary

logger = logging.getLogger(__name__)


class _AgentProtocol(Protocol):
    analyzer: Any
    data_question_service: Any
    visualizer: Any
    _last_results: Any
    _last_summary: Any
    _last_charts: Dict[str, Any]
    FILE_SIZE_THRESHOLD_MB: float

    def _load_data_with_backend(self, filepath: str) -> Tuple[Any, Dict[str, Any], str, float, bool, Optional[str]]:
        ...

    def _normalize_shape(self, info: Dict[str, Any]) -> Tuple[int, int]:
        ...

    def _get_active_analyzer(self) -> Any:
        ...

    def persist_loaded_data(self, analyzer: Any) -> bool:
        ...

    def persist_analysis_outputs(self, results: Any, summary: Any) -> None:
        ...


def create_agent_tools(agent: _AgentProtocol) -> List[Tool]:
    """Create tools for the agent using modular handlers + reporting renderers."""

    def _active_analyzer() -> Any:
        return agent._get_active_analyzer()

    def _backend_label(analyzer: Any) -> str:
        analyzer_name = type(analyzer).__name__.lower()
        return "spark" if "spark" in analyzer_name else "pandas"

    def _unsupported(operation: str, analyzer: Any) -> AgentUserFacingError:
        return AgentUserFacingError(
            "BACKEND_OPERATION_UNSUPPORTED",
            f"{operation} is not supported for the active backend ({_backend_label(analyzer)}).",
        )

    def _require_pandas_dataframe(analyzer: Any, operation: str) -> Any:
        df = getattr(analyzer, "df", None)
        if df is None:
            raise ValueError("No data loaded")
        if hasattr(df, "groupBy") and hasattr(df, "select"):
            raise _unsupported(operation, analyzer)
        return df

    def _handle_tool_error(
        *,
        operation: str,
        prefix: str,
        error: Exception,
        default_code: str = "TOOL_EXECUTION_FAILED",
        default_message: str = "Unable to complete that operation.",
    ) -> str:
        logger.exception("Tool failure: %s", operation)
        return render_tool_error(
            prefix,
            error,
            default_code=default_code,
            default_message=default_message,
        )

    def load_csv(filepath: str) -> str:
        """Load a CSV file for analysis"""
        logger.info("Tool load_csv started (file=%s)", filepath)
        try:
            analyzer, info, backend, file_size_mb, spark_selected, fallback_note = agent._load_data_with_backend(filepath)
            shape = agent._normalize_shape(info)
            columns = info["columns"]
            suggestions = analyzer.detect_columns()
            agent.persist_loaded_data(analyzer)
            logger.info(
                "Tool load_csv completed (backend=%s, rows=%s, cols=%s)",
                backend,
                shape[0],
                shape[1],
            )

            return render_load_csv_success(
                filepath=filepath,
                file_size_mb=file_size_mb,
                backend=backend,
                file_size_threshold_mb=agent.FILE_SIZE_THRESHOLD_MB,
                spark_selected=spark_selected,
                fallback_note=fallback_note,
                shape=shape,
                columns=columns,
                suggestions=suggestions,
            )

        except Exception as e:
            return _handle_tool_error(
                operation="load_csv",
                prefix="Error loading file",
                error=e,
                default_code="LOAD_CSV_FAILED",
                default_message="Unable to load CSV data.",
            )

    load_csv_tool = Tool(
        name="load_csv",
        func=load_csv,
        description="Load a CSV file for A/B test analysis. Input should be the file path. Use this when you need to inspect the data before analysis. For best-guess mode, use load_and_auto_analyze instead.",
    )

    def load_and_auto_analyze(filepath: str) -> str:
        """Load CSV and automatically run full analysis using best guesses"""
        logger.info("Tool load_and_auto_analyze started (file=%s)", filepath)
        try:
            analyzer, info, backend, file_size_mb, _spark_selected, fallback_note = agent._load_data_with_backend(filepath)
            shape = agent._normalize_shape(info)

            config = analyzer.auto_configure()

            if not config["success"]:
                return f"Loaded file but auto-configuration failed: {config.get('error', 'Unknown error')}"

            results = analyzer.run_segmented_analysis()
            summary = analyzer.generate_summary(results)
            agent.persist_loaded_data(analyzer)
            agent.persist_analysis_outputs(results, summary)
            logger.info(
                "Tool load_and_auto_analyze completed (backend=%s, segments=%s)",
                backend,
                summary.get("total_segments_analyzed"),
            )

            agent._last_results = results
            agent._last_summary = summary

            return render_load_and_auto_analyze_report(
                filepath=filepath,
                file_size_mb=file_size_mb,
                shape=shape,
                backend=backend,
                fallback_note=fallback_note,
                config=config,
                summary=summary,
            )

        except Exception as e:
            return _handle_tool_error(
                operation="load_and_auto_analyze",
                prefix="Error in load and auto-analyze",
                error=e,
                default_code="LOAD_AND_ANALYZE_FAILED",
                default_message="Unable to complete automatic analysis.",
            )

    load_auto_analyze_tool = Tool(
        name="load_and_auto_analyze",
        func=load_and_auto_analyze,
        description="""Load a CSV file AND automatically run full A/B test analysis using best guesses - ALL IN ONE STEP.
Use this when the user says 'best guess', 'auto', 'automatic', 'figure it out', or wants quick analysis without manual configuration.
Input: file path. This is the FASTEST way to get results.""",
    )

    def set_column_mapping(
        customer_id: Optional[str] = None,
        group: str = "",
        effect_value: str = "",
        segment: Optional[str] = None,
        duration: Optional[str] = None,
    ) -> str:
        """Set the column mapping for analysis"""
        logger.info("Tool set_column_mapping started")
        mapping = {}
        if customer_id:
            mapping["customer_id"] = customer_id
        if group:
            mapping["group"] = group
        if effect_value:
            mapping["effect_value"] = effect_value
        if segment:
            mapping["segment"] = segment
        if duration:
            mapping["duration"] = duration

        try:
            analyzer = agent._get_active_analyzer()
            analyzer.set_column_mapping(mapping)
            logger.info("Tool set_column_mapping completed (fields=%s)", sorted(mapping.keys()))

            group_info = analyzer.get_group_values() if hasattr(analyzer, "get_group_values") else {"unique_values": []}
            return render_set_column_mapping_success(mapping, group, group_info)

        except Exception as e:
            return _handle_tool_error(
                operation="set_column_mapping",
                prefix="Error setting column mapping",
                error=e,
                default_code="SET_COLUMN_MAPPING_FAILED",
                default_message="Unable to apply column mapping.",
            )

    set_mapping_tool = StructuredTool.from_function(
        func=set_column_mapping,
        name="set_column_mapping",
        description="Set the column mapping for A/B test analysis. Specify which columns contain customer ID, group indicator, effect value, segments, and duration.",
    )

    def set_group_labels(treatment_label: str, control_label: str) -> str:
        """Set the labels used for treatment and control groups"""
        logger.info("Tool set_group_labels started")
        try:
            analyzer = agent._get_active_analyzer()
            analyzer.set_group_labels(treatment_label, control_label)
            logger.info("Tool set_group_labels completed")
            return f"Group labels set: Treatment='{treatment_label}', Control='{control_label}'"
        except Exception as e:
            return _handle_tool_error(
                operation="set_group_labels",
                prefix="Error setting group labels",
                error=e,
                default_code="SET_GROUP_LABELS_FAILED",
                default_message="Unable to set treatment/control labels.",
            )

    set_labels_tool = Tool(
        name="set_group_labels",
        func=lambda x: set_group_labels(*[s.strip() for s in x.split(",")]),
        description="Set the treatment and control group labels. Input format: 'treatment_label, control_label'",
    )

    def run_ab_test(segment: Optional[str] = None) -> str:
        """Run A/B test for a specific segment or overall"""
        logger.info("Tool run_ab_test started (segment=%s)", segment or "overall")
        try:
            analyzer = _active_analyzer()
            if segment and segment.lower() not in ["none", "overall", "all", ""]:
                result = analyzer.run_ab_test(segment_filter=segment)
            else:
                result = analyzer.run_ab_test()
            summary = analyzer.generate_summary([result])
            agent._last_results = [result]
            agent._last_summary = summary
            agent.persist_analysis_outputs([result], summary)
            logger.info("Tool run_ab_test completed (segment=%s)", result.segment)

            return render_run_ab_test_output(result)

        except Exception as e:
            return _handle_tool_error(
                operation="run_ab_test",
                prefix="Error running A/B test",
                error=e,
                default_code="RUN_AB_TEST_FAILED",
                default_message="Unable to run the A/B test.",
            )

    run_test_tool = Tool(
        name="run_ab_test",
        func=run_ab_test,
        description="Run A/B test for a specific segment or overall. Input: segment name (or 'overall' for all data)",
    )

    def run_full_analysis(_: str = "") -> str:
        """Run A/B tests for all segments and generate summary"""
        logger.info("Tool run_full_analysis started")
        try:
            analyzer = _active_analyzer()
            results = analyzer.run_segmented_analysis()
            summary = analyzer.generate_summary(results)
            agent.persist_analysis_outputs(results, summary)
            logger.info(
                "Tool run_full_analysis completed (segments=%s)",
                summary.get("total_segments_analyzed"),
            )

            agent._last_results = results
            agent._last_summary = summary

            return render_full_analysis_output(summary)

        except Exception as e:
            return _handle_tool_error(
                operation="run_full_analysis",
                prefix="Error running full analysis",
                error=e,
                default_code="RUN_FULL_ANALYSIS_FAILED",
                default_message="Unable to run full analysis.",
            )

    full_analysis_tool = Tool(
        name="run_full_analysis",
        func=run_full_analysis,
        description="Run A/B tests for ALL segments and generate a comprehensive summary with recommendations. Use this for complete analysis.",
    )

    def answer_data_question(question: str) -> str:
        """Answer a natural-language question about raw data or analysis results."""
        logger.info("Tool answer_data_question started")
        try:
            analyzer = _active_analyzer()
            if getattr(analyzer, "df", None) is None:
                return "No data loaded. Please load a CSV file first."

            answer = agent.data_question_service.answer_question(question)
            logger.info("Tool answer_data_question completed")
            return render_data_question_output(answer)
        except Exception as e:
            return _handle_tool_error(
                operation="answer_data_question",
                prefix="Error answering data question",
                error=e,
                default_code="DATA_QUESTION_FAILED",
                default_message="Unable to answer that question from the current data.",
            )

    answer_data_question_tool = Tool(
        name="answer_data_question",
        func=answer_data_question,
        description=(
            "Answer a natural-language question about the loaded raw data or computed analysis results. "
            "Use this for questions like counts by segment, total effect size for a segment, or highest/lowest metrics."
        ),
    )

    def query_data(query: str) -> str:
        """Query the data using pandas query syntax"""
        logger.info("Tool query_data started")
        try:
            analyzer = _active_analyzer()
            if getattr(analyzer, "df", None) is None:
                return "No data loaded. Please load a CSV file first."

            if not hasattr(analyzer, "query_data"):
                raise _unsupported("Querying data", analyzer)

            result = analyzer.query_data(query)
            logger.info("Tool query_data completed (rows=%s)", len(result))
            return render_query_data_output(result)
        except Exception as e:
            return _handle_tool_error(
                operation="query_data",
                prefix="Error querying data",
                error=e,
                default_code="QUERY_DATA_FAILED",
                default_message="Unable to execute the requested query.",
            )

    query_tool = Tool(
        name="query_data",
        func=query_data,
        description="Query the loaded data using pandas query syntax. Example: 'segment == \"Premium\"' or 'effect_value > 100'",
    )

    def get_data_summary(_: str = "") -> str:
        """Get summary statistics of the data"""
        logger.info("Tool get_data_summary started")
        try:
            analyzer = _active_analyzer()
            if not hasattr(analyzer, "get_data_summary"):
                raise _unsupported("Getting a data summary", analyzer)

            summary = analyzer.get_data_summary()
            logger.info("Tool get_data_summary completed")
            return render_data_summary_output(summary)

        except Exception as e:
            return _handle_tool_error(
                operation="get_data_summary",
                prefix="Error getting data summary",
                error=e,
                default_code="GET_DATA_SUMMARY_FAILED",
                default_message="Unable to summarize the data.",
            )

    summary_tool = Tool(
        name="get_data_summary",
        func=get_data_summary,
        description="Get summary statistics of the loaded data including column types, missing values, and descriptive statistics.",
    )

    def get_segment_distribution(_: str = "") -> str:
        """Get distribution of customers across segments and groups"""
        logger.info("Tool get_segment_distribution started")
        try:
            analyzer = _active_analyzer()
            if not hasattr(analyzer, "get_segment_distribution"):
                raise _unsupported("Getting the segment distribution", analyzer)

            dist = analyzer.get_segment_distribution()
            logger.info("Tool get_segment_distribution completed")
            return render_segment_distribution_output(dist)

        except Exception as e:
            return _handle_tool_error(
                operation="get_segment_distribution",
                prefix="Error getting distribution",
                error=e,
                default_code="GET_DISTRIBUTION_FAILED",
                default_message="Unable to compute segment distribution.",
            )

    dist_tool = Tool(
        name="get_segment_distribution",
        func=get_segment_distribution,
        description="Get the distribution of customers across segments and treatment/control groups.",
    )

    def get_column_values(column_name: str) -> str:
        """Get unique values in a specific column"""
        logger.info("Tool get_column_values started (column=%s)", column_name)
        try:
            analyzer = _active_analyzer()
            df = _require_pandas_dataframe(analyzer, "Listing column values")

            if column_name not in df.columns:
                return f"Column '{column_name}' not found. Available columns: {list(df.columns)}"

            values = df[column_name].unique()
            value_counts = df[column_name].value_counts()
            logger.info("Tool get_column_values completed (column=%s, unique=%s)", column_name, len(values))

            return render_column_values_output(column_name, values, value_counts)

        except Exception as e:
            return _handle_tool_error(
                operation="get_column_values",
                prefix="Error",
                error=e,
                default_code="GET_COLUMN_VALUES_FAILED",
                default_message="Unable to list column values.",
            )

    column_values_tool = Tool(
        name="get_column_values",
        func=get_column_values,
        description="Get unique values and their counts for a specific column. Input: column name",
    )

    def calculate_stats(column_name: str) -> str:
        """Calculate detailed statistics for a numeric column"""
        logger.info("Tool calculate_statistics started (column=%s)", column_name)
        try:
            analyzer = _active_analyzer()
            df = _require_pandas_dataframe(analyzer, "Calculating column statistics")

            if column_name not in df.columns:
                return f"Column '{column_name}' not found."

            col = df[column_name]

            if col.dtype not in ["float64", "int64", "float32", "int32"]:
                return f"Column '{column_name}' is not numeric (type: {col.dtype})"

            output = render_calculate_stats_output(column_name, col)
            logger.info("Tool calculate_statistics completed (column=%s)", column_name)
            return output

        except Exception as e:
            return _handle_tool_error(
                operation="calculate_statistics",
                prefix="Error",
                error=e,
                default_code="CALCULATE_STATISTICS_FAILED",
                default_message="Unable to calculate column statistics.",
            )

    stats_tool = Tool(
        name="calculate_statistics",
        func=calculate_stats,
        description="Calculate detailed statistics for a numeric column including mean, median, std dev, percentiles.",
    )

    def configure_and_analyze(
        group_column: str,
        effect_column: str,
        treatment_label: str,
        control_label: str,
        segment_column: Optional[str] = None,
        customer_id_column: Optional[str] = None,
    ) -> str:
        """Configure column mappings, set group labels, and run full analysis in one step"""
        logger.info("Tool configure_and_analyze started")
        try:
            analyzer = agent._get_active_analyzer()

            mapping = {"group": group_column, "effect_value": effect_column}
            if segment_column:
                mapping["segment"] = segment_column
            if customer_id_column:
                mapping["customer_id"] = customer_id_column

            analyzer.set_column_mapping(mapping)
            analyzer.set_group_labels(treatment_label, control_label)

            results = analyzer.run_segmented_analysis()
            summary = analyzer.generate_summary(results)
            agent.persist_analysis_outputs(results, summary)
            logger.info(
                "Tool configure_and_analyze completed (segments=%s)",
                summary.get("total_segments_analyzed"),
            )

            agent._last_results = results
            agent._last_summary = summary

            return render_configure_and_analyze_report(
                group_column=group_column,
                effect_column=effect_column,
                treatment_label=treatment_label,
                control_label=control_label,
                segment_column=segment_column,
                summary=summary,
            )

        except Exception as e:
            return _handle_tool_error(
                operation="configure_and_analyze",
                prefix="Error in configure and analyze",
                error=e,
                default_code="CONFIGURE_AND_ANALYZE_FAILED",
                default_message="Unable to configure and run analysis.",
            )

    configure_analyze_tool = StructuredTool.from_function(
        func=configure_and_analyze,
        name="configure_and_analyze",
        description="""Configure column mappings, set treatment/control labels, and run full A/B test analysis in ONE step.
Use this tool to quickly set up and analyze data without multiple separate steps.
Required: group_column, effect_column, treatment_label, control_label
Optional: segment_column, customer_id_column""",
    )

    def auto_configure_and_analyze(_: str = "") -> str:
        """Automatically configure everything using best guesses and run full analysis"""
        logger.info("Tool auto_configure_and_analyze started")
        try:
            analyzer = agent._get_active_analyzer()

            if not hasattr(analyzer, "df") or analyzer.df is None:
                return "No data loaded. Please load a CSV file first."

            config = analyzer.auto_configure()

            if not config["success"]:
                return f"Auto-configuration failed: {config.get('error', 'Unknown error')}"

            results = analyzer.run_segmented_analysis()
            summary = analyzer.generate_summary(results)
            agent.persist_analysis_outputs(results, summary)
            logger.info(
                "Tool auto_configure_and_analyze completed (segments=%s)",
                summary.get("total_segments_analyzed"),
            )

            agent._last_results = results
            agent._last_summary = summary

            return render_auto_configure_and_analyze_report(config, summary)

        except Exception as e:
            return _handle_tool_error(
                operation="auto_configure_and_analyze",
                prefix="Error in auto-configure and analyze",
                error=e,
                default_code="AUTO_CONFIGURE_AND_ANALYZE_FAILED",
                default_message="Unable to auto-configure and analyze the data.",
            )

    auto_analyze_tool = Tool(
        name="auto_configure_and_analyze",
        func=auto_configure_and_analyze,
        description="""Automatically detect column mappings and treatment/control labels using best guesses, then run full A/B test analysis.
Use this when the user says 'best guess', 'auto', 'automatic', or wants you to figure out the configuration yourself.
This is the fastest way to analyze data without manual configuration.""",
    )

    def generate_charts(chart_type: str = "all") -> str:
        """Generate visualization charts for the A/B test results"""
        logger.info("Tool generate_charts started (chart_type=%s)", chart_type)
        try:
            analyzer = _active_analyzer()

            if getattr(analyzer, "df", None) is None:
                return "No data loaded. Please load a CSV file first before generating charts."

            if "group" not in analyzer.column_mapping:
                return "Column mapping not set. Please configure the group and effect_value columns first."

            if analyzer.treatment_label is None:
                return "Group labels not set. Please specify which values represent treatment and control."

            if agent._last_results is None:
                results = analyzer.run_segmented_analysis()
                if not results:
                    return "No results from analysis. Please check your data configuration."
                summary = analyzer.generate_summary(results)
                agent._last_results = results
                agent._last_summary = summary
            else:
                results = agent._last_results
                summary = agent._last_summary

            normalized_summary = to_ab_test_summary(summary)
            agent._last_summary = normalized_summary

            if normalized_summary.error:
                return "Could not generate summary. Please run the full analysis first."

            chart_type = chart_type.lower().strip()

            agent._last_charts = {}

            if chart_type in ["all", "summary", "statistical_summary", "stats", "table"]:
                agent._last_charts["statistical_summary"] = agent.visualizer.plot_statistical_summary(results)

            if chart_type in ["all", "dashboard"]:
                agent._last_charts["dashboard"] = agent.visualizer.plot_summary_dashboard(
                    results, normalized_summary
                )

            if chart_type in ["all", "treatment_control", "comparison", "means"]:
                agent._last_charts["treatment_vs_control"] = agent.visualizer.plot_treatment_vs_control(results)

            if chart_type in ["all", "effect", "effect_size", "effects"]:
                agent._last_charts["effect_sizes"] = agent.visualizer.plot_effect_sizes(results)

            if chart_type in ["all", "pvalue", "p_value", "significance"]:
                agent._last_charts["p_values"] = agent.visualizer.plot_p_values(results)

            if chart_type in ["all", "power", "power_analysis"]:
                agent._last_charts["power_analysis"] = agent.visualizer.plot_power_analysis(results)

            if chart_type in ["all", "cohens_d", "cohen", "effect_magnitude"]:
                agent._last_charts["cohens_d"] = agent.visualizer.plot_cohens_d(results)

            if chart_type in ["all", "sample", "sample_size", "samples"]:
                agent._last_charts["sample_sizes"] = agent.visualizer.plot_sample_sizes(results)

            if chart_type in ["all", "waterfall", "contribution"]:
                agent._last_charts["effect_waterfall"] = agent.visualizer.plot_effect_waterfall(results)

            if chart_type in ["all", "bayesian", "bayesian_probability", "probability"]:
                agent._last_charts["bayesian_probability"] = agent.visualizer.plot_bayesian_probability(results)

            if chart_type in ["all", "bayesian", "bayesian_credible", "credible_interval", "credible"]:
                agent._last_charts["bayesian_credible_intervals"] = agent.visualizer.plot_bayesian_credible_intervals(results)

            if chart_type in ["all", "bayesian", "bayesian_loss", "expected_loss", "loss"]:
                agent._last_charts["bayesian_expected_loss"] = agent.visualizer.plot_bayesian_expected_loss(results)

            chart_names = list(agent._last_charts.keys())
            logger.info("Tool generate_charts completed (chart_count=%s)", len(chart_names))
            return render_generate_charts_output(chart_names)

        except Exception as e:
            return _handle_tool_error(
                operation="generate_charts",
                prefix="Error generating charts",
                error=e,
                default_code="GENERATE_CHARTS_FAILED",
                default_message=(
                    "Unable to generate charts. Please ensure data is loaded, column mapping is set, "
                    "and group labels are configured."
                ),
            )

    generate_charts_tool = Tool(
        name="generate_charts",
        func=generate_charts,
        description="""Generate visualization charts for A/B test results.
Input options:
- 'all': Generate all charts (frequentist + Bayesian)
- 'summary' or 'statistical_summary': Statistical summary with T-test & Proportion test p-values, effects, and total effects (RECOMMENDED)
- 'dashboard': Summary dashboard
- 'treatment_control' or 'comparison': Treatment vs Control means chart
- 'effect' or 'effect_size': Effect sizes with confidence intervals
- 'pvalue' or 'significance': P-values chart
- 'power' or 'power_analysis': Statistical power chart
- 'cohens_d' or 'cohen': Cohen's d effect size chart
- 'sample' or 'sample_size': Sample sizes chart
- 'waterfall' or 'contribution': Effect contribution waterfall chart
- 'bayesian': All Bayesian charts (probability, credible intervals, expected loss)
- 'bayesian_probability' or 'probability': P(Treatment > Control) chart
- 'bayesian_credible' or 'credible': 95% credible intervals forest plot
- 'bayesian_loss' or 'expected_loss': Expected loss chart
Use this tool when the user asks to see charts, visualizations, or graphs.""",
    )

    def show_distribution_chart(_: str = "") -> str:
        """Generate distribution chart showing treatment/control split"""
        logger.info("Tool show_distribution_chart started")
        try:
            analyzer = _active_analyzer()

            if getattr(analyzer, "df", None) is None:
                return "No data loaded. Please load a CSV file first."

            if "group" not in analyzer.column_mapping:
                return "Group column not set. Please configure column mappings first."

            group_col = analyzer.column_mapping["group"]
            segment_col = analyzer.column_mapping.get("segment")
            df = analyzer.df
            if hasattr(df, "groupBy") and hasattr(df, "select"):
                selected_columns = [group_col, *( [segment_col] if segment_col else [] )]
                df = df.select(*selected_columns).toPandas()

            agent._last_charts["distribution"] = agent.visualizer.plot_segment_distribution(
                df, group_col, segment_col
            )
            logger.info("Tool show_distribution_chart completed")

            return "Generated distribution chart showing treatment/control split. The chart is now displayed in the UI."

        except Exception as e:
            return _handle_tool_error(
                operation="show_distribution_chart",
                prefix="Error generating distribution chart",
                error=e,
                default_code="SHOW_DISTRIBUTION_CHART_FAILED",
                default_message="Unable to generate distribution chart.",
            )

    distribution_chart_tool = Tool(
        name="show_distribution_chart",
        func=show_distribution_chart,
        description="Generate a pie/sunburst chart showing the distribution of customers across treatment and control groups. Use when user asks about group distribution or wants to visualize the experiment split.",
    )

    return [
        load_csv_tool,
        load_auto_analyze_tool,
        set_mapping_tool,
        set_labels_tool,
        run_test_tool,
        full_analysis_tool,
        answer_data_question_tool,
        configure_analyze_tool,
        auto_analyze_tool,
        query_tool,
        summary_tool,
        dist_tool,
        column_values_tool,
        stats_tool,
        generate_charts_tool,
        distribution_chart_tool,
    ]

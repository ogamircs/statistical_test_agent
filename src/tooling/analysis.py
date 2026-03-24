"""Analysis, querying, and exploration tools."""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.tools import Tool

from ..agent_reporting import (
    render_calculate_stats_output,
    render_column_values_output,
    render_data_question_output,
    render_data_summary_output,
    render_full_analysis_output,
    render_query_data_output,
    render_run_ab_test_output,
    render_segment_distribution_output,
)
from .common import ToolContext

logger = logging.getLogger(__name__)


def create_analysis_tools(context: ToolContext) -> List[Tool]:
    agent = context.agent

    def run_ab_test(segment: Optional[str] = None) -> str:
        logger.info("Tool run_ab_test started (segment=%s)", segment or "overall")
        try:
            analyzer = context.active_analyzer()
            if segment and segment.lower() not in ["none", "overall", "all", ""]:
                result = analyzer.run_ab_test(segment_filter=segment)
            else:
                result = analyzer.run_ab_test()

            summary = analyzer.generate_summary([result])
            context.remember_analysis([result], summary)
            logger.info("Tool run_ab_test completed (segment=%s)", result.segment)
            return render_run_ab_test_output(result)
        except Exception as error:
            return context.handle_tool_error(
                operation="run_ab_test",
                prefix="Error running A/B test",
                error=error,
                default_code="RUN_AB_TEST_FAILED",
                default_message="Unable to run the A/B test.",
            )

    def run_full_analysis(_: str = "") -> str:
        logger.info("Tool run_full_analysis started")
        try:
            analyzer = context.active_analyzer()
            results = analyzer.run_segmented_analysis()
            summary = analyzer.generate_summary(results)
            context.remember_analysis(results, summary)
            logger.info(
                "Tool run_full_analysis completed (segments=%s)",
                getattr(summary, 'total_segments_analyzed', None),
            )
            return render_full_analysis_output(summary)
        except Exception as error:
            return context.handle_tool_error(
                operation="run_full_analysis",
                prefix="Error running full analysis",
                error=error,
                default_code="RUN_FULL_ANALYSIS_FAILED",
                default_message="Unable to run full analysis.",
            )

    def answer_data_question(question: str) -> str:
        logger.info("Tool answer_data_question started")
        try:
            analyzer = context.active_analyzer()
            if getattr(analyzer, "df", None) is None:
                return "No data loaded. Please load a CSV file first."

            answer = agent.data_question_service.answer_question(question)
            logger.info("Tool answer_data_question completed")
            return render_data_question_output(answer)
        except Exception as error:
            return context.handle_tool_error(
                operation="answer_data_question",
                prefix="Error answering data question",
                error=error,
                default_code="DATA_QUESTION_FAILED",
                default_message="Unable to answer that question from the current data.",
            )

    def query_data(query: str) -> str:
        logger.info("Tool query_data started")
        try:
            analyzer = context.active_analyzer()
            if getattr(analyzer, "df", None) is None:
                return "No data loaded. Please load a CSV file first."
            if not hasattr(analyzer, "query_data"):
                raise context.unsupported("Querying data", analyzer)

            result = analyzer.query_data(query)
            logger.info("Tool query_data completed (rows=%s)", len(result))
            return render_query_data_output(result)
        except Exception as error:
            return context.handle_tool_error(
                operation="query_data",
                prefix="Error querying data",
                error=error,
                default_code="QUERY_DATA_FAILED",
                default_message="Unable to execute the requested query.",
            )

    def get_data_summary(_: str = "") -> str:
        logger.info("Tool get_data_summary started")
        try:
            analyzer = context.active_analyzer()
            if not hasattr(analyzer, "get_data_summary"):
                raise context.unsupported("Getting a data summary", analyzer)

            summary = analyzer.get_data_summary()
            logger.info("Tool get_data_summary completed")
            return render_data_summary_output(summary)
        except Exception as error:
            return context.handle_tool_error(
                operation="get_data_summary",
                prefix="Error getting data summary",
                error=error,
                default_code="GET_DATA_SUMMARY_FAILED",
                default_message="Unable to summarize the data.",
            )

    def get_segment_distribution(_: str = "") -> str:
        logger.info("Tool get_segment_distribution started")
        try:
            analyzer = context.active_analyzer()
            if not hasattr(analyzer, "get_segment_distribution"):
                raise context.unsupported("Getting the segment distribution", analyzer)

            dist = analyzer.get_segment_distribution()
            logger.info("Tool get_segment_distribution completed")
            return render_segment_distribution_output(dist)
        except Exception as error:
            return context.handle_tool_error(
                operation="get_segment_distribution",
                prefix="Error getting distribution",
                error=error,
                default_code="GET_DISTRIBUTION_FAILED",
                default_message="Unable to compute segment distribution.",
            )

    def get_column_values(column_name: str) -> str:
        logger.info("Tool get_column_values started (column=%s)", column_name)
        try:
            analyzer = context.active_analyzer()
            df = context.require_pandas_dataframe(analyzer, "Listing column values")
            if column_name not in df.columns:
                return f"Column '{column_name}' not found. Available columns: {list(df.columns)}"

            values = df[column_name].unique()
            value_counts = df[column_name].value_counts()
            logger.info("Tool get_column_values completed (column=%s, unique=%s)", column_name, len(values))
            return render_column_values_output(column_name, values, value_counts)
        except Exception as error:
            return context.handle_tool_error(
                operation="get_column_values",
                prefix="Error",
                error=error,
                default_code="GET_COLUMN_VALUES_FAILED",
                default_message="Unable to list column values.",
            )

    def calculate_stats(column_name: str) -> str:
        logger.info("Tool calculate_statistics started (column=%s)", column_name)
        try:
            analyzer = context.active_analyzer()
            df = context.require_pandas_dataframe(analyzer, "Calculating column statistics")
            if column_name not in df.columns:
                return f"Column '{column_name}' not found."

            col = df[column_name]
            if col.dtype not in ["float64", "int64", "float32", "int32"]:
                return f"Column '{column_name}' is not numeric (type: {col.dtype})"

            output = render_calculate_stats_output(column_name, col)
            logger.info("Tool calculate_statistics completed (column=%s)", column_name)
            return output
        except Exception as error:
            return context.handle_tool_error(
                operation="calculate_statistics",
                prefix="Error",
                error=error,
                default_code="CALCULATE_STATISTICS_FAILED",
                default_message="Unable to calculate column statistics.",
            )

    return [
        Tool(
            name="run_ab_test",
            func=run_ab_test,
            description="Run A/B test for a specific segment or overall. Input: segment name (or 'overall' for all data)",
        ),
        Tool(
            name="run_full_analysis",
            func=run_full_analysis,
            description="Run A/B tests for ALL segments and generate a comprehensive summary with recommendations. Use this for complete analysis.",
        ),
        Tool(
            name="answer_data_question",
            func=answer_data_question,
            description=(
                "Answer a natural-language question about the loaded raw data or computed analysis results. "
                "Use this for questions like counts by segment, total effect size for a segment, or highest/lowest metrics."
            ),
        ),
        Tool(
            name="query_data",
            func=query_data,
            description="Query the loaded data using pandas query syntax. Example: 'segment == \"Premium\"' or 'effect_value > 100'",
        ),
        Tool(
            name="get_data_summary",
            func=get_data_summary,
            description="Get summary statistics of the loaded data including column types, missing values, and descriptive statistics.",
        ),
        Tool(
            name="get_segment_distribution",
            func=get_segment_distribution,
            description="Get the distribution of customers across segments and treatment/control groups.",
        ),
        Tool(
            name="get_column_values",
            func=get_column_values,
            description="Get unique values and their counts for a specific column. Input: column name",
        ),
        Tool(
            name="calculate_statistics",
            func=calculate_stats,
            description="Calculate detailed statistics for a numeric column including mean, median, std dev, percentiles.",
        ),
    ]

"""Visualization-related tool handlers."""

from __future__ import annotations

import logging
from typing import List

from langchain_core.tools import Tool

from ..agent_reporting import render_generate_charts_output
from ..statistics.chart_catalog import build_chart_map, resolve_chart_keys
from ..statistics.models import to_ab_test_summary
from .common import ToolContext

logger = logging.getLogger(__name__)


def create_visualization_tools(context: ToolContext) -> List[Tool]:
    agent = context.agent

    def generate_charts(chart_type: str = "all") -> str:
        logger.info("Tool generate_charts started (chart_type=%s)", chart_type)
        try:
            analyzer = context.active_analyzer()
            if getattr(analyzer, "df", None) is None:
                return "No data loaded. Please load a CSV file first before generating charts."
            if "group" not in analyzer.column_mapping:
                return "Column mapping not set. Please configure the group and effect_value columns first."
            if analyzer.treatment_label is None:
                return "Group labels not set. Please specify which values represent treatment and control."

            results = agent._last_results
            summary = agent._last_summary
            if results is None:
                results = analyzer.run_segmented_analysis()
                if not results:
                    return "No results from analysis. Please check your data configuration."
                summary = analyzer.generate_summary(results)
                agent._last_results = results
                agent._last_summary = summary

            normalized_summary = to_ab_test_summary(summary)
            agent._last_summary = normalized_summary
            if normalized_summary.error:
                return "Could not generate summary. Please run the full analysis first."

            selected_keys = resolve_chart_keys(chart_type)
            if not selected_keys:
                return "Unknown chart type. Please choose 'all', 'summary', 'dashboard', 'effect', 'pvalue', 'power', 'sample', 'waterfall', or 'bayesian'."

            agent._last_charts = build_chart_map(
                agent.visualizer,
                results,
                normalized_summary,
                selected_keys,
            )
            chart_names = list(agent._last_charts.keys())
            logger.info("Tool generate_charts completed (chart_count=%s)", len(chart_names))
            return render_generate_charts_output(chart_names)
        except Exception as error:
            return context.handle_tool_error(
                operation="generate_charts",
                prefix="Error generating charts",
                error=error,
                default_code="GENERATE_CHARTS_FAILED",
                default_message=(
                    "Unable to generate charts. Please ensure data is loaded, column mapping is set, "
                    "and group labels are configured."
                ),
            )

    def show_distribution_chart(_: str = "") -> str:
        logger.info("Tool show_distribution_chart started")
        try:
            analyzer = context.active_analyzer()
            if getattr(analyzer, "df", None) is None:
                return "No data loaded. Please load a CSV file first."
            if "group" not in analyzer.column_mapping:
                return "Group column not set. Please configure column mappings first."

            group_col = analyzer.column_mapping["group"]
            segment_col = analyzer.column_mapping.get("segment")
            df = analyzer.df
            if hasattr(df, "groupBy") and hasattr(df, "select"):
                selected_columns = [group_col, *([segment_col] if segment_col else [])]
                df = df.select(*selected_columns).toPandas()

            agent._last_charts["distribution"] = agent.visualizer.plot_segment_distribution(
                df, group_col, segment_col
            )
            logger.info("Tool show_distribution_chart completed")
            return "Generated distribution chart showing treatment/control split. The chart is now displayed in the UI."
        except Exception as error:
            return context.handle_tool_error(
                operation="show_distribution_chart",
                prefix="Error generating distribution chart",
                error=error,
                default_code="SHOW_DISTRIBUTION_CHART_FAILED",
                default_message="Unable to generate distribution chart.",
            )

    return [
        Tool(
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
        ),
        Tool(
            name="show_distribution_chart",
            func=show_distribution_chart,
            description="Generate a pie/sunburst chart showing the distribution of customers across treatment and control groups. Use when user asks about group distribution or wants to visualize the experiment split.",
        ),
    ]

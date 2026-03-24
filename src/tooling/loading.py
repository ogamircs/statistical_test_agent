"""Data loading and configuration tools."""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.tools import StructuredTool, Tool

from ..agent_reporting import (
    render_auto_configure_and_analyze_report,
    render_configure_and_analyze_report,
    render_load_and_auto_analyze_report,
    render_load_csv_success,
    render_set_column_mapping_success,
)
from .common import ToolContext

logger = logging.getLogger(__name__)


def create_loading_tools(context: ToolContext) -> List[Tool]:
    agent = context.agent

    def load_csv(filepath: str) -> str:
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
        except Exception as error:
            return context.handle_tool_error(
                operation="load_csv",
                prefix="Error loading file",
                error=error,
                default_code="LOAD_CSV_FAILED",
                default_message="Unable to load CSV data.",
            )

    def load_and_auto_analyze(filepath: str) -> str:
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
            context.remember_analysis(results, summary)
            logger.info(
                "Tool load_and_auto_analyze completed (backend=%s, segments=%s)",
                backend,
                getattr(summary, 'total_segments_analyzed', None),
            )
            return render_load_and_auto_analyze_report(
                filepath=filepath,
                file_size_mb=file_size_mb,
                shape=shape,
                backend=backend,
                fallback_note=fallback_note,
                config=config,
                summary=summary,
            )
        except Exception as error:
            return context.handle_tool_error(
                operation="load_and_auto_analyze",
                prefix="Error in load and auto-analyze",
                error=error,
                default_code="LOAD_AND_ANALYZE_FAILED",
                default_message="Unable to complete automatic analysis.",
            )

    def set_column_mapping(
        customer_id: Optional[str] = None,
        group: str = "",
        effect_value: str = "",
        segment: Optional[str] = None,
        duration: Optional[str] = None,
    ) -> str:
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
            analyzer = context.active_analyzer()
            analyzer.set_column_mapping(mapping)
            logger.info("Tool set_column_mapping completed (fields=%s)", sorted(mapping.keys()))
            group_info = analyzer.get_group_values() if hasattr(analyzer, "get_group_values") else {"unique_values": []}
            return render_set_column_mapping_success(mapping, group, group_info)
        except Exception as error:
            return context.handle_tool_error(
                operation="set_column_mapping",
                prefix="Error setting column mapping",
                error=error,
                default_code="SET_COLUMN_MAPPING_FAILED",
                default_message="Unable to apply column mapping.",
            )

    def set_group_labels(treatment_label: str, control_label: str) -> str:
        logger.info("Tool set_group_labels started")
        try:
            analyzer = context.active_analyzer()
            analyzer.set_group_labels(treatment_label, control_label)
            logger.info("Tool set_group_labels completed")
            return f"Group labels set: Treatment='{treatment_label}', Control='{control_label}'"
        except Exception as error:
            return context.handle_tool_error(
                operation="set_group_labels",
                prefix="Error setting group labels",
                error=error,
                default_code="SET_GROUP_LABELS_FAILED",
                default_message="Unable to set treatment/control labels.",
            )

    def configure_and_analyze(
        group_column: str,
        effect_column: str,
        treatment_label: str,
        control_label: str,
        segment_column: Optional[str] = None,
        customer_id_column: Optional[str] = None,
    ) -> str:
        logger.info("Tool configure_and_analyze started")
        try:
            analyzer = context.active_analyzer()
            mapping = {"group": group_column, "effect_value": effect_column}
            if segment_column:
                mapping["segment"] = segment_column
            if customer_id_column:
                mapping["customer_id"] = customer_id_column
            analyzer.set_column_mapping(mapping)
            analyzer.set_group_labels(treatment_label, control_label)

            results = analyzer.run_segmented_analysis()
            summary = analyzer.generate_summary(results)
            context.remember_analysis(results, summary)
            logger.info(
                "Tool configure_and_analyze completed (segments=%s)",
                getattr(summary, 'total_segments_analyzed', None),
            )
            return render_configure_and_analyze_report(
                group_column=group_column,
                effect_column=effect_column,
                treatment_label=treatment_label,
                control_label=control_label,
                segment_column=segment_column,
                summary=summary,
            )
        except Exception as error:
            return context.handle_tool_error(
                operation="configure_and_analyze",
                prefix="Error in configure and analyze",
                error=error,
                default_code="CONFIGURE_AND_ANALYZE_FAILED",
                default_message="Unable to configure and run analysis.",
            )

    def auto_configure_and_analyze(_: str = "") -> str:
        logger.info("Tool auto_configure_and_analyze started")
        try:
            analyzer = context.active_analyzer()
            if not hasattr(analyzer, "df") or analyzer.df is None:
                return "No data loaded. Please load a CSV file first."

            config = analyzer.auto_configure()
            if not config["success"]:
                return f"Auto-configuration failed: {config.get('error', 'Unknown error')}"

            results = analyzer.run_segmented_analysis()
            summary = analyzer.generate_summary(results)
            context.remember_analysis(results, summary)
            logger.info(
                "Tool auto_configure_and_analyze completed (segments=%s)",
                getattr(summary, 'total_segments_analyzed', None),
            )
            return render_auto_configure_and_analyze_report(config, summary)
        except Exception as error:
            return context.handle_tool_error(
                operation="auto_configure_and_analyze",
                prefix="Error in auto-configure and analyze",
                error=error,
                default_code="AUTO_CONFIGURE_AND_ANALYZE_FAILED",
                default_message="Unable to auto-configure and analyze the data.",
            )

    return [
        Tool(
            name="load_csv",
            func=load_csv,
            description="Load a CSV file for A/B test analysis. Input should be the file path. Use this when you need to inspect the data before analysis. For best-guess mode, use load_and_auto_analyze instead.",
        ),
        Tool(
            name="load_and_auto_analyze",
            func=load_and_auto_analyze,
            description="""Load a CSV file AND automatically run full A/B test analysis using best guesses - ALL IN ONE STEP.
Use this when the user says 'best guess', 'auto', 'automatic', 'figure it out', or wants quick analysis without manual configuration.
Input: file path. This is the FASTEST way to get results.""",
        ),
        StructuredTool.from_function(
            func=set_column_mapping,
            name="set_column_mapping",
            description="Set the column mapping for A/B test analysis. Specify which columns contain customer ID, group indicator, effect value, segments, and duration.",
        ),
        Tool(
            name="set_group_labels",
            func=lambda x: set_group_labels(*[s.strip() for s in x.split(",")]),
            description="Set the treatment and control group labels. Input format: 'treatment_label, control_label'",
        ),
        StructuredTool.from_function(
            func=configure_and_analyze,
            name="configure_and_analyze",
            description="""Configure column mappings, set treatment/control labels, and run full A/B test analysis in ONE step.
Use this tool to quickly set up and analyze data without multiple separate steps.
Required: group_column, effect_column, treatment_label, control_label
Optional: segment_column, customer_id_column""",
        ),
        Tool(
            name="auto_configure_and_analyze",
            func=auto_configure_and_analyze,
            description="""Automatically detect column mappings and treatment/control labels using best guesses, then run full A/B test analysis.
Use this when the user says 'best guess', 'auto', 'automatic', or wants you to figure out the configuration yourself.
This is the fastest way to analyze data without manual configuration.""",
        ),
    ]

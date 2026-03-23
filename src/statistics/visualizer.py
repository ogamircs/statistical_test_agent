"""
A/B Test Visualization Module

Creates clean, interactive Plotly charts for A/B test results:
- Treatment vs Control comparison
- Effect sizes across segments
- Statistical significance
- Power analysis
- Confidence intervals
"""

from typing import Dict, List, Optional, Any

import pandas as pd
import plotly.graph_objects as go

from .chart_catalog import all_chart_keys, build_chart_map
from .charts_core import CoreChartsMixin
from .charts_dashboard import DashboardChartsMixin
from .charts_extended import ExtendedChartsMixin
from .models import ABTestResult


class ABTestVisualizer(CoreChartsMixin, ExtendedChartsMixin, DashboardChartsMixin):
    """
    Visualization class for A/B Test Results

    Creates clean, professional Plotly charts with consistent styling.
    """

    def __init__(self):
        # Clean, modern color palette
        self.colors = {
            'treatment': '#10B981',      # Emerald green
            'control': '#3B82F6',         # Blue
            'significant_pos': '#059669', # Dark green
            'significant_neg': '#DC2626', # Red
            'not_significant': '#9CA3AF', # Gray
            'adequate': '#10B981',        # Green
            'inadequate': '#F59E0B',      # Amber
            'background': '#FFFFFF',
            'grid': '#E5E7EB',
            'text': '#374151',
            't_test': '#3B82F6',          # Blue for t-test
            'proportion': '#8B5CF6',      # Purple for proportion test
            'combined': '#10B981'         # Green for combined
        }

        # Common layout settings
        self.layout_defaults = dict(
            template='plotly_white',
            font=dict(family='Inter, system-ui, sans-serif', size=12, color=self.colors['text']),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            margin=dict(l=60, r=40, t=80, b=60),
            hoverlabel=dict(bgcolor='white', font_size=12),
        )

    def _apply_layout(self, fig: go.Figure, title: str, height: int = 500) -> go.Figure:
        """Apply consistent layout settings to a figure"""
        fig.update_layout(
            **self.layout_defaults,
            title=dict(
                text=title,
                font=dict(size=16, color=self.colors['text']),
                x=0.5,
                xanchor='center'
            ),
            height=height
        )
        fig.update_xaxes(
            showgrid=False,
            linecolor=self.colors['grid'],
            tickfont=dict(size=11)
        )
        fig.update_yaxes(
            gridcolor=self.colors['grid'],
            linecolor=self.colors['grid'],
            tickfont=dict(size=11)
        )
        return fig

    def create_all_charts(self, results: List[ABTestResult], summary: Dict[str, Any],
                         df: Optional[pd.DataFrame] = None,
                         group_col: Optional[str] = None,
                         segment_col: Optional[str] = None) -> Dict[str, go.Figure]:
        """
        Generate all visualization charts

        Returns a dictionary of chart name -> Plotly figure
        """
        selected_keys = all_chart_keys() if df is not None and group_col is not None else None

        charts = build_chart_map(
            self, results, summary,
            selected_keys=selected_keys,
            df=df, group_col=group_col, segment_col=segment_col,
        )

        return charts

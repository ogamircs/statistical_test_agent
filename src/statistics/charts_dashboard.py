"""Dashboard chart mixins for A/B test visualizations.

Contains multi-panel summary and dashboard chart methods.

These methods rely on ``self.colors`` and ``self._apply_layout`` provided
by the main ``ABTestVisualizer`` class.
"""

from __future__ import annotations

from typing import Any, Dict, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .chart_builders import (
    add_grouped_pair_bars,
    apply_multi_panel_theme,
    make_bar_trace,
    style_subplot_axes,
    style_subplot_titles,
)
from .models import ABTestResult, to_ab_test_summary


class DashboardChartsMixin:
    """Mixin providing multi-panel dashboard chart methods."""

    def plot_statistical_summary(self, results: List[ABTestResult]) -> go.Figure:
        """
        Create a compact, readable statistical summary for chat UI.

        This layout intentionally uses only 2 panels to avoid crowded small multiples
        in narrow chat containers.
        """
        segments = [r.segment for r in results]
        t_pvals = [r.p_value for r in results]
        prop_pvals = [r.proportion_p_value for r in results]

        t_effects = [r.effect_size for r in results]
        prop_effects = [r.proportion_effect_per_customer for r in results]
        total_effects = [r.total_effect_per_customer for r in results]

        # Fixed metric colors are easier to scan than significance-colored bars
        # in narrow chat containers.
        t_pval_color = 'rgba(59, 130, 246, 0.78)'
        prop_pval_color = 'rgba(139, 92, 246, 0.78)'
        t_effect_color = 'rgba(16, 185, 129, 0.95)'
        prop_effect_color = 'rgba(139, 92, 246, 0.92)'

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                '<b>P-Values (T-test vs Proportion)</b>',
                '<b>Effects per Customer (T-test, Proportion, Total)</b>'
            ),
            vertical_spacing=0.22
        )

        # Panel 1: p-values for both tests
        fig.add_trace(
            make_bar_trace(
                name='T-test p-value',
                x=segments,
                y=t_pvals,
                marker_color=t_pval_color,
                hovertemplate='Segment: %{x}<br>T-test p: %{y:.4f}<extra></extra>',
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            make_bar_trace(
                name='Proportion p-value',
                x=segments,
                y=prop_pvals,
                marker_color=prop_pval_color,
                hovertemplate='Segment: %{x}<br>Proportion p: %{y:.4f}<extra></extra>',
            ),
            row=1,
            col=1
        )
        fig.add_hline(
            y=0.05,
            line_dash='dash',
            line_color=self.colors['significant_neg'],
            line_width=2,
            row=1,
            col=1
        )

        # Panel 2: effect comparison
        fig.add_trace(
            make_bar_trace(
                name='T-test effect',
                x=segments,
                y=t_effects,
                marker_color=t_effect_color,
                hovertemplate='Segment: %{x}<br>T-test effect: %{y:.4f}<extra></extra>',
            ),
            row=2,
            col=1
        )
        fig.add_trace(
            make_bar_trace(
                name='Proportion effect',
                x=segments,
                y=prop_effects,
                marker_color=prop_effect_color,
                hovertemplate='Segment: %{x}<br>Proportion effect: %{y:.4f}<extra></extra>',
            ),
            row=2,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                name='Total effect',
                x=segments,
                y=total_effects,
                mode='lines+markers',
                marker=dict(size=8, color=self.colors['combined']),
                line=dict(width=2, color=self.colors['combined']),
                hovertemplate='Segment: %{x}<br>Total effect: %{y:.4f}<extra></extra>',
            ),
            row=2,
            col=1
        )
        fig.add_hline(y=0, line_dash='solid', line_color=self.colors['grid'], row=2, col=1)

        # Layout tuned for readability inside chat UI
        apply_multi_panel_theme(
            fig,
            colors=self.colors,
            title_text='<b>Statistical Results Summary</b>',
            height=900,
            legend_y=1.02,
            barmode='group',
            bargap=0.24,
            bargroupgap=0.1,
            margin=dict(l=80, r=50, t=100, b=90),
        )
        fig.update_layout(hovermode='x', showlegend=False)

        # Axis titles
        fig.update_yaxes(title_text='P-value', row=1, col=1, title_font=dict(size=13))
        fig.update_yaxes(title_text='Effect', row=2, col=1, title_font=dict(size=13))

        # Keep p-value charts in a readable range with alpha threshold visible
        max_p = max(max(t_pvals), max(prop_pvals))
        fig.update_yaxes(range=[0, max(max_p * 1.15, 0.1)], row=1, col=1)

        # Style subplots for compact chat rendering
        style_subplot_axes(
            fig,
            rows=2,
            cols=1,
            grid_color=self.colors['grid'],
            tickfont_size=11,
            x_tickangle=-20,
        )
        style_subplot_titles(fig, text_color=self.colors['text'], size=14)

        return fig

    def plot_summary_dashboard(self, results: List[ABTestResult], summary: Dict[str, Any]) -> go.Figure:
        """Create a comprehensive dashboard with multiple visualizations"""
        normalized_summary = to_ab_test_summary(summary)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>Treatment vs Control Means</b>',
                '<b>Combined Effects (T-test + Proportion)</b>',
                '<b>Conversion Rates</b>',
                '<b>P-Values (T-test)</b>'
            ),
            vertical_spacing=0.18,
            horizontal_spacing=0.12
        )

        segments = [r.segment for r in results]

        # Plot 1: Treatment vs Control Means
        add_grouped_pair_bars(
            fig,
            segments=segments,
            left_name='Treatment',
            left_values=[r.treatment_mean for r in results],
            left_color=self.colors['treatment'],
            right_name='Control',
            right_values=[r.control_mean for r in results],
            right_color=self.colors['control'],
            row=1,
            col=1,
            showlegend=True,
        )

        # Plot 2: Combined Effects (stacked T-test + Proportion)
        t_test_effects = [r.effect_size * r.treatment_size if r.is_significant else 0 for r in results]
        prop_effects = [r.proportion_effect for r in results]

        fig.add_trace(
            make_bar_trace(
                name='T-test Effect',
                x=segments,
                y=t_test_effects,
                marker_color=self.colors['t_test'],
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            make_bar_trace(
                name='Proportion Effect',
                x=segments,
                y=prop_effects,
                marker_color=self.colors['proportion'],
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line_dash="solid", line_color=self.colors['grid'], row=1, col=2)

        # Plot 3: Conversion Rates (Proportion comparison)
        add_grouped_pair_bars(
            fig,
            segments=segments,
            left_name='Treatment Conv',
            left_values=[r.treatment_proportion * 100 for r in results],
            left_color=self.colors['treatment'],
            right_name='Control Conv',
            right_values=[r.control_proportion * 100 for r in results],
            right_color=self.colors['control'],
            row=2,
            col=1,
            showlegend=False,
        )

        # Plot 4: P-Values (T-test)
        p_values = [r.p_value for r in results]
        p_colors = [
            self.colors['significant_pos'] if p < 0.05 else self.colors['not_significant']
            for p in p_values
        ]
        fig.add_trace(
            make_bar_trace(
                x=segments,
                y=p_values,
                marker_color=p_colors,
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.add_hline(y=0.05, line_dash="dash", line_color=self.colors['significant_neg'], line_width=1.5, row=2, col=2)

        # Apply layout
        title_text = (
            "<b>A/B Test Analysis Dashboard</b><br>"
            f"<span style='font-size:11px;color:#6B7280'>"
            f"T-test sig: {normalized_summary.t_test_significant_segments}/{normalized_summary.total_segments_analyzed}"
            " · "
            f"Prop sig: {normalized_summary.prop_test_significant_segments}/{normalized_summary.total_segments_analyzed}"
            " · "
            f"Combined effect: {normalized_summary.combined_total_effect:,.0f}</span>"
        )
        apply_multi_panel_theme(
            fig,
            colors=self.colors,
            title_text=title_text,
            height=880,
            legend_y=-0.15,
            barmode='group',
            bargap=0.18,
            bargroupgap=0.12,
            margin=dict(l=80, r=50, t=145, b=125),
        )
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center',
                font=dict(size=18, color=self.colors['text']),
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.15,
                xanchor='center',
                x=0.5,
                font=dict(size=11),
                bgcolor='rgba(255,255,255,0.92)',
            ),
        )

        # Update axes
        fig.update_yaxes(title_text='Mean', row=1, col=1, title_font=dict(size=11))
        fig.update_yaxes(title_text='Total Effect', row=1, col=2, title_font=dict(size=11))
        fig.update_yaxes(title_text='Conv Rate %', row=2, col=1, title_font=dict(size=11))
        fig.update_yaxes(title_text='P-Value', row=2, col=2, title_font=dict(size=11))

        # Make effect chart stacked
        fig.update_layout(barmode='group')
        fig.data[2].offsetgroup = 0  # T-test effect
        fig.data[3].offsetgroup = 0  # Proportion effect (stack on top)

        style_subplot_axes(fig, rows=2, cols=2, grid_color=self.colors['grid'], tickfont_size=10)
        style_subplot_titles(fig, text_color=self.colors['text'], size=12)

        return fig

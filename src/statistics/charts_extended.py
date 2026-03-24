"""Extended chart mixins for A/B test visualizations.

Contains Bayesian charts, effect waterfall, and segment distribution.

These methods rely on ``self.colors`` and ``self._apply_layout`` provided
by the main ``ABTestVisualizer`` class.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .chart_builders import (
    add_significance_legend,
    interval_error_arrays,
    make_bar_trace,
    significance_colors,
)
from .models import ABTestResult


class ExtendedChartsMixin:
    """Mixin providing Bayesian, waterfall, and distribution chart methods."""

    def plot_bayesian_probability(self, results: List[ABTestResult]) -> go.Figure:
        """Create bar chart showing Bayesian probability that treatment is better"""
        segments = [r.segment for r in results]
        probs = [r.bayesian_prob_treatment_better * 100 for r in results]

        colors = []
        for r in results:
            if r.bayesian_prob_treatment_better > 0.95:
                colors.append(self.colors['significant_pos'])
            elif r.bayesian_prob_treatment_better < 0.05:
                colors.append(self.colors['significant_neg'])
            else:
                colors.append(self.colors['not_significant'])

        fig = go.Figure()

        fig.add_trace(
            make_bar_trace(
                x=segments,
                y=probs,
                marker_color=colors,
                text=[f'{p:.1f}%' for p in probs],
            )
        )

        # Add threshold lines
        fig.add_hline(
            y=95,
            line_dash="dash",
            line_color=self.colors['significant_pos'],
            line_width=2,
            annotation_text="95% (Significant +)",
            annotation_position="right",
            annotation_font=dict(size=10, color=self.colors['significant_pos'])
        )
        fig.add_hline(
            y=50,
            line_dash="dot",
            line_color=self.colors['grid'],
            line_width=1,
            annotation_text="50% (No difference)",
            annotation_position="right",
            annotation_font=dict(size=9, color=self.colors['grid'])
        )
        fig.add_hline(
            y=5,
            line_dash="dash",
            line_color=self.colors['significant_neg'],
            line_width=2,
            annotation_text="5% (Significant -)",
            annotation_position="right",
            annotation_font=dict(size=10, color=self.colors['significant_neg'])
        )

        self._apply_layout(fig, 'Bayesian P(Treatment > Control)', 400)
        fig.update_layout(
            xaxis_title='Segment',
            yaxis_title='Probability (%)',
            yaxis=dict(range=[0, 105]),
            bargap=0.3,
            margin=dict(r=120)
        )

        return fig

    def plot_bayesian_credible_intervals(self, results: List[ABTestResult]) -> go.Figure:
        """Create forest plot showing Bayesian credible intervals for effect"""
        segments = [r.segment for r in results]
        effects = [r.effect_size for r in results]
        ci_lower = [r.bayesian_credible_interval[0] for r in results]
        ci_upper = [r.bayesian_credible_interval[1] for r in results]

        colors = significance_colors(
            values=effects,
            is_significant=[r.bayesian_is_significant for r in results],
            positive_color=self.colors['significant_pos'],
            negative_color=self.colors['significant_neg'],
            neutral_color=self.colors['not_significant'],
        )

        fig = go.Figure()

        # Error bars for credible intervals
        fig.add_trace(go.Scatter(
            x=effects,
            y=segments,
            mode='markers',
            marker=dict(size=12, color=colors, symbol='diamond'),
            error_x=interval_error_arrays(
                values=effects,
                lower_bounds=ci_lower,
                upper_bounds=ci_upper,
            ),
            text=[f'{e:.4f} [{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]' for i, e in enumerate(effects)],
            hoverinfo='text+y',
            showlegend=False
        ))

        # Add vertical line at 0
        fig.add_vline(x=0, line_dash="solid", line_color=self.colors['grid'], line_width=2)

        self._apply_layout(fig, 'Bayesian 95% Credible Intervals for Effect', 400)
        fig.update_layout(
            xaxis_title='Effect Size (Treatment - Control)',
            yaxis_title='Segment',
            yaxis=dict(autorange='reversed')  # Keep segments in order from top to bottom
        )

        add_significance_legend(
            fig,
            positive_color=self.colors['significant_pos'],
            negative_color=self.colors['significant_neg'],
            neutral_color=self.colors['not_significant'],
            symbol='diamond',
        )
        fig.update_layout(
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255,255,255,0.8)'
            )
        )

        return fig

    def plot_bayesian_expected_loss(self, results: List[ABTestResult]) -> go.Figure:
        """Create bar chart showing expected loss for each segment"""
        segments = [r.segment for r in results]
        # For each segment, show the minimum expected loss (best decision)
        min_losses = [min(r.bayesian_expected_loss_treatment, r.bayesian_expected_loss_control)
                      for r in results]

        # Color based on which option has lower loss
        colors = [
            self.colors['treatment'] if r.bayesian_expected_loss_treatment <= r.bayesian_expected_loss_control
            else self.colors['control']
            for r in results
        ]

        fig = go.Figure()

        fig.add_trace(
            make_bar_trace(
                x=segments,
                y=min_losses,
                marker_color=colors,
                text=[f'{loss:.4f}' for loss in min_losses],
            )
        )

        self._apply_layout(fig, 'Bayesian Expected Loss (Lower is Better)', 400)
        fig.update_layout(
            xaxis_title='Segment',
            yaxis_title='Expected Loss',
            bargap=0.3
        )

        # Add legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(size=10, color=self.colors['treatment'], symbol='square'),
                                name='Recommend Treatment'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(size=10, color=self.colors['control'], symbol='square'),
                                name='Recommend Control'))
        fig.update_layout(
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255,255,255,0.8)'
            )
        )

        return fig

    def plot_effect_waterfall(self, results: List[ABTestResult]) -> go.Figure:
        """Create waterfall chart showing combined effect contribution by segment"""
        # Sort by combined total effect
        sorted_results = sorted(results, key=lambda r: r.total_effect, reverse=True)

        segments = [r.segment for r in sorted_results]
        contributions = [r.total_effect for r in sorted_results]

        fig = go.Figure(go.Waterfall(
            name="Combined Effect",
            orientation="v",
            measure=["relative"] * len(segments) + ["total"],
            x=segments + ["Total"],
            y=contributions + [sum(contributions)],
            textposition="outside",
            text=[f"{c:+,.0f}" for c in contributions] + [f"{sum(contributions):,.0f}"],
            textfont=dict(size=10),
            connector={"line": {"color": self.colors['grid'], "width": 1}},
            increasing={"marker": {"color": self.colors['significant_pos']}},
            decreasing={"marker": {"color": self.colors['significant_neg']}},
            totals={"marker": {"color": self.colors['combined']}}
        ))

        self._apply_layout(fig, 'Combined Effect Contribution (T-test + Proportion)', 400)
        fig.update_layout(
            xaxis_title='Segment',
            yaxis_title='Total Effect'
        )

        return fig

    def plot_segment_distribution(self, df: pd.DataFrame, group_col: str,
                                  segment_col: Optional[str] = None) -> go.Figure:
        """Create pie or sunburst chart of segment distribution"""
        if segment_col:
            counts = df.groupby([segment_col, group_col]).size().reset_index(name='count')

            fig = px.sunburst(
                counts,
                path=[segment_col, group_col],
                values='count',
                color=group_col,
                color_discrete_map={
                    'treatment': self.colors['treatment'],
                    'control': self.colors['control'],
                    'test': self.colors['treatment'],
                    'ctrl': self.colors['control']
                }
            )
            title = 'Customer Distribution by Segment'
        else:
            counts = df[group_col].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=counts.index,
                values=counts.values,
                marker_colors=[self.colors['treatment'], self.colors['control']],
                hole=0.4,
                textinfo='label+percent',
                textfont=dict(size=12)
            )])
            title = 'Treatment vs Control Distribution'

        self._apply_layout(fig, title, 450)
        return fig

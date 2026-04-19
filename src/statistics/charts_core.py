"""Core chart mixins for A/B test visualizations.

Contains the most commonly used chart methods: treatment vs control,
effect sizes, combined effects, proportions, p-values, sample sizes,
power analysis, and Cohen's d.

These methods rely on ``self.colors`` and ``self._apply_layout`` provided
by the main ``ABTestVisualizer`` class.
"""

from __future__ import annotations

from typing import List

import plotly.graph_objects as go

from .chart_builders import (
    add_grouped_pair_bars,
    add_significance_legend,
    apply_grouped_bar_layout,
    interval_error_arrays,
    make_bar_trace,
    significance_colors,
)
from .models import ABTestResult


class CoreChartsMixin:
    """Mixin providing core A/B chart methods."""

    def plot_treatment_vs_control(self, results: List[ABTestResult]) -> go.Figure:
        """Create grouped bar chart comparing treatment vs control means by segment"""
        segments = [r.segment for r in results]
        treatment_means = [r.treatment_mean for r in results]
        control_means = [r.control_mean for r in results]

        fig = go.Figure()
        add_grouped_pair_bars(
            fig,
            segments=segments,
            left_name='Treatment',
            left_values=treatment_means,
            left_color=self.colors['treatment'],
            right_name='Control',
            right_values=control_means,
            right_color=self.colors['control'],
            left_text=[f'{v:.1f}' for v in treatment_means],
            right_text=[f'{v:.1f}' for v in control_means],
        )

        self._apply_layout(fig, 'Treatment vs Control Mean by Segment', 400)
        apply_grouped_bar_layout(fig, xaxis_title='Segment', yaxis_title='Mean Value')

        return fig

    def plot_effect_sizes(self, results: List[ABTestResult]) -> go.Figure:
        """Create bar chart of effect sizes with confidence intervals"""
        segments = [r.segment for r in results]
        effects = [r.effect_size for r in results]
        ci_lower = [r.confidence_interval[0] for r in results]
        ci_upper = [r.confidence_interval[1] for r in results]

        colors = significance_colors(
            values=effects,
            is_significant=[r.is_significant for r in results],
            positive_color=self.colors['significant_pos'],
            negative_color=self.colors['significant_neg'],
            neutral_color=self.colors['not_significant'],
        )

        fig = go.Figure()
        fig.add_trace(
            make_bar_trace(
                x=segments,
                y=effects,
                marker_color=colors,
                text=[f'{e:+.2f}' for e in effects],
                error_y=interval_error_arrays(
                    values=effects,
                    lower_bounds=ci_lower,
                    upper_bounds=ci_upper,
                ),
                showlegend=False,
            )
        )

        fig.add_hline(y=0, line_dash="solid", line_color=self.colors['grid'], line_width=1)

        self._apply_layout(fig, 'Effect Size by Segment (with 95% CI)', 400)
        fig.update_layout(
            xaxis_title='Segment',
            yaxis_title='Effect Size',
            bargap=0.3
        )

        add_significance_legend(
            fig,
            positive_color=self.colors['significant_pos'],
            negative_color=self.colors['significant_neg'],
            neutral_color=self.colors['not_significant'],
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

    def plot_combined_effects(self, results: List[ABTestResult]) -> go.Figure:
        """Create stacked bar chart showing T-test and Proportion effects per segment"""
        segments = [r.segment for r in results]

        # Calculate effects (using per-customer values for clearer comparison)
        t_test_effects = [r.effect_size if r.is_significant else 0 for r in results]
        prop_effects = [r.proportion_effect_per_customer for r in results]
        total_effects = [r.total_effect_per_customer for r in results]

        fig = go.Figure()

        # T-test effect bars
        fig.add_trace(go.Bar(
            name='T-test Effect',
            x=segments,
            y=t_test_effects,
            marker_color=self.colors['t_test'],
            marker_line_width=0
        ))

        # Proportion effect bars (stacked)
        fig.add_trace(go.Bar(
            name='Proportion Effect',
            x=segments,
            y=prop_effects,
            marker_color=self.colors['proportion'],
            marker_line_width=0
        ))

        # Add total line markers
        fig.add_trace(go.Scatter(
            name='Combined Total',
            x=segments,
            y=total_effects,
            mode='markers+text',
            marker=dict(
                color=self.colors['combined'],
                size=12,
                symbol='diamond'
            ),
            text=[f'{t:.3f}' for t in total_effects],
            textposition='top center',
            textfont=dict(size=9)
        ))

        fig.add_hline(y=0, line_dash="solid", line_color=self.colors['grid'], line_width=1)

        self._apply_layout(fig, 'Combined Effects by Segment (T-test + Proportion)', 450)
        fig.update_layout(
            barmode='stack',
            xaxis_title='Segment',
            yaxis_title='Effect per Customer',
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

    def plot_proportion_comparison(self, results: List[ABTestResult]) -> go.Figure:
        """Create grouped bar chart comparing treatment vs control proportions"""
        segments = [r.segment for r in results]
        treatment_props = [r.treatment_proportion * 100 for r in results]
        control_props = [r.control_proportion * 100 for r in results]

        fig = go.Figure()
        add_grouped_pair_bars(
            fig,
            segments=segments,
            left_name='Treatment %',
            left_values=treatment_props,
            left_color=self.colors['treatment'],
            right_name='Control %',
            right_values=control_props,
            right_color=self.colors['control'],
            left_text=[f'{v:.1f}%' for v in treatment_props],
            right_text=[f'{v:.1f}%' for v in control_props],
        )

        self._apply_layout(fig, 'Conversion Rate: Treatment vs Control', 400)
        apply_grouped_bar_layout(
            fig,
            xaxis_title='Segment',
            yaxis_title='Conversion Rate (%)',
        )

        return fig

    def plot_p_values(self, results: List[ABTestResult]) -> go.Figure:
        """Create bar chart of p-values with significance threshold line"""
        segments = [r.segment for r in results]
        p_values = [r.p_value for r in results]

        colors = [
            self.colors['significant_pos'] if p < 0.05 else self.colors['not_significant']
            for p in p_values
        ]

        fig = go.Figure()
        fig.add_trace(
            make_bar_trace(
                x=segments,
                y=p_values,
                marker_color=colors,
                text=[f'{p:.4f}' for p in p_values],
            )
        )

        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color=self.colors['significant_neg'],
            line_width=2,
            annotation_text="\u03b1 = 0.05",
            annotation_position="right",
            annotation_font=dict(size=10, color=self.colors['significant_neg'])
        )

        self._apply_layout(fig, 'P-Values by Segment', 400)
        fig.update_layout(
            xaxis_title='Segment',
            yaxis_title='P-Value',
            yaxis=dict(range=[0, max(max(p_values) * 1.3, 0.1)]),
            bargap=0.3
        )

        return fig

    def plot_sample_sizes(self, results: List[ABTestResult]) -> go.Figure:
        """Create grouped bar chart of sample sizes by segment"""
        segments = [r.segment for r in results]
        treatment_n = [r.treatment_size for r in results]
        control_n = [r.control_size for r in results]

        fig = go.Figure()
        add_grouped_pair_bars(
            fig,
            segments=segments,
            left_name='Treatment',
            left_values=treatment_n,
            left_color=self.colors['treatment'],
            right_name='Control',
            right_values=control_n,
            right_color=self.colors['control'],
            left_text=treatment_n,
            right_text=control_n,
        )

        self._apply_layout(fig, 'Sample Sizes by Segment', 400)
        apply_grouped_bar_layout(fig, xaxis_title='Segment', yaxis_title='Sample Size')

        return fig

    def plot_power_analysis(self, results: List[ABTestResult]) -> go.Figure:
        """Create bar chart of statistical power with adequacy threshold"""
        segments = [r.segment for r in results]
        powers = [r.power * 100 for r in results]

        colors = [self.colors['adequate'] if r.is_sample_adequate else self.colors['inadequate']
                 for r in results]

        fig = go.Figure()

        fig.add_trace(
            make_bar_trace(
                x=segments,
                y=powers,
                marker_color=colors,
                text=[f'{p:.0f}%' for p in powers],
            )
        )

        fig.add_hline(
            y=80,
            line_dash="dash",
            line_color=self.colors['inadequate'],
            line_width=2,
            annotation_text="80% threshold",
            annotation_position="right",
            annotation_font=dict(size=10, color=self.colors['inadequate'])
        )

        self._apply_layout(fig, 'Statistical Power by Segment', 400)
        fig.update_layout(
            xaxis_title='Segment',
            yaxis_title='Power (%)',
            yaxis=dict(range=[0, 110]),
            bargap=0.3
        )

        return fig

    def plot_cohens_d(self, results: List[ABTestResult]) -> go.Figure:
        """Create bar chart of Cohen's d effect sizes with interpretation bands"""
        segments = [r.segment for r in results]
        cohens_d = [r.cohens_d for r in results]

        colors = []
        for d in cohens_d:
            abs_d = abs(d)
            if abs_d >= 0.8:
                colors.append(self.colors['significant_neg'] if d < 0 else self.colors['significant_pos'])
            elif abs_d >= 0.5:
                colors.append('#F97316' if d < 0 else '#22C55E')  # Orange / Light green
            elif abs_d >= 0.2:
                colors.append('#FBBF24' if d < 0 else '#86EFAC')  # Yellow / Pale green
            else:
                colors.append(self.colors['not_significant'])

        fig = go.Figure()

        fig.add_trace(
            make_bar_trace(
                x=segments,
                y=cohens_d,
                marker_color=colors,
                text=[f'{d:.2f}' for d in cohens_d],
            )
        )

        # Add reference bands
        for y_val, _label in [(0.8, 'Large'), (0.5, 'Medium'), (0.2, 'Small')]:
            fig.add_hline(y=y_val, line_dash="dot", line_color=self.colors['grid'], line_width=1)
            fig.add_hline(y=-y_val, line_dash="dot", line_color=self.colors['grid'], line_width=1)

        fig.add_hline(y=0, line_dash="solid", line_color=self.colors['text'], line_width=1)

        self._apply_layout(fig, "Cohen's d Effect Size by Segment", 400)
        fig.update_layout(
            xaxis_title='Segment',
            yaxis_title="Cohen's d",
            bargap=0.3,
            annotations=[
                dict(x=1.02, y=0.8, xref='paper', yref='y', text='Large', showarrow=False,
                     font=dict(size=9, color=self.colors['not_significant'])),
                dict(x=1.02, y=0.5, xref='paper', yref='y', text='Medium', showarrow=False,
                     font=dict(size=9, color=self.colors['not_significant'])),
                dict(x=1.02, y=0.2, xref='paper', yref='y', text='Small', showarrow=False,
                     font=dict(size=9, color=self.colors['not_significant'])),
            ],
            margin=dict(r=80)
        )

        return fig

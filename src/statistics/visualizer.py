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
import plotly.express as px
from plotly.subplots import make_subplots

from .models import ABTestResult


class ABTestVisualizer:
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

    def _apply_layout(self, fig: go.Figure, title: str, height: int = 450) -> go.Figure:
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

    def plot_treatment_vs_control(self, results: List[ABTestResult]) -> go.Figure:
        """Create grouped bar chart comparing treatment vs control means by segment"""
        segments = [r.segment for r in results]
        treatment_means = [r.treatment_mean for r in results]
        control_means = [r.control_mean for r in results]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Treatment',
            x=segments,
            y=treatment_means,
            marker_color=self.colors['treatment'],
            marker_line_width=0,
            text=[f'{v:.1f}' for v in treatment_means],
            textposition='outside',
            textfont=dict(size=10)
        ))

        fig.add_trace(go.Bar(
            name='Control',
            x=segments,
            y=control_means,
            marker_color=self.colors['control'],
            marker_line_width=0,
            text=[f'{v:.1f}' for v in control_means],
            textposition='outside',
            textfont=dict(size=10)
        ))

        self._apply_layout(fig, 'Treatment vs Control Mean by Segment', 400)
        fig.update_layout(
            barmode='group',
            bargap=0.2,
            bargroupgap=0.1,
            xaxis_title='Segment',
            yaxis_title='Mean Value',
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

    def plot_effect_sizes(self, results: List[ABTestResult]) -> go.Figure:
        """Create bar chart of effect sizes with confidence intervals"""
        segments = [r.segment for r in results]
        effects = [r.effect_size for r in results]
        ci_lower = [r.confidence_interval[0] for r in results]
        ci_upper = [r.confidence_interval[1] for r in results]

        colors = []
        for r in results:
            if r.is_significant:
                colors.append(self.colors['significant_pos'] if r.effect_size > 0
                            else self.colors['significant_neg'])
            else:
                colors.append(self.colors['not_significant'])

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=segments,
            y=effects,
            marker_color=colors,
            marker_line_width=0,
            text=[f'{e:+.2f}' for e in effects],
            textposition='outside',
            textfont=dict(size=10),
            error_y=dict(
                type='data',
                symmetric=False,
                array=[ci_upper[i] - effects[i] for i in range(len(effects))],
                arrayminus=[effects[i] - ci_lower[i] for i in range(len(effects))],
                color='rgba(0,0,0,0.2)',
                thickness=1.5,
                width=4
            ),
            showlegend=False
        ))

        fig.add_hline(y=0, line_dash="solid", line_color=self.colors['grid'], line_width=1)

        self._apply_layout(fig, 'Effect Size by Segment (with 95% CI)', 400)
        fig.update_layout(
            xaxis_title='Segment',
            yaxis_title='Effect Size',
            bargap=0.3
        )

        # Add legend for colors
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(size=10, color=self.colors['significant_pos'], symbol='square'),
                                name='Significant (+)'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(size=10, color=self.colors['significant_neg'], symbol='square'),
                                name='Significant (-)'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(size=10, color=self.colors['not_significant'], symbol='square'),
                                name='Not Significant'))
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

    def plot_p_values(self, results: List[ABTestResult]) -> go.Figure:
        """Create bar chart of p-values with significance threshold line"""
        segments = [r.segment for r in results]
        p_values = [r.p_value for r in results]

        colors = [self.colors['significant_pos'] if p < 0.05 else self.colors['not_significant']
                 for p in p_values]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=segments,
            y=p_values,
            marker_color=colors,
            marker_line_width=0,
            text=[f'{p:.4f}' for p in p_values],
            textposition='outside',
            textfont=dict(size=10)
        ))

        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color=self.colors['significant_neg'],
            line_width=2,
            annotation_text="α = 0.05",
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

        fig.add_trace(go.Bar(
            name='Treatment',
            x=segments,
            y=treatment_n,
            marker_color=self.colors['treatment'],
            marker_line_width=0,
            text=treatment_n,
            textposition='outside',
            textfont=dict(size=10)
        ))

        fig.add_trace(go.Bar(
            name='Control',
            x=segments,
            y=control_n,
            marker_color=self.colors['control'],
            marker_line_width=0,
            text=control_n,
            textposition='outside',
            textfont=dict(size=10)
        ))

        self._apply_layout(fig, 'Sample Sizes by Segment', 400)
        fig.update_layout(
            barmode='group',
            bargap=0.2,
            bargroupgap=0.1,
            xaxis_title='Segment',
            yaxis_title='Sample Size',
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

    def plot_power_analysis(self, results: List[ABTestResult]) -> go.Figure:
        """Create bar chart of statistical power with adequacy threshold"""
        segments = [r.segment for r in results]
        powers = [r.power * 100 for r in results]

        colors = [self.colors['adequate'] if r.is_sample_adequate else self.colors['inadequate']
                 for r in results]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=segments,
            y=powers,
            marker_color=colors,
            marker_line_width=0,
            text=[f'{p:.0f}%' for p in powers],
            textposition='outside',
            textfont=dict(size=10)
        ))

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

        fig.add_trace(go.Bar(
            x=segments,
            y=cohens_d,
            marker_color=colors,
            marker_line_width=0,
            text=[f'{d:.2f}' for d in cohens_d],
            textposition='outside',
            textfont=dict(size=10)
        ))

        # Add reference bands
        for y_val, label in [(0.8, 'Large'), (0.5, 'Medium'), (0.2, 'Small')]:
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

    def plot_statistical_summary(self, results: List[ABTestResult]) -> go.Figure:
        """
        Create a comprehensive 2x3 dashboard showing all key metrics from the results table:
        - T-test p-values
        - T-test effect sizes
        - Proportion test p-values
        - Proportion test effect sizes
        - Total effect sizes
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                '<b>T-test P-Values</b>',
                '<b>T-test Effect Size</b>',
                '<b>Total Effect Size</b>',
                '<b>Proportion P-Values</b>',
                '<b>Proportion Effect Size</b>',
                '<b>All Effects Comparison</b>'
            ),
            vertical_spacing=0.25,
            horizontal_spacing=0.12
        )

        segments = [r.segment for r in results]

        # Row 1, Col 1: T-test P-Values
        t_pvals = [r.p_value for r in results]
        t_pval_colors = [self.colors['significant_pos'] if p < 0.05 else self.colors['not_significant'] for p in t_pvals]
        fig.add_trace(go.Bar(
            x=segments, y=t_pvals,
            marker_color=t_pval_colors,
            text=[f'{p:.4f}' for p in t_pvals],
            textposition='outside',
            textfont=dict(size=11),
            showlegend=False
        ), row=1, col=1)
        fig.add_hline(y=0.05, line_dash="dash", line_color=self.colors['significant_neg'], line_width=2, row=1, col=1)

        # Row 1, Col 2: T-test Effect Size
        t_effects = [r.effect_size for r in results]
        t_effect_colors = [
            self.colors['significant_pos'] if r.is_significant and r.effect_size > 0
            else self.colors['significant_neg'] if r.is_significant and r.effect_size < 0
            else self.colors['not_significant']
            for r in results
        ]
        fig.add_trace(go.Bar(
            x=segments, y=t_effects,
            marker_color=t_effect_colors,
            text=[f'{e:.4f}' for e in t_effects],
            textposition='outside',
            textfont=dict(size=11),
            showlegend=False
        ), row=1, col=2)
        fig.add_hline(y=0, line_dash="solid", line_color=self.colors['grid'], row=1, col=2)

        # Row 1, Col 3: Total Effect Size
        total_effects = [r.total_effect_per_customer for r in results]
        total_colors = [
            self.colors['combined'] if t > 0 else self.colors['significant_neg'] if t < 0 else self.colors['not_significant']
            for t in total_effects
        ]
        fig.add_trace(go.Bar(
            x=segments, y=total_effects,
            marker_color=total_colors,
            text=[f'{t:.4f}' for t in total_effects],
            textposition='outside',
            textfont=dict(size=11),
            showlegend=False
        ), row=1, col=3)
        fig.add_hline(y=0, line_dash="solid", line_color=self.colors['grid'], row=1, col=3)

        # Row 2, Col 1: Proportion P-Values
        prop_pvals = [r.proportion_p_value for r in results]
        prop_pval_colors = [self.colors['significant_pos'] if p < 0.05 else self.colors['not_significant'] for p in prop_pvals]
        fig.add_trace(go.Bar(
            x=segments, y=prop_pvals,
            marker_color=prop_pval_colors,
            text=[f'{p:.4f}' for p in prop_pvals],
            textposition='outside',
            textfont=dict(size=11),
            showlegend=False
        ), row=2, col=1)
        fig.add_hline(y=0.05, line_dash="dash", line_color=self.colors['significant_neg'], line_width=2, row=2, col=1)

        # Row 2, Col 2: Proportion Effect Size
        prop_effects = [r.proportion_effect_per_customer for r in results]
        prop_effect_colors = [
            self.colors['proportion'] if r.proportion_is_significant and e > 0
            else self.colors['significant_neg'] if r.proportion_is_significant and e < 0
            else self.colors['not_significant']
            for r, e in zip(results, prop_effects)
        ]
        fig.add_trace(go.Bar(
            x=segments, y=prop_effects,
            marker_color=prop_effect_colors,
            text=[f'{e:.4f}' for e in prop_effects],
            textposition='outside',
            textfont=dict(size=11),
            showlegend=False
        ), row=2, col=2)
        fig.add_hline(y=0, line_dash="solid", line_color=self.colors['grid'], row=2, col=2)

        # Row 2, Col 3: All Effects Comparison (grouped bar)
        fig.add_trace(go.Bar(
            name='T-test Effect',
            x=segments, y=t_effects,
            marker_color=self.colors['t_test'],
            showlegend=True
        ), row=2, col=3)
        fig.add_trace(go.Bar(
            name='Prop Effect',
            x=segments, y=prop_effects,
            marker_color=self.colors['proportion'],
            showlegend=True
        ), row=2, col=3)
        fig.add_trace(go.Bar(
            name='Total Effect',
            x=segments, y=total_effects,
            marker_color=self.colors['combined'],
            showlegend=True
        ), row=2, col=3)
        fig.add_hline(y=0, line_dash="solid", line_color=self.colors['grid'], row=2, col=3)

        # Apply layout (without layout_defaults to avoid duplicate margin)
        fig.update_layout(
            template='plotly_white',
            font=dict(family='Inter, system-ui, sans-serif', size=12, color=self.colors['text']),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            hoverlabel=dict(bgcolor='white', font_size=12),
            title=dict(
                text='<b>Statistical Results Summary</b><br><span style="font-size:12px;color:#6B7280">T-test & Proportion Test: P-Values, Effect Sizes, and Total Effects</span>',
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            height=800,
            barmode='group',
            bargap=0.3,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255,255,255,0.9)',
                font=dict(size=11)
            ),
            margin=dict(l=70, r=50, t=120, b=60)
        )

        # Update axes labels
        fig.update_yaxes(title_text='P-Value', row=1, col=1, title_font=dict(size=12))
        fig.update_yaxes(title_text='Effect', row=1, col=2, title_font=dict(size=12))
        fig.update_yaxes(title_text='Total Effect', row=1, col=3, title_font=dict(size=12))
        fig.update_yaxes(title_text='P-Value', row=2, col=1, title_font=dict(size=12))
        fig.update_yaxes(title_text='Effect', row=2, col=2, title_font=dict(size=12))
        fig.update_yaxes(title_text='Effect', row=2, col=3, title_font=dict(size=12))

        # Style all subplots
        for i in range(1, 3):
            for j in range(1, 4):
                fig.update_xaxes(showgrid=False, row=i, col=j, tickfont=dict(size=11))
                fig.update_yaxes(gridcolor=self.colors['grid'], row=i, col=j, tickfont=dict(size=11))

        # Update subplot title styling
        for annotation in fig['layout']['annotations']:
            if annotation['text'].startswith('<b>'):
                annotation['font'] = dict(size=14, color=self.colors['text'])

        return fig

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

        fig.add_trace(go.Bar(
            x=segments,
            y=probs,
            marker_color=colors,
            marker_line_width=0,
            text=[f'{p:.1f}%' for p in probs],
            textposition='outside',
            textfont=dict(size=10)
        ))

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

        colors = []
        for r in results:
            if r.bayesian_is_significant:
                colors.append(self.colors['significant_pos'] if r.effect_size > 0
                            else self.colors['significant_neg'])
            else:
                colors.append(self.colors['not_significant'])

        fig = go.Figure()

        # Error bars for credible intervals
        fig.add_trace(go.Scatter(
            x=effects,
            y=segments,
            mode='markers',
            marker=dict(size=12, color=colors, symbol='diamond'),
            error_x=dict(
                type='data',
                symmetric=False,
                array=[ci_upper[i] - effects[i] for i in range(len(effects))],
                arrayminus=[effects[i] - ci_lower[i] for i in range(len(effects))],
                color='rgba(0,0,0,0.3)',
                thickness=2,
                width=6
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

        # Add legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(size=10, color=self.colors['significant_pos'], symbol='diamond'),
                                name='Significant (+)'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(size=10, color=self.colors['significant_neg'], symbol='diamond'),
                                name='Significant (-)'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(size=10, color=self.colors['not_significant'], symbol='diamond'),
                                name='Not Significant'))
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

        fig.add_trace(go.Bar(
            x=segments,
            y=min_losses,
            marker_color=colors,
            marker_line_width=0,
            text=[f'{loss:.4f}' for loss in min_losses],
            textposition='outside',
            textfont=dict(size=10)
        ))

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

    def plot_proportion_comparison(self, results: List[ABTestResult]) -> go.Figure:
        """Create grouped bar chart comparing treatment vs control proportions"""
        segments = [r.segment for r in results]
        treatment_props = [r.treatment_proportion * 100 for r in results]
        control_props = [r.control_proportion * 100 for r in results]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Treatment %',
            x=segments,
            y=treatment_props,
            marker_color=self.colors['treatment'],
            marker_line_width=0,
            text=[f'{v:.1f}%' for v in treatment_props],
            textposition='outside',
            textfont=dict(size=10)
        ))

        fig.add_trace(go.Bar(
            name='Control %',
            x=segments,
            y=control_props,
            marker_color=self.colors['control'],
            marker_line_width=0,
            text=[f'{v:.1f}%' for v in control_props],
            textposition='outside',
            textfont=dict(size=10)
        ))

        self._apply_layout(fig, 'Conversion Rate: Treatment vs Control', 400)
        fig.update_layout(
            barmode='group',
            bargap=0.2,
            bargroupgap=0.1,
            xaxis_title='Segment',
            yaxis_title='Conversion Rate (%)',
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

    def plot_summary_dashboard(self, results: List[ABTestResult], summary: Dict[str, Any]) -> go.Figure:
        """Create a comprehensive dashboard with multiple visualizations"""
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
        fig.add_trace(go.Bar(
            name='Treatment',
            x=segments,
            y=[r.treatment_mean for r in results],
            marker_color=self.colors['treatment'],
            marker_line_width=0,
            showlegend=True
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            name='Control',
            x=segments,
            y=[r.control_mean for r in results],
            marker_color=self.colors['control'],
            marker_line_width=0,
            showlegend=True
        ), row=1, col=1)

        # Plot 2: Combined Effects (stacked T-test + Proportion)
        t_test_effects = [r.effect_size * r.treatment_size if r.is_significant else 0 for r in results]
        prop_effects = [r.proportion_effect for r in results]

        fig.add_trace(go.Bar(
            name='T-test Effect',
            x=segments,
            y=t_test_effects,
            marker_color=self.colors['t_test'],
            marker_line_width=0,
            showlegend=True
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            name='Proportion Effect',
            x=segments,
            y=prop_effects,
            marker_color=self.colors['proportion'],
            marker_line_width=0,
            showlegend=True
        ), row=1, col=2)
        fig.add_hline(y=0, line_dash="solid", line_color=self.colors['grid'], row=1, col=2)

        # Plot 3: Conversion Rates (Proportion comparison)
        fig.add_trace(go.Bar(
            name='Treatment Conv',
            x=segments,
            y=[r.treatment_proportion * 100 for r in results],
            marker_color=self.colors['treatment'],
            marker_line_width=0,
            showlegend=False
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            name='Control Conv',
            x=segments,
            y=[r.control_proportion * 100 for r in results],
            marker_color=self.colors['control'],
            marker_line_width=0,
            showlegend=False
        ), row=2, col=1)

        # Plot 4: P-Values (T-test)
        p_values = [r.p_value for r in results]
        p_colors = [
            self.colors['significant_pos'] if p < 0.05 else self.colors['not_significant']
            for p in p_values
        ]
        fig.add_trace(go.Bar(
            x=segments,
            y=p_values,
            marker_color=p_colors,
            marker_line_width=0,
            showlegend=False
        ), row=2, col=2)
        fig.add_hline(y=0.05, line_dash="dash", line_color=self.colors['significant_neg'], line_width=1.5, row=2, col=2)

        # Apply layout
        t_test_sig = summary.get('t_test_significant_segments', summary.get('significant_segments', 0))
        prop_sig = summary.get('prop_test_significant_segments', 0)
        total_count = summary['total_segments_analyzed']
        combined_effect = summary.get('combined_total_effect', summary.get('total_effect_size', 0))

        fig.update_layout(
            template='plotly_white',
            font=dict(family='Inter, system-ui, sans-serif', size=12, color=self.colors['text']),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            hoverlabel=dict(bgcolor='white', font_size=12),
            title=dict(
                text=f"<b>A/B Test Analysis Dashboard</b><br><span style='font-size:12px;color:#6B7280'>T-test sig: {t_test_sig}/{total_count} · Prop sig: {prop_sig}/{total_count} · Combined effect: {combined_effect:,.0f}</span>",
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            height=650,
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.08,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255,255,255,0.9)'
            ),
            margin=dict(l=60, r=40, t=120, b=60)
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

        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(showgrid=False, row=i, col=j, tickfont=dict(size=10))
                fig.update_yaxes(gridcolor=self.colors['grid'], row=i, col=j, tickfont=dict(size=10))

        # Update subplot title styling
        for annotation in fig['layout']['annotations']:
            if annotation['text'].startswith('<b>'):
                annotation['font'] = dict(size=12, color=self.colors['text'])

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

    def create_all_charts(self, results: List[ABTestResult], summary: Dict[str, Any],
                         df: Optional[pd.DataFrame] = None,
                         group_col: Optional[str] = None,
                         segment_col: Optional[str] = None) -> Dict[str, go.Figure]:
        """
        Generate all visualization charts

        Returns a dictionary of chart name -> Plotly figure
        """
        charts = {
            'statistical_summary': self.plot_statistical_summary(results),  # New focused chart
            'dashboard': self.plot_summary_dashboard(results, summary),
            'treatment_vs_control': self.plot_treatment_vs_control(results),
            'effect_sizes': self.plot_effect_sizes(results),
            'combined_effects': self.plot_combined_effects(results),
            'proportion_comparison': self.plot_proportion_comparison(results),
            'p_values': self.plot_p_values(results),
            'sample_sizes': self.plot_sample_sizes(results),
            'power_analysis': self.plot_power_analysis(results),
            'cohens_d': self.plot_cohens_d(results),
            'effect_waterfall': self.plot_effect_waterfall(results),
            # Bayesian visualizations
            'bayesian_probability': self.plot_bayesian_probability(results),
            'bayesian_credible_intervals': self.plot_bayesian_credible_intervals(results),
            'bayesian_expected_loss': self.plot_bayesian_expected_loss(results)
        }

        if df is not None and group_col:
            charts['distribution'] = self.plot_segment_distribution(df, group_col, segment_col)

        return charts

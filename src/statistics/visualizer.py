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
            'text': '#374151'
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

    def plot_summary_dashboard(self, results: List[ABTestResult], summary: Dict[str, Any]) -> go.Figure:
        """Create a comprehensive dashboard with multiple visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>Treatment vs Control</b>',
                '<b>Effect Sizes</b>',
                '<b>Statistical Power</b>',
                '<b>P-Values</b>'
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

        # Plot 2: Effect Sizes with CI
        effects = [r.effect_size for r in results]
        effect_colors = [
            self.colors['significant_pos'] if r.is_significant and r.effect_size > 0
            else self.colors['significant_neg'] if r.is_significant and r.effect_size < 0
            else self.colors['not_significant']
            for r in results
        ]

        fig.add_trace(go.Bar(
            x=segments,
            y=effects,
            marker_color=effect_colors,
            marker_line_width=0,
            error_y=dict(
                type='data',
                symmetric=False,
                array=[r.confidence_interval[1] - r.effect_size for r in results],
                arrayminus=[r.effect_size - r.confidence_interval[0] for r in results],
                color='rgba(0,0,0,0.2)',
                thickness=1.5,
                width=3
            ),
            showlegend=False
        ), row=1, col=2)
        fig.add_hline(y=0, line_dash="solid", line_color=self.colors['grid'], row=1, col=2)

        # Plot 3: Power Analysis
        powers = [r.power * 100 for r in results]
        power_colors = [
            self.colors['adequate'] if r.is_sample_adequate else self.colors['inadequate']
            for r in results
        ]
        fig.add_trace(go.Bar(
            x=segments,
            y=powers,
            marker_color=power_colors,
            marker_line_width=0,
            showlegend=False
        ), row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color=self.colors['inadequate'], line_width=1.5, row=2, col=1)

        # Plot 4: P-Values
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
        sig_count = summary['significant_segments']
        total_count = summary['total_segments_analyzed']
        total_effect = summary['total_effect_size']

        fig.update_layout(
            **self.layout_defaults,
            title=dict(
                text=f"<b>A/B Test Analysis Dashboard</b><br><span style='font-size:12px;color:#6B7280'>{sig_count}/{total_count} significant segments · Total effect: {total_effect:,.0f}</span>",
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            height=600,
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
        fig.update_yaxes(title_text='Effect', row=1, col=2, title_font=dict(size=11))
        fig.update_yaxes(title_text='Power %', row=2, col=1, range=[0, 105], title_font=dict(size=11))
        fig.update_yaxes(title_text='P-Value', row=2, col=2, title_font=dict(size=11))

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
        """Create waterfall chart showing cumulative effect contribution by segment"""
        sorted_results = sorted(results, key=lambda r: r.effect_size * r.treatment_size, reverse=True)

        segments = [r.segment for r in sorted_results]
        contributions = [r.effect_size * r.treatment_size for r in sorted_results]

        fig = go.Figure(go.Waterfall(
            name="Effect Contribution",
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
            totals={"marker": {"color": self.colors['control']}}
        ))

        self._apply_layout(fig, 'Effect Contribution by Segment', 400)
        fig.update_layout(
            xaxis_title='Segment',
            yaxis_title='Effect × Sample Size'
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
            'dashboard': self.plot_summary_dashboard(results, summary),
            'treatment_vs_control': self.plot_treatment_vs_control(results),
            'effect_sizes': self.plot_effect_sizes(results),
            'p_values': self.plot_p_values(results),
            'sample_sizes': self.plot_sample_sizes(results),
            'power_analysis': self.plot_power_analysis(results),
            'cohens_d': self.plot_cohens_d(results),
            'effect_waterfall': self.plot_effect_waterfall(results)
        }

        if df is not None and group_col:
            charts['distribution'] = self.plot_segment_distribution(df, group_col, segment_col)

        return charts

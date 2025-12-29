"""
A/B Test Visualization Module

Creates interactive Plotly charts for A/B test results:
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

    Creates interactive Plotly charts for:
    - Treatment vs Control comparison
    - Effect sizes across segments
    - Statistical significance
    - Power analysis
    - Confidence intervals
    """

    def __init__(self):
        self.colors = {
            'treatment': '#2ecc71',
            'control': '#3498db',
            'significant_pos': '#27ae60',
            'significant_neg': '#e74c3c',
            'not_significant': '#95a5a6',
            'adequate': '#2ecc71',
            'inadequate': '#e74c3c'
        }

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
            text=[f'{v:.2f}' for v in treatment_means],
            textposition='outside'
        ))

        fig.add_trace(go.Bar(
            name='Control',
            x=segments,
            y=control_means,
            marker_color=self.colors['control'],
            text=[f'{v:.2f}' for v in control_means],
            textposition='outside'
        ))

        fig.update_layout(
            title='Treatment vs Control Mean by Segment',
            xaxis_title='Segment',
            yaxis_title='Mean Effect Value',
            barmode='group',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=450
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
            name='Effect Size',
            text=[f'{e:.2f}' for e in effects],
            textposition='outside',
            error_y=dict(
                type='data',
                symmetric=False,
                array=[ci_upper[i] - effects[i] for i in range(len(effects))],
                arrayminus=[effects[i] - ci_lower[i] for i in range(len(effects))],
                color='rgba(0,0,0,0.3)'
            )
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        annotations = []
        for i, r in enumerate(results):
            if r.is_significant:
                annotations.append(dict(
                    x=segments[i],
                    y=effects[i] + (ci_upper[i] - effects[i]) + 1,
                    text='*' if r.p_value < 0.05 else '',
                    showarrow=False,
                    font=dict(size=16, color='gold')
                ))

        fig.update_layout(
            title='Effect Size by Segment (with 95% CI)',
            xaxis_title='Segment',
            yaxis_title='Effect Size (Treatment - Control)',
            template='plotly_white',
            annotations=annotations,
            height=450,
            showlegend=False
        )

        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(size=10, color=self.colors['significant_pos']),
                                name='Significant Positive'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(size=10, color=self.colors['significant_neg']),
                                name='Significant Negative'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                marker=dict(size=10, color=self.colors['not_significant']),
                                name='Not Significant'))
        fig.update_layout(showlegend=True,
                         legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

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
            text=[f'{p:.4f}' for p in p_values],
            textposition='outside'
        ))

        fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                     annotation_text="alpha = 0.05", annotation_position="right")

        fig.update_layout(
            title='P-Values by Segment',
            xaxis_title='Segment',
            yaxis_title='P-Value',
            template='plotly_white',
            yaxis=dict(range=[0, max(max(p_values) * 1.2, 0.1)]),
            height=400
        )

        return fig

    def plot_sample_sizes(self, results: List[ABTestResult]) -> go.Figure:
        """Create stacked bar chart of sample sizes by segment"""
        segments = [r.segment for r in results]
        treatment_n = [r.treatment_size for r in results]
        control_n = [r.control_size for r in results]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Treatment',
            x=segments,
            y=treatment_n,
            marker_color=self.colors['treatment'],
            text=treatment_n,
            textposition='inside'
        ))

        fig.add_trace(go.Bar(
            name='Control',
            x=segments,
            y=control_n,
            marker_color=self.colors['control'],
            text=control_n,
            textposition='inside'
        ))

        fig.update_layout(
            title='Sample Sizes by Segment',
            xaxis_title='Segment',
            yaxis_title='Number of Customers',
            barmode='stack',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400
        )

        return fig

    def plot_power_analysis(self, results: List[ABTestResult]) -> go.Figure:
        """Create bar chart of statistical power with adequacy threshold"""
        segments = [r.segment for r in results]
        powers = [r.power for r in results]

        colors = [self.colors['adequate'] if r.is_sample_adequate else self.colors['inadequate']
                 for r in results]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=segments,
            y=[p * 100 for p in powers],
            marker_color=colors,
            text=[f'{p:.1%}' for p in powers],
            textposition='outside'
        ))

        fig.add_hline(y=80, line_dash="dash", line_color="orange",
                     annotation_text="80% Power Threshold", annotation_position="right")

        fig.update_layout(
            title='Statistical Power by Segment',
            xaxis_title='Segment',
            yaxis_title='Power (%)',
            template='plotly_white',
            yaxis=dict(range=[0, 110]),
            height=400
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
                colors.append('#e74c3c' if d < 0 else '#27ae60')
            elif abs_d >= 0.5:
                colors.append('#e67e22' if d < 0 else '#2ecc71')
            elif abs_d >= 0.2:
                colors.append('#f39c12' if d < 0 else '#82e0aa')
            else:
                colors.append(self.colors['not_significant'])

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=segments,
            y=cohens_d,
            marker_color=colors,
            text=[f'{d:.3f}' for d in cohens_d],
            textposition='outside'
        ))

        fig.add_hline(y=0.8, line_dash="dot", line_color="green", opacity=0.5)
        fig.add_hline(y=0.5, line_dash="dot", line_color="orange", opacity=0.5)
        fig.add_hline(y=0.2, line_dash="dot", line_color="yellow", opacity=0.5)
        fig.add_hline(y=0, line_dash="solid", line_color="gray")
        fig.add_hline(y=-0.2, line_dash="dot", line_color="yellow", opacity=0.5)
        fig.add_hline(y=-0.5, line_dash="dot", line_color="orange", opacity=0.5)
        fig.add_hline(y=-0.8, line_dash="dot", line_color="red", opacity=0.5)

        fig.add_annotation(x=1.02, y=0.8, xref='paper', text='Large (0.8)',
                          showarrow=False, font=dict(size=10))
        fig.add_annotation(x=1.02, y=0.5, xref='paper', text='Medium (0.5)',
                          showarrow=False, font=dict(size=10))
        fig.add_annotation(x=1.02, y=0.2, xref='paper', text='Small (0.2)',
                          showarrow=False, font=dict(size=10))

        fig.update_layout(
            title="Cohen's d Effect Size by Segment",
            xaxis_title='Segment',
            yaxis_title="Cohen's d",
            template='plotly_white',
            height=450,
            margin=dict(r=100)
        )

        return fig

    def plot_summary_dashboard(self, results: List[ABTestResult], summary: Dict[str, Any]) -> go.Figure:
        """Create a comprehensive dashboard with multiple visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Treatment vs Control Means',
                'Effect Sizes with 95% CI',
                'Statistical Power',
                'P-Values'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        segments = [r.segment for r in results]

        # Plot 1: Treatment vs Control Means
        fig.add_trace(go.Bar(
            name='Treatment', x=segments, y=[r.treatment_mean for r in results],
            marker_color=self.colors['treatment'], showlegend=True
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            name='Control', x=segments, y=[r.control_mean for r in results],
            marker_color=self.colors['control'], showlegend=True
        ), row=1, col=1)

        # Plot 2: Effect Sizes
        effects = [r.effect_size for r in results]
        colors = [self.colors['significant_pos'] if r.is_significant and r.effect_size > 0
                 else self.colors['significant_neg'] if r.is_significant and r.effect_size < 0
                 else self.colors['not_significant'] for r in results]

        fig.add_trace(go.Bar(
            x=segments, y=effects, marker_color=colors,
            error_y=dict(
                type='data', symmetric=False,
                array=[r.confidence_interval[1] - r.effect_size for r in results],
                arrayminus=[r.effect_size - r.confidence_interval[0] for r in results]
            ),
            showlegend=False
        ), row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

        # Plot 3: Power Analysis
        powers = [r.power * 100 for r in results]
        power_colors = [self.colors['adequate'] if r.is_sample_adequate
                       else self.colors['inadequate'] for r in results]
        fig.add_trace(go.Bar(
            x=segments, y=powers, marker_color=power_colors, showlegend=False
        ), row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="orange", row=2, col=1)

        # Plot 4: P-Values
        p_values = [r.p_value for r in results]
        p_colors = [self.colors['significant_pos'] if p < 0.05
                   else self.colors['not_significant'] for p in p_values]
        fig.add_trace(go.Bar(
            x=segments, y=p_values, marker_color=p_colors, showlegend=False
        ), row=2, col=2)
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", row=2, col=2)

        fig.update_layout(
            title=dict(
                text=f"A/B Test Analysis Dashboard<br><sub>Significant: {summary['significant_segments']}/{summary['total_segments_analyzed']} segments | "
                     f"Total Effect: {summary['total_effect_size']:.2f}</sub>",
                x=0.5
            ),
            template='plotly_white',
            height=700,
            barmode='group',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        fig.update_yaxes(title_text='Mean Value', row=1, col=1)
        fig.update_yaxes(title_text='Effect Size', row=1, col=2)
        fig.update_yaxes(title_text='Power (%)', row=2, col=1)
        fig.update_yaxes(title_text='P-Value', row=2, col=2)

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
                color_discrete_map={'treatment': self.colors['treatment'],
                                   'control': self.colors['control']}
            )
            fig.update_layout(title='Customer Distribution by Segment and Group')
        else:
            counts = df[group_col].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=counts.index,
                values=counts.values,
                marker_colors=[self.colors['treatment'], self.colors['control']],
                hole=0.4
            )])
            fig.update_layout(title='Treatment vs Control Distribution')

        fig.update_layout(height=450, template='plotly_white')
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
            text=[f"{c:.1f}" for c in contributions] + [f"{sum(contributions):.1f}"],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": self.colors['significant_pos']}},
            decreasing={"marker": {"color": self.colors['significant_neg']}},
            totals={"marker": {"color": "#3498db"}}
        ))

        fig.update_layout(
            title="Total Effect Contribution by Segment (Effect x Treatment Size)",
            xaxis_title="Segment",
            yaxis_title="Effect Contribution",
            template='plotly_white',
            height=450,
            showlegend=False
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

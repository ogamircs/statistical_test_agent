"""Reusable Plotly chart-building helpers for A/B test visualizations."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import plotly.graph_objects as go


def interval_error_arrays(
    *,
    values: Sequence[float],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
) -> dict[str, Any]:
    """Build Plotly asymmetric error arrays from lower/upper interval bounds."""
    return {
        "type": "data",
        "symmetric": False,
        "array": [upper_bounds[idx] - values[idx] for idx in range(len(values))],
        "arrayminus": [values[idx] - lower_bounds[idx] for idx in range(len(values))],
        "color": "rgba(0,0,0,0.2)",
        "thickness": 1.5,
        "width": 4,
    }


def significance_colors(
    *,
    values: Iterable[float],
    is_significant: Iterable[bool],
    positive_color: str,
    negative_color: str,
    neutral_color: str,
) -> list[str]:
    """Assign colors based on sign and statistical significance."""
    colors: list[str] = []
    for value, significant in zip(values, is_significant):
        if significant:
            colors.append(positive_color if value >= 0 else negative_color)
        else:
            colors.append(neutral_color)
    return colors


def make_bar_trace(
    *,
    x: Sequence[Any],
    y: Sequence[Any],
    name: str | None = None,
    marker_color: Any,
    text: Sequence[Any] | None = None,
    showlegend: bool | None = None,
    error_y: dict[str, Any] | None = None,
    hovertemplate: str | None = None,
) -> go.Bar:
    """Create a consistently styled Plotly bar trace."""
    kwargs: dict[str, Any] = {
        "x": list(x),
        "y": list(y),
        "marker_color": marker_color,
        "marker_line_width": 0,
    }
    if name is not None:
        kwargs["name"] = name
    if text is not None:
        kwargs["text"] = list(text)
        kwargs["textposition"] = "outside"
        kwargs["textfont"] = {"size": 10}
    if showlegend is not None:
        kwargs["showlegend"] = showlegend
    if error_y is not None:
        kwargs["error_y"] = error_y
    if hovertemplate is not None:
        kwargs["hovertemplate"] = hovertemplate
    return go.Bar(**kwargs)


def add_grouped_pair_bars(
    fig: go.Figure,
    *,
    segments: Sequence[Any],
    left_name: str,
    left_values: Sequence[Any],
    left_color: str,
    right_name: str,
    right_values: Sequence[Any],
    right_color: str,
    left_text: Sequence[Any] | None = None,
    right_text: Sequence[Any] | None = None,
    row: int | None = None,
    col: int | None = None,
    showlegend: bool = True,
) -> None:
    """Add a standard treatment/control-style grouped pair of bars."""
    bar_kwargs = {"row": row, "col": col} if row is not None and col is not None else {}
    fig.add_trace(
        make_bar_trace(
            name=left_name,
            x=segments,
            y=left_values,
            marker_color=left_color,
            text=left_text,
            showlegend=showlegend,
        ),
        **bar_kwargs,
    )
    fig.add_trace(
        make_bar_trace(
            name=right_name,
            x=segments,
            y=right_values,
            marker_color=right_color,
            text=right_text,
            showlegend=showlegend,
        ),
        **bar_kwargs,
    )


def add_significance_legend(
    fig: go.Figure,
    *,
    positive_color: str,
    negative_color: str,
    neutral_color: str,
    symbol: str = "square",
) -> None:
    """Append a standard legend for positive/negative/non-significant outcomes."""
    for name, color in (
        ("Significant (+)", positive_color),
        ("Significant (-)", negative_color),
        ("Not Significant", neutral_color),
    ):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": color, "symbol": symbol},
                name=name,
            )
        )


def apply_grouped_bar_layout(
    fig: go.Figure,
    *,
    xaxis_title: str,
    yaxis_title: str,
    legend_y: float = 1.02,
    bargap: float = 0.2,
    bargroupgap: float = 0.1,
) -> None:
    """Apply the shared grouped-bar layout shell used by pairwise comparison charts."""
    fig.update_layout(
        barmode="group",
        bargap=bargap,
        bargroupgap=bargroupgap,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": legend_y,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255,255,255,0.8)",
        },
    )


def apply_multi_panel_theme(
    fig: go.Figure,
    *,
    colors: dict[str, str],
    title_text: str,
    height: int,
    legend_y: float = 1.02,
    barmode: str = "group",
    bargap: float = 0.2,
    bargroupgap: float = 0.1,
    margin: dict[str, int] | None = None,
) -> None:
    """Apply a shared multi-panel dashboard theme."""
    fig.update_layout(
        template="plotly_white",
        font={"family": "Inter, system-ui, sans-serif", "size": 12, "color": colors["text"]},
        paper_bgcolor=colors["background"],
        plot_bgcolor=colors["background"],
        hoverlabel={"bgcolor": "white", "font_size": 12},
        title={"text": title_text, "x": 0.5, "xanchor": "center"},
        height=height,
        barmode=barmode,
        bargap=bargap,
        bargroupgap=bargroupgap,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": legend_y,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255,255,255,0.9)",
        },
        margin=margin or {"l": 60, "r": 40, "t": 100, "b": 60},
    )


def style_subplot_axes(
    fig: go.Figure,
    *,
    rows: int,
    cols: int,
    grid_color: str,
    tickfont_size: int = 10,
    x_tickangle: int | None = None,
) -> None:
    """Apply consistent axis styling across every subplot cell."""
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            fig.update_xaxes(
                showgrid=False,
                row=row,
                col=col,
                tickfont={"size": tickfont_size},
                tickangle=x_tickangle,
                automargin=True,
            )
            fig.update_yaxes(
                gridcolor=grid_color,
                row=row,
                col=col,
                tickfont={"size": tickfont_size},
            )


def style_subplot_titles(fig: go.Figure, *, text_color: str, size: int = 12) -> None:
    """Normalize subplot title annotation styling."""
    for annotation in fig["layout"]["annotations"]:
        if annotation["text"].startswith("<b>"):
            annotation["font"] = {"size": size, "color": text_color}

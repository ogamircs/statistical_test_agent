import { useEffect, useImperativeHandle, useRef, type Ref } from "react";
import type { Config, Layout } from "plotly.js";
import type { ChartSpec } from "../lib/types";
import { COLORWAY, type Theme } from "../lib/theme";

type PlotlyModule = typeof import("plotly.js-dist-min").default;

let plotlyPromise: Promise<PlotlyModule> | null = null;
/** Plotly is ~4.5 MB: load it only when the first chart renders. */
export function loadPlotly(): Promise<PlotlyModule> {
  plotlyPromise ??= import("plotly.js-dist-min").then((module) => module.default);
  return plotlyPromise;
}

export interface PlotlyChartHandle {
  download: (format: "png" | "svg") => Promise<void>;
}

interface Props {
  chart: ChartSpec;
  theme: Theme;
  height?: number;
  handleRef?: Ref<PlotlyChartHandle>;
}

type Styled = { font?: Record<string, unknown> } & Record<string, unknown>;

/** Grey/black text the server bakes in; colored labels keep their meaning. */
export function isNeutralColor(color: unknown): boolean {
  if (typeof color !== "string") return true;
  const hex = color.trim().match(/^#([0-9a-f]{6})$/i)?.[1];
  const rgb = color.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
  let channels: number[] | null = null;
  if (hex) channels = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
  else if (rgb) channels = rgb.slice(1, 4).map(Number);
  if (!channels) return false;
  // Relative saturation: slate greys (#374151 ≈ 0.32) count as neutral, the
  // palette's meaningful colors (green/red/blue/amber ≥ 0.7) do not.
  const max = Math.max(...channels);
  return max === 0 || (max - Math.min(...channels)) / max < 0.4;
}

function withTextColor<T extends Styled | undefined>(item: T, color: string): T {
  if (!item || typeof item !== "object") return item;
  return { ...item, font: { ...(item.font ?? {}), color } };
}

export function themedLayout(layout: Partial<Layout>, theme: Theme, height?: number): Partial<Layout> {
  const dark = theme === "dark";
  const text = dark ? "#e6e8ee" : "#1f2430";
  const grid = dark ? "#343a48" : "#e3e6ec";
  const axis = { gridcolor: grid, zerolinecolor: grid, linecolor: grid, tickfont: { color: text } };
  const themed: Record<string, unknown> = {
    ...layout,
    autosize: true,
    height: height ?? layout.height,
    width: undefined,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { ...(layout.font ?? {}), color: text },
    colorway: COLORWAY,
    legend: { ...(layout.legend ?? {}), font: { color: text } },
  };
  // Server figures pin dark title/subplot-title colors that vanish on a dark
  // background; override every neutral text color with the theme's.
  if (layout.title && typeof layout.title === "object") {
    themed.title = withTextColor(layout.title as Styled, text);
  }
  if (Array.isArray(layout.annotations)) {
    themed.annotations = layout.annotations.map((note) =>
      isNeutralColor((note as Styled).font?.color) ? withTextColor(note as Styled, text) : note,
    );
  }
  // Restyle every axis the server figure defines (xaxis, yaxis2, …).
  for (const [key, value] of Object.entries(layout)) {
    if (/^[xy]axis\d*$/.test(key) && value && typeof value === "object") {
      const axisTitle = (value as Styled).title;
      themed[key] = {
        ...(value as object),
        ...axis,
        ...(axisTitle && typeof axisTitle === "object"
          ? { title: withTextColor(axisTitle as Styled, text) }
          : {}),
      };
    }
  }
  if (!("xaxis" in layout)) themed.xaxis = axis;
  if (!("yaxis" in layout)) themed.yaxis = axis;
  return themed as Partial<Layout>;
}

const PLOT_CONFIG: Partial<Config> = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};

/**
 * Height the chart should occupy. Plots.resize() runs an autosize relayout
 * that drops layout.height and re-measures the container, so the container
 * itself must carry the figure's height or charts collapse on the first
 * resize (e.g. when the chat pane reflows after a reply).
 */
export function chartHeight(chart: ChartSpec, height?: number): number {
  const fromFigure = chart.figure.layout?.height;
  return height ?? (typeof fromFigure === "number" ? fromFigure : 420);
}

export function PlotlyChart({ chart, theme, height, handleRef }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const targetHeight = chartHeight(chart, height);

  useImperativeHandle(handleRef, () => ({
    download: async (format) => {
      const Plotly = await loadPlotly();
      if (!container.current) return;
      await Plotly.downloadImage(container.current, {
        format,
        filename: chart.name,
        width: 1400,
        height: targetHeight,
      });
    },
  }));

  useEffect(() => {
    let cancelled = false;
    const element = container.current;
    let observer: ResizeObserver | null = null;
    void loadPlotly().then((Plotly) => {
      if (cancelled || !element) return;
      void Plotly.react(
        element,
        chart.figure.data,
        themedLayout(chart.figure.layout, theme, targetHeight),
        PLOT_CONFIG,
      );
      observer = new ResizeObserver(() => Plotly.Plots.resize(element));
      observer.observe(element);
    });
    return () => {
      cancelled = true;
      observer?.disconnect();
    };
  }, [chart, theme, targetHeight]);

  useEffect(() => {
    const element = container.current;
    return () => {
      if (element && plotlyPromise) void plotlyPromise.then((Plotly) => Plotly.purge(element));
    };
  }, []);

  return (
    <div
      ref={container}
      className="plot"
      role="img"
      aria-label={`Chart: ${chart.title}`}
      style={{ height: targetHeight }}
    />
  );
}

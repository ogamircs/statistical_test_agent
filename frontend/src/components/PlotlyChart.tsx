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

function themedLayout(layout: Partial<Layout>, theme: Theme, height?: number): Partial<Layout> {
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
  // Restyle every axis the server figure defines (xaxis, yaxis2, …).
  for (const [key, value] of Object.entries(layout)) {
    if (/^[xy]axis\d*$/.test(key) && value && typeof value === "object") {
      themed[key] = { ...(value as object), ...axis };
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

export function PlotlyChart({ chart, theme, height, handleRef }: Props) {
  const container = useRef<HTMLDivElement>(null);

  useImperativeHandle(handleRef, () => ({
    download: async (format) => {
      const Plotly = await loadPlotly();
      if (!container.current) return;
      await Plotly.downloadImage(container.current, {
        format,
        filename: chart.name,
        width: 1400,
        height: height ?? 800,
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
        themedLayout(chart.figure.layout, theme, height),
        PLOT_CONFIG,
      );
      observer = new ResizeObserver(() => Plotly.Plots.resize(element));
      observer.observe(element);
    });
    return () => {
      cancelled = true;
      observer?.disconnect();
    };
  }, [chart, theme, height]);

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
      style={{ minHeight: height ?? 420 }}
    />
  );
}

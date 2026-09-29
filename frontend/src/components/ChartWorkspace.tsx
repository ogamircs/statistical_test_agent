import { useEffect, useRef, useState } from "react";
import type { ChartSpec, ChartTypeOption } from "../lib/types";
import type { Theme } from "../lib/theme";
import { PlotlyChart, type PlotlyChartHandle } from "./PlotlyChart";

interface Props {
  charts: ChartSpec[];
  chartTypes: ChartTypeOption[];
  theme: Theme;
  loading: boolean;
  error: string | null;
  canRequest: boolean;
  onRequest: (type: string) => void;
  onClose: () => void;
}

export function ChartWorkspace({
  charts,
  chartTypes,
  theme,
  loading,
  error,
  canRequest,
  onRequest,
  onClose,
}: Props) {
  const [active, setActive] = useState(0);
  const [layout, setLayout] = useState<"tabs" | "grid">("tabs");
  const [selectedType, setSelectedType] = useState("dashboard");
  const [fullscreen, setFullscreen] = useState<ChartSpec | null>(null);
  const handle = useRef<PlotlyChartHandle>(null);

  useEffect(() => setActive(0), [charts]);
  const current = charts[Math.min(active, charts.length - 1)];

  return (
    <aside className="workspace" aria-label="Chart workspace">
      <header className="workspace-header">
        <h2>Charts</h2>
        <form
          className="chart-picker"
          onSubmit={(event) => {
            event.preventDefault();
            onRequest(selectedType);
          }}
        >
          <label className="visually-hidden" htmlFor="chart-type">
            Chart type
          </label>
          <select
            id="chart-type"
            value={selectedType}
            onChange={(event) => setSelectedType(event.target.value)}
          >
            {chartTypes.map((option) => (
              <option key={option.key} value={option.key}>
                {option.label}
              </option>
            ))}
          </select>
          <button type="submit" disabled={!canRequest || loading}>
            {loading ? "Building…" : "Show"}
          </button>
        </form>
        <div className="workspace-actions">
          <button
            type="button"
            className="icon-btn"
            onClick={() => setLayout(layout === "tabs" ? "grid" : "tabs")}
            aria-pressed={layout === "grid"}
            title={layout === "tabs" ? "Show all charts in a grid" : "Show one chart at a time"}
          >
            {layout === "tabs" ? "▦" : "▭"}
          </button>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close chart panel">
            ✕
          </button>
        </div>
      </header>

      {error && (
        <p className="workspace-error" role="alert">
          {error}
        </p>
      )}

      {charts.length === 0 ? (
        <div className="workspace-empty">
          <p>No charts yet.</p>
          <p className="muted">
            {canRequest
              ? "Pick a chart type above, or ask the agent to visualize the results."
              : "Run an analysis first — charts appear here automatically."}
          </p>
        </div>
      ) : layout === "tabs" && current ? (
        <>
          <div className="chart-tabs" role="tablist" aria-label="Charts">
            {charts.map((chart, index) => (
              <button
                key={chart.name}
                role="tab"
                type="button"
                aria-selected={index === active}
                className={index === active ? "tab active" : "tab"}
                onClick={() => setActive(index)}
              >
                {chart.title}
              </button>
            ))}
          </div>
          <div className="chart-card" role="tabpanel">
            <ChartToolbar
              onFullscreen={() => setFullscreen(current)}
              onDownload={(format) => void handle.current?.download(format)}
            />
            <PlotlyChart key={current.name} chart={current} theme={theme} handleRef={handle} />
          </div>
        </>
      ) : (
        <div className="chart-grid">
          {charts.map((chart) => (
            <GridChart key={chart.name} chart={chart} theme={theme} onFullscreen={setFullscreen} />
          ))}
        </div>
      )}

      {fullscreen && (
        <ChartModal chart={fullscreen} theme={theme} onClose={() => setFullscreen(null)} />
      )}
    </aside>
  );
}

function GridChart({
  chart,
  theme,
  onFullscreen,
}: {
  chart: ChartSpec;
  theme: Theme;
  onFullscreen: (chart: ChartSpec) => void;
}) {
  const handle = useRef<PlotlyChartHandle>(null);
  return (
    <div className="chart-card">
      <h3>{chart.title}</h3>
      <ChartToolbar
        onFullscreen={() => onFullscreen(chart)}
        onDownload={(format) => void handle.current?.download(format)}
      />
      <PlotlyChart chart={chart} theme={theme} height={340} handleRef={handle} />
    </div>
  );
}

function ChartToolbar({
  onFullscreen,
  onDownload,
}: {
  onFullscreen: () => void;
  onDownload: (format: "png" | "svg") => void;
}) {
  return (
    <div className="chart-toolbar">
      <button type="button" onClick={() => onDownload("png")}>
        PNG
      </button>
      <button type="button" onClick={() => onDownload("svg")}>
        SVG
      </button>
      <button type="button" onClick={onFullscreen} aria-label="Open chart fullscreen">
        ⤢ Fullscreen
      </button>
    </div>
  );
}

function ChartModal({ chart, theme, onClose }: { chart: ChartSpec; theme: Theme; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const handle = useRef<PlotlyChartHandle>(null);

  useEffect(() => {
    const element = dialog.current;
    if (element && !element.open) element.showModal?.();
    return () => element?.close?.();
  }, []);

  return (
    <dialog ref={dialog} className="chart-modal" aria-label={chart.title} onClose={onClose}>
      <header>
        <h2>{chart.title}</h2>
        <ChartToolbar onFullscreen={onClose} onDownload={(format) => void handle.current?.download(format)} />
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Close fullscreen chart">
          ✕
        </button>
      </header>
      <PlotlyChart chart={chart} theme={theme} height={Math.round(window.innerHeight * 0.78)} handleRef={handle} />
    </dialog>
  );
}

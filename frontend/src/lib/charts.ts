import type { ChartSpec, ChartsEventData } from "./types";

/**
 * Charts to show after a `charts` SSE event. "cleared" drops charts of a
 * superseded analysis (new data, or a re-run that was not charted), so the
 * workspace never presents stale results as current.
 */
export function chartsAfterEvent(current: ChartSpec[], data: ChartsEventData): ChartSpec[] {
  switch (data.state) {
    case "updated":
      return data.charts;
    case "cleared":
      return [];
    case "unchanged":
      return current;
    default:
      // Older servers without `state`: only non-empty lists were meaningful.
      return data.charts.length > 0 ? data.charts : current;
  }
}

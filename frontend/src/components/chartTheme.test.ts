import { describe, expect, it } from "vitest";
import type { Layout } from "plotly.js";
import { chartHeight, isNeutralColor, themedLayout } from "./PlotlyChart";
import type { ChartSpec } from "../lib/types";

// Mirrors the server dashboard: pinned dark title, subplot titles as
// annotations, axis titles, plus one colored annotation that must survive.
const serverLayout = {
  title: { text: "Dashboard", font: { color: "#1f2937", size: 18 } },
  annotations: [
    { text: "Means", font: { color: "#374151", size: 12 } },
    { text: "significant", font: { color: "#16a34a" } },
  ],
  xaxis: { title: { text: "Segment", font: { color: "#374151" } } },
  yaxis2: { title: { text: "P-Value" } },
} as unknown as Partial<Layout>;

describe("themedLayout (dark)", () => {
  const themed = themedLayout(serverLayout, "dark") as unknown as Record<string, any>;

  it("recolors the pinned chart title", () => {
    expect(themed.title.font.color).toBe("#e6e8ee");
    expect(themed.title.font.size).toBe(18);
  });

  it("recolors neutral subplot titles but keeps colored annotations", () => {
    expect(themed.annotations[0].font.color).toBe("#e6e8ee");
    expect(themed.annotations[1].font.color).toBe("#16a34a");
  });

  it("recolors axis titles", () => {
    expect(themed.xaxis.title.font.color).toBe("#e6e8ee");
    expect(themed.yaxis2.title.font.color).toBe("#e6e8ee");
  });
});

describe("isNeutralColor", () => {
  it("treats greys as neutral and saturated colors as meaningful", () => {
    expect(isNeutralColor("#374151")).toBe(true);
    expect(isNeutralColor("rgb(40, 40, 40)")).toBe(true);
    expect(isNeutralColor("#16a34a")).toBe(false);
    expect(isNeutralColor("#dc2626")).toBe(false);
  });
});

describe("chartHeight", () => {
  const spec = (layout: object) => ({ name: "d", title: "D", figure: { data: [], layout } }) as unknown as ChartSpec;

  it("keeps the server figure's height so autosize resizes cannot collapse it", () => {
    expect(chartHeight(spec({ height: 880 }))).toBe(880);
  });

  it("prefers an explicit height and falls back to 420", () => {
    expect(chartHeight(spec({ height: 880 }), 340)).toBe(340);
    expect(chartHeight(spec({}))).toBe(420);
  });
});

import { describe, expect, it } from "vitest";
import { chartsAfterEvent } from "./charts";
import type { ChartSpec } from "./types";

const chart = (name: string) => ({ name, title: name, figure: { data: [], layout: {} } }) as unknown as ChartSpec;
const shown = [chart("dashboard")];

describe("chartsAfterEvent", () => {
  it("replaces charts on 'updated'", () => {
    expect(chartsAfterEvent(shown, { charts: [chart("p_values")], state: "updated" })).toEqual([chart("p_values")]);
  });

  it("drops charts of a replaced analysis on 'cleared'", () => {
    // PR #10 review: an empty charts event used to be ignored, leaving the old
    // dataset's charts on screen as if current.
    expect(chartsAfterEvent(shown, { charts: [], state: "cleared" })).toEqual([]);
  });

  it("keeps charts on 'unchanged'", () => {
    expect(chartsAfterEvent(shown, { charts: [], state: "unchanged" })).toBe(shown);
  });

  it("stays compatible with events that carry no state", () => {
    expect(chartsAfterEvent(shown, { charts: [] })).toBe(shown);
    expect(chartsAfterEvent(shown, { charts: [chart("x")] })).toEqual([chart("x")]);
  });
});

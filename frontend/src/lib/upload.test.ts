import { describe, expect, it } from "vitest";
import { formatBytes, validateCsvFile } from "./upload";

describe("validateCsvFile", () => {
  it("accepts a CSV under the limit (case-insensitive extension)", () => {
    expect(validateCsvFile({ name: "Results.CSV", size: 1024 }, 50)).toEqual({ ok: true });
  });

  it("rejects non-CSV files", () => {
    expect(validateCsvFile({ name: "data.xlsx", size: 10 }, 50)).toEqual({
      ok: false,
      reason: "Only .csv files can be uploaded.",
    });
  });

  it("rejects empty and oversized files", () => {
    expect(validateCsvFile({ name: "a.csv", size: 0 }, 50).ok).toBe(false);
    const tooBig = validateCsvFile({ name: "a.csv", size: 51 * 1024 * 1024 }, 50);
    expect(tooBig).toEqual({ ok: false, reason: "File exceeds the 50 MB upload limit." });
  });
});

describe("formatBytes", () => {
  it("formats bytes, kilobytes and megabytes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(3 * 1024 * 1024)).toBe("3.0 MB");
  });
});

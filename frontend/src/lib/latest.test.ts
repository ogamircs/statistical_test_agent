import { describe, expect, it } from "vitest";
import { LatestKey } from "./latest";

describe("LatestKey", () => {
  it("rejects a response whose session is no longer active", () => {
    const active = new LatestKey<string>();
    active.set("session-a");
    const requestedFor = "session-a";

    active.set("session-b"); // user switched while the request was in flight

    expect(active.isCurrent(requestedFor)).toBe(false);
    expect(active.isCurrent("session-b")).toBe(true);
  });

  it("accepts a response when the session did not change", () => {
    const active = new LatestKey<string>();
    active.set("session-a");
    expect(active.isCurrent("session-a")).toBe(true);
  });
});

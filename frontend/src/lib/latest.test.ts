import { describe, expect, it } from "vitest";
import { LatestKey, NavigationGuard } from "./latest";

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

describe("NavigationGuard", () => {
  it("lets only the newest navigation apply its response", () => {
    const nav = new NavigationGuard();
    const openA = nav.begin();
    const openB = nav.begin(); // user clicked B before A's history arrived

    expect(nav.isLatest(openB)).toBe(true);
    expect(nav.isLatest(openA)).toBe(false); // A's late response is dropped
  });

  it("a New analysis click supersedes an in-flight open", () => {
    const nav = new NavigationGuard();
    const openA = nav.begin();
    nav.begin(); // New analysis
    expect(nav.isLatest(openA)).toBe(false);
  });
});

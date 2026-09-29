import { afterEach, describe, expect, it, vi } from "vitest";
import { readToken, storeToken } from "./api";

describe("token storage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    storeToken(null);
  });

  it("keeps the token in memory when sessionStorage throws", () => {
    // PR #10 review: the token used to be discarded, so every request after
    // login returned 401 in browsers that block storage.
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("denied", "SecurityError");
    });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("denied", "SecurityError");
    });

    storeToken("abc.def");

    expect(readToken()).toBe("abc.def");
  });

  it("forgets the token on sign-out even without storage", () => {
    vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
      throw new DOMException("denied", "SecurityError");
    });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("denied", "SecurityError");
    });
    storeToken("abc.def");
    storeToken(null);

    expect(readToken()).toBeNull();
  });

  it("uses sessionStorage when it works", () => {
    storeToken("t1");
    expect(sessionStorage.getItem("statagent.token")).toBe("t1");
    expect(readToken()).toBe("t1");
  });
});

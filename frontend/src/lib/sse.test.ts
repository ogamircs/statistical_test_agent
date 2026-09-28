import { afterEach, describe, expect, it, vi } from "vitest";
import { SseParser } from "./sse";
import { api } from "./api";

describe("SseParser", () => {
  it("parses complete frames and buffers partial ones across chunks", () => {
    const parser = new SseParser();
    expect(parser.push('event: status\ndata: {"state":"sta')).toEqual([]);
    const events = parser.push('rted"}\n\nevent: done\ndata: {}\n\n');
    expect(events).toEqual([
      { event: "status", data: { state: "started" } },
      { event: "done", data: {} },
    ]);
  });

  it("handles CRLF line endings, comments and multi-line data", () => {
    const parser = new SseParser();
    const events = parser.push(': keep-alive\r\nevent: message\r\ndata: {"content":\r\ndata: "hi","error_code":null}\r\n\r\n');
    expect(events).toEqual([{ event: "message", data: { content: "hi", error_code: null } }]);
  });

  it("skips frames with invalid JSON instead of throwing", () => {
    const parser = new SseParser();
    expect(parser.push("event: message\ndata: {not json}\n\n")).toEqual([]);
  });
});

describe("api.chat", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("streams events from a chunked response body", async () => {
    const encoder = new TextEncoder();
    const chunks = [
      'event: tool_start\ndata: {"id":"1","name":"load_csv","label":"Loading data"}\n\nevent: tool_',
      'end\ndata: {"id":"1","name":"load_csv","ok":true}\n\n',
      'event: done\ndata: {}\n\n',
    ];
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    });
    const fetchMock = vi.fn().mockResolvedValue(new Response(body, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const received = [];
    for await (const event of api.chat("abc", "hello", "f1")) received.push(event.event);

    expect(received).toEqual(["tool_start", "tool_end", "done"]);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/sessions/abc/chat");
    expect(JSON.parse(init.body as string)).toEqual({ message: "hello", file_id: "f1" });
  });

  it("raises an ApiError with the server error code", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: { code: "SESSION_BUSY", message: "busy" } }), { status: 409 }),
      ),
    );
    const iterate = async () => {
      for await (const event of api.chat("abc", "x", null)) void event;
    };
    await expect(iterate()).rejects.toMatchObject({ status: 409, code: "SESSION_BUSY" });
  });
});

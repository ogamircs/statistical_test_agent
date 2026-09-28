import type { StreamEvent } from "./types";

/**
 * Incremental parser for a `text/event-stream` body.
 *
 * Feed it decoded text chunks as they arrive (frames may be split across
 * chunks); it returns every complete event. Only `event:` and `data:`
 * fields are used — the API always sends JSON on a single data line, but
 * multi-line data is joined per the SSE spec.
 */
export class SseParser {
  private buffer = "";

  push(chunk: string): StreamEvent[] {
    this.buffer += chunk.replace(/\r\n?/g, "\n");
    const events: StreamEvent[] = [];
    let boundary = this.buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      const parsed = parseFrame(frame);
      if (parsed) events.push(parsed);
      boundary = this.buffer.indexOf("\n\n");
    }
    return events;
  }
}

function parseFrame(frame: string): StreamEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of frame.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  if (data.length === 0) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) } as StreamEvent;
  } catch {
    return null;
  }
}

import { describe, expect, it } from "vitest";
import { TokenBuffer } from "./tokenBuffer";

function manualScheduler() {
  const queue = new Map<number, () => void>();
  let next = 1;
  return {
    schedule: (cb: () => void) => {
      queue.set(next, cb);
      return next++;
    },
    cancel: (handle: number) => queue.delete(handle),
    runFrame: () => {
      const callbacks = [...queue.values()];
      queue.clear();
      callbacks.forEach((cb) => cb());
    },
    pending: () => queue.size,
  };
}

describe("TokenBuffer", () => {
  it("batches many tokens into one flush per frame", () => {
    const frames = manualScheduler();
    const flushed: string[] = [];
    const buffer = new TokenBuffer((text) => flushed.push(text), frames.schedule, frames.cancel);

    buffer.append("Prem");
    buffer.append("ium ");
    buffer.append("won");
    expect(frames.pending()).toBe(1);
    frames.runFrame();
    buffer.append(".");
    frames.runFrame();

    expect(flushed).toEqual(["Premium won", "Premium won."]);
  });

  it("reset clears the preamble before a tool call", () => {
    const frames = manualScheduler();
    const flushed: string[] = [];
    const buffer = new TokenBuffer((text) => flushed.push(text), frames.schedule, frames.cancel);

    buffer.append("Let me check.");
    buffer.reset();
    frames.runFrame(); // the cancelled preamble flush never fires
    buffer.append("Answer");
    frames.runFrame();

    expect(flushed).toEqual(["", "Answer"]);
  });

  it("stop drops a pending flush", () => {
    const frames = manualScheduler();
    const flushed: string[] = [];
    const buffer = new TokenBuffer((text) => flushed.push(text), frames.schedule, frames.cancel);

    buffer.append("partial");
    buffer.stop();
    frames.runFrame();

    expect(flushed).toEqual([]);
  });
});

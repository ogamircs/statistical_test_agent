/**
 * Accumulates streamed answer text and flushes it at most once per frame,
 * so a fast token stream causes one React update per animation frame
 * instead of one per token.
 */
export class TokenBuffer {
  private text = "";
  private handle: number | null = null;

  constructor(
    private readonly onFlush: (text: string) => void,
    private readonly schedule: (callback: () => void) => number = (cb) => requestAnimationFrame(cb),
    private readonly cancel: (handle: number) => void = (handle) => cancelAnimationFrame(handle),
  ) {}

  append(chunk: string): void {
    this.text += chunk;
    if (this.handle === null) {
      this.handle = this.schedule(() => {
        this.handle = null;
        this.onFlush(this.text);
      });
    }
  }

  /**
   * Drop what has streamed so far. Text a model writes before calling a tool
   * is a preamble, not the answer, so the UI restarts on each tool call.
   */
  reset(): void {
    this.text = "";
    this.stop();
    this.onFlush("");
  }

  /** Stop without flushing (the final `message` event replaces the text). */
  stop(): void {
    if (this.handle !== null) {
      this.cancel(this.handle);
      this.handle = null;
    }
  }
}

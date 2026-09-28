/**
 * Tracks the key (e.g. session id) that async results may still apply to.
 * A slow response started for one session must not land in another one the
 * user switched to meanwhile.
 */
export class LatestKey<K> {
  private current: K | null = null;

  set(key: K | null): void {
    this.current = key;
  }

  /** True if a response requested for `key` may still be applied. */
  isCurrent(key: K): boolean {
    return this.current === key;
  }
}

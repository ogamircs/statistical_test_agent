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

/**
 * Orders user navigations. Each navigation takes a ticket; only the newest
 * ticket may apply its (async) result, so the view follows the user's last
 * click rather than whichever response arrives last.
 */
export class NavigationGuard {
  private latest = 0;

  begin(): number {
    this.latest += 1;
    return this.latest;
  }

  isLatest(ticket: number): boolean {
    return ticket === this.latest;
  }
}

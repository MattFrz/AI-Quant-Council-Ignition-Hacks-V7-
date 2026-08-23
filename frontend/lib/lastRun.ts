/**
 * The most recent research result, shared across tabs.
 *
 * Opportunities and Portfolio are views of a run that happened on the Research
 * tab, so they need its result after a client-side navigation. sessionStorage
 * rather than a context provider: it survives a refresh and a direct link to
 * /portfolio, which a context reset on mount would not.
 *
 * Session-scoped on purpose. A result is a snapshot of one run, and finding a
 * stale one waiting in a new tab tomorrow would be worse than finding nothing.
 */
import type { ResearchResponse } from "./types";

const KEY = "aqc:last-run";

export function saveLastRun(result: ResearchResponse): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(result));
  } catch {
    // Private browsing, or a result larger than the quota. Losing the
    // cross-tab view is not worth breaking the run that just succeeded.
  }
}

export function loadLastRun(): ResearchResponse | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as ResearchResponse) : null;
  } catch {
    return null;
  }
}

export function clearLastRun(): void {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    /* nothing to do */
  }
}

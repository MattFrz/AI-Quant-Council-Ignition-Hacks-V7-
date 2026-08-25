/**
 * Display formatting. Kept in one file so a number never renders two ways in
 * two places.
 *
 * Every function tolerates null, because most numeric fields on the contract
 * are nullable and a degraded run legitimately returns nothing for a stage.
 * Rendering "-" is correct there; rendering "0" would be a lie.
 */

export const DASH = "-";

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  return `${(value * 100).toFixed(digits)}%`;
}

export function signedPct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(digits)}%`;
}

export function num(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  return value.toFixed(digits);
}

export function int(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  return value.toLocaleString("en-US");
}

/** Tone for a value where positive is good (returns, excess, alpha). */
export function toneForReturn(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "neutral";
  return value >= 0 ? "positive" : "negative";
}

export function riskTone(band: string | null | undefined) {
  if (band === "low") return "positive";
  if (band === "medium") return "warning";
  if (band === "high") return "negative";
  return "neutral";
}

export function verdictTone(verdict: string | null | undefined) {
  if (verdict === "survived") return "positive";
  if (verdict === "rejected") return "negative";
  return "warning";
}

/** ISO date to something a human reads, without pulling in a date library. */
export function shortDate(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

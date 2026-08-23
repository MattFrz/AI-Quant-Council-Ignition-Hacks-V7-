import type { Catalyst } from "../../lib/types";
import { pct, shortDate } from "../../lib/format";
import { SourceLink } from "./SourceLink";

const DIRECTION_TONE: Record<string, string> = {
  bullish: "positive",
  bearish: "negative",
  neutral: "neutral",
};

/**
 * One catalyst: what it claims, the verbatim sentence it came from, and the
 * link to the filing.
 *
 * The quote renders inside real quotation marks and in italics because it is
 * someone else's words, not ours. Paraphrasing it in our own voice is exactly
 * the failure this whole audit trail exists to prevent.
 */
export function CatalystCard({ catalyst }: { catalyst: Catalyst }) {
  return (
    <article className="catalyst">
      <div className="catalyst-head">
        <span className="badge" data-tone={DIRECTION_TONE[catalyst.direction] ?? "neutral"}>
          {catalyst.direction}
        </span>
        <h3 className="catalyst-headline">{catalyst.headline}</h3>
      </div>

      {catalyst.quote ? <blockquote className="catalyst-quote">“{catalyst.quote}”</blockquote> : null}

      <div className="catalyst-meta">
        <span className="mono">{catalyst.ticker}</span>
        <SourceLink
          url={catalyst.source_url}
          sourceType={catalyst.source_type}
          sourceDate={catalyst.source_date}
        />
        {catalyst.event_date ? <span>event {shortDate(catalyst.event_date)}</span> : null}
        <span>confidence {pct(catalyst.confidence, 0)}</span>
      </div>
    </article>
  );
}

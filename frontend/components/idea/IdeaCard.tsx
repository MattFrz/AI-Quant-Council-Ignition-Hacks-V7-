import type { TradeIdea } from "../../lib/types";
import { num, pct, signedPct, verdictTone } from "../../lib/format";

/**
 * The headline verdict: what it picked, how strongly, and whether the
 * quantitative validator let it through.
 *
 * alpha_score is a percentile rank across the scored universe, not a raw
 * z-score, so it is labelled as one. An unlabelled "9.4 / 10" invites the
 * reader to think it means something absolute about the company.
 */
export function IdeaCard({ idea, universeSize }: { idea: TradeIdea; universeSize?: number }) {
  const sideTone =
    idea.side === "LONG" ? "positive" : idea.side === "SHORT" ? "negative" : "neutral";

  return (
    <section className="card" aria-label="Recommendation">
      <div className="idea-head">
        <div>
          <div className="idea-ticker">{idea.ticker}</div>
          <div className="idea-company">
            {idea.company_name}
            {idea.risk?.sector ? ` · ${idea.risk.sector}` : ""}
          </div>
        </div>
        <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
          <span className="badge" data-tone={sideTone}>
            {idea.side}
          </span>
          <span className="badge" data-tone={verdictTone(idea.validator_verdict)}>
            validator: {idea.validator_verdict}
          </span>
        </div>
      </div>

      <div className="metrics">
        <div>
          <div className="metric-label">Alpha score</div>
          <div className="metric-value">{num(idea.alpha_score, 1)} / 10</div>
          <div className="metric-note">
            percentile rank{universeSize ? ` across ${universeSize} scored names` : ""}
          </div>
        </div>
        <div>
          <div className="metric-label">Confidence</div>
          <div className="metric-value">{pct(idea.confidence, 0)}</div>
          <div className="metric-note">verdict + evidence + quant</div>
        </div>
        <div>
          <div className="metric-label">Expected alpha</div>
          <div className="metric-value">{signedPct(idea.expected_alpha)}</div>
          <div className="metric-note">annualised, vs benchmark</div>
        </div>
        <div>
          <div className="metric-label">Position size</div>
          <div className="metric-value">{pct((idea.position_size_pct ?? 0) / 100, 1)}</div>
          <div className="metric-note">of book, before risk limits</div>
        </div>
      </div>
    </section>
  );
}

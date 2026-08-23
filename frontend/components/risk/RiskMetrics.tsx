import type { RiskMetrics as RiskMetricsType } from "../../lib/types";
import { num, pct, riskTone } from "../../lib/format";

/**
 * Single-name risk for the recommended stock.
 *
 * These are deliberately NOT the strategy's numbers. A card about one stock
 * showing portfolio-level beta reads as though the stock is tame when it is
 * not - the earlier version of this panel showed beta 0.64 next to a name
 * whose real beta was 1.76. The strategy's own risk lives in the quant
 * section, with its own heading.
 */
export function RiskPanel({ risk }: { risk: RiskMetricsType | null }) {
  if (!risk) {
    return (
      <section className="card" aria-label="Risk">
        <div className="card-title">
          <h2>Single-name risk</h2>
        </div>
        <p className="empty">Risk stage did not complete for this run.</p>
      </section>
    );
  }

  return (
    <section className="card" aria-label="Risk">
      <div className="card-title">
        <h2>Single-name risk</h2>
        <span className="badge" data-tone={riskTone(risk.risk_band)}>
          {risk.risk_band} risk
        </span>
      </div>

      <div className="metrics">
        <div>
          <div className="metric-label">Beta</div>
          <div className="metric-value">{num(risk.beta)}</div>
          <div className="metric-note">vs benchmark</div>
        </div>
        <div>
          <div className="metric-label">Volatility</div>
          <div className="metric-value">{pct(risk.volatility)}</div>
          <div className="metric-note">annualised</div>
        </div>
        <div>
          <div className="metric-label">Max drawdown</div>
          <div className="metric-value text-negative">{pct(risk.max_drawdown)}</div>
          <div className="metric-note">the stock, not the book</div>
        </div>
        <div>
          <div className="metric-label">VaR 95</div>
          <div className="metric-value">{pct(risk.var_95)}</div>
          <div className="metric-note">daily</div>
        </div>
        <div>
          <div className="metric-label">CVaR 95</div>
          <div className="metric-value">{pct(risk.cvar_95)}</div>
          <div className="metric-note">tail average</div>
        </div>
        <div>
          <div className="metric-label">Sector</div>
          <div className="metric-value" style={{ fontSize: "var(--text-base)" }}>
            {risk.sector ?? "–"}
          </div>
        </div>
        <div>
          <div className="metric-label">Concentration</div>
          <div className="metric-value">{pct(risk.concentration)}</div>
          <div className="metric-note">largest single weight</div>
        </div>
        <div>
          <div className="metric-label">Avg correlation</div>
          <div className="metric-value">{num(risk.avg_correlation)}</div>
          <div className="metric-note">to the rest of the book</div>
        </div>
        <div>
          <div className="metric-label">Days to liquidate</div>
          <div className="metric-value">{num(risk.days_to_liquidate, 3)}</div>
          <div className="metric-note">at the ADV cap</div>
        </div>
      </div>
    </section>
  );
}

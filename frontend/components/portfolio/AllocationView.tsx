import type { PortfolioResponse } from "../../lib/api";
import { pct } from "../../lib/format";

/**
 * Invested against cash, as one bar.
 *
 * A book that is 0% invested is the headline when it happens, so cash is drawn
 * rather than left as the absence of positions. "Nothing here" and "we looked
 * and declined" render identically otherwise.
 */
export function AllocationView({ portfolio }: { portfolio: PortfolioResponse }) {
  const invested = portfolio.total_invested_pct ?? 0;
  const cash = portfolio.cash_pct ?? 100 - invested;
  const fullyCash = invested <= 0;

  return (
    <section className="card" aria-label="Allocation">
      <div className="card-title">
        <h2>Allocation</h2>
        <span className="count">
          {portfolio.positions.length} position
          {portfolio.positions.length === 1 ? "" : "s"} ·{" "}
          {portfolio.excluded.length} excluded
        </span>
      </div>

      <div className="alloc-bar" role="img" aria-label={`${pct(invested / 100)} invested`}>
        <div className="alloc-invested" style={{ width: `${Math.min(invested, 100)}%` }} />
      </div>

      <div className="metrics">
        <div>
          <div className="metric-label">Invested</div>
          <div className="metric-value">{pct(invested / 100)}</div>
        </div>
        <div>
          <div className="metric-label">Cash</div>
          <div className="metric-value">{pct(cash / 100)}</div>
        </div>
        <div>
          <div className="metric-label">Funded</div>
          <div className="metric-value">{portfolio.positions.length}</div>
        </div>
        <div>
          <div className="metric-label">Refused</div>
          <div className="metric-value">{portfolio.excluded.length}</div>
        </div>
      </div>

      {fullyCash ? (
        <div className="notice" data-tone="info" style={{ marginTop: "var(--space-4)" }}>
          Fully in cash. Every candidate cleared research and the backtest, then
          failed a risk constraint below. The system is permitted to decline its
          own recommendation.
        </div>
      ) : null}
    </section>
  );
}

import type { PortfolioResponse } from "../../lib/api";
import { num, pct, riskTone } from "../../lib/format";

export function PositionTable({ portfolio }: { portfolio: PortfolioResponse }) {
  return (
    <>
      <section className="card" aria-label="Positions">
        <div className="card-title">
          <h2>Positions</h2>
        </div>
        {portfolio.positions.length === 0 ? (
          <p className="empty">No position was funded on this run.</p>
        ) : (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Company</th>
                  <th>Side</th>
                  <th className="num">Size</th>
                  <th className="num">Alpha</th>
                  <th className="num">Confidence</th>
                  <th>Risk</th>
                </tr>
              </thead>
              <tbody>
                {portfolio.positions.map((p) => (
                  <tr key={p.ticker}>
                    <td className="mono strong">{p.ticker}</td>
                    <td>{p.company_name}</td>
                    <td className="mono">{p.side}</td>
                    <td className="num mono">{pct(p.position_size_pct / 100)}</td>
                    <td className="num mono">{num(p.alpha_score, 1)}</td>
                    <td className="num mono">{pct(p.confidence, 0)}</td>
                    <td>
                      <span className="badge" data-tone={riskTone(p.risk_band)}>
                        {p.risk_band}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* The exclusions carry the reasoning, so they get a full section rather
          than a line under the table. Each reason is the constraint that
          actually fired, quoted from the risk layer. */}
      <section className="card" aria-label="Excluded candidates">
        <div className="card-title">
          <h2>Refused, and why</h2>
          <span className="count">{portfolio.excluded.length}</span>
        </div>
        {portfolio.excluded.length === 0 ? (
          <p className="empty">Nothing was excluded on this run.</p>
        ) : (
          <ul className="exclusions">
            {portfolio.excluded.map((e) => (
              <li key={e.ticker}>
                <span className="mono strong">{e.ticker}</span>
                <ul>
                  {e.reasons.map((r) => (
                    <li key={r} className="text-secondary">
                      {r}
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}

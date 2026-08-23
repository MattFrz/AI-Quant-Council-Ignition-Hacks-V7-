import type { BacktestResult } from "../../lib/types";
import { int, num, pct, signedPct } from "../../lib/format";
import { EquityCurve } from "../../charts/EquityCurve";
import { DrawdownChart } from "../../charts/DrawdownChart";

/**
 * The strategy's own record: what the backtest did, after costs.
 *
 * Two things are stated rather than left implicit. Costs are named, because a
 * backtest that does not say what it charged is not a result. And
 * `is_walk_forward` is surfaced directly - an in-sample number presented
 * without that label is the single easiest way for this whole page to be
 * dismissed by someone who knows what to ask.
 */
export function QuantPanel({ backtest }: { backtest: BacktestResult | null }) {
  if (!backtest) {
    return (
      <section className="card" aria-label="Quantitative validation">
        <div className="card-title">
          <h2>Quantitative validation</h2>
        </div>
        <p className="empty">Backtest stage did not complete for this run.</p>
      </section>
    );
  }

  return (
    <section className="card" aria-label="Quantitative validation">
      <div className="card-title">
        <h2>Quantitative validation</h2>
        <span className="count">
          {backtest.strategy_name} · {int(backtest.universe_size)} names
        </span>
      </div>

      <EquityCurve
        equity={backtest.equity_curve ?? []}
        benchmark={backtest.benchmark_curve ?? []}
      />

      <h3 className="subhead">Drawdown</h3>
      <DrawdownChart drawdown={backtest.drawdown_curve ?? []} />

      <div className="metrics">
        <div>
          <div className="metric-label">Annualised</div>
          <div className="metric-value">{pct(backtest.annualized_return)}</div>
          <div className="metric-note">
            SPY {pct(backtest.benchmark_annualized_return)}
          </div>
        </div>
        <div>
          <div className="metric-label">Excess</div>
          <div className="metric-value text-positive">{signedPct(backtest.excess_return)}</div>
          <div className="metric-note">over benchmark</div>
        </div>
        <div>
          <div className="metric-label">Sharpe</div>
          <div className="metric-value">{num(backtest.sharpe)}</div>
          <div className="metric-note">Sortino {num(backtest.sortino)}</div>
        </div>
        <div>
          <div className="metric-label">Max drawdown</div>
          <div className="metric-value text-negative">{pct(backtest.max_drawdown)}</div>
          <div className="metric-note">vol {pct(backtest.volatility)}</div>
        </div>
        <div>
          <div className="metric-label">Trades</div>
          <div className="metric-value">{int(backtest.n_trades)}</div>
          <div className="metric-note">
            turnover {num(backtest.turnover)}x · win {pct(backtest.win_rate, 0)}
          </div>
        </div>
        <div>
          <div className="metric-label">Costs charged</div>
          <div className="metric-value" style={{ fontSize: "var(--text-base)" }}>
            {num(backtest.commission_bps, 0)}bp
          </div>
          <div className="metric-note">
            + {backtest.slippage_model.replace(/_/g, " ")} slippage
          </div>
        </div>
        <div>
          <div className="metric-label">Total return</div>
          <div className="metric-value">{pct(backtest.total_return)}</div>
          <div className="metric-note">whole window</div>
        </div>
        <div>
          <div className="metric-label">ADV cap</div>
          <div className="metric-value">{pct(backtest.max_adv_participation, 0)}</div>
          <div className="metric-note">max participation</div>
        </div>
        {backtest.window ? (
          <div>
            <div className="metric-label">Window</div>
            <div className="metric-value" style={{ fontSize: "var(--text-base)" }}>
              {String(backtest.window.test_start).slice(0, 7)} to{" "}
              {String(backtest.window.test_end).slice(0, 7)}
            </div>
            <div className="metric-note">test period</div>
          </div>
        ) : null}
      </div>

      <p
        className="metric-note"
        style={{ marginTop: "var(--space-4)", fontFamily: "var(--font-mono)" }}
      >
        {backtest.is_walk_forward
          ? "Walk-forward: tested out of sample."
          : "Single window: this is an in-sample fit, not a walk-forward result."}
        {backtest.notes ? ` ${backtest.notes}` : ""}
      </p>
    </section>
  );
}

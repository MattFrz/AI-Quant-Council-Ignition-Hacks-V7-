"use client";

import { useCallback, useEffect, useState } from "react";
import { simulateExecution } from "../../lib/api";
import type { SimulationResponse } from "../../lib/api";
import { BookLadder } from "../../components/execution/BookLadder";
import { int, num, pct } from "../../lib/format";

/**
 * The C++ execution layer, running.
 *
 * This page exists because "we wrote an order book in C++" is a claim, and a
 * fill that gets worse as the order gets bigger is evidence. Every number below
 * comes back from the native extension walking a real price-time-priority book.
 */
const MODE_LABEL: Record<string, string> = {
  market: "Market order",
  sliced: "Sliced order",
  limit: "Passive limit",
};

export default function ExecutionPage() {
  const [shares, setShares] = useState(2500);
  const [slices, setSlices] = useState(4);
  const [side, setSide] = useState<"BUY" | "SELL">("BUY");
  const [data, setData] = useState<SimulationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await simulateExecution({ shares, slices, side }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [shares, slices, side]);

  useEffect(() => {
    void run();
    // Intentionally on mount only: the controls call run() themselves, and
    // re-firing on every keystroke would hammer the endpoint mid-typing.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const depth = data ? data.asks.reduce((a, l) => a + l.shares, 0) : 0;

  return (
    <main className="page">
      <div className="container">
        <header className="masthead">
          <h1>Execution</h1>
          <p className="lede">
            A limit order book, an ITCH replay parser and an execution simulator,
            written in C++ and called through pybind11. Size an order and watch
            what it actually costs to trade.
          </p>
        </header>

        <section className="card">
          <div className="card-title">
            <h2>Order</h2>
            <span className="count">{data?.engine ?? ""}</span>
          </div>

          <div className="controls">
            <label>
              <span className="metric-label">Shares</span>
              <input
                type="range"
                min={100}
                max={12000}
                step={100}
                value={shares}
                onChange={(e) => setShares(Number(e.target.value))}
                onMouseUp={run}
                onTouchEnd={run}
                onKeyUp={run}
              />
              <span className="metric-value">{int(shares)}</span>
            </label>

            <label>
              <span className="metric-label">Slices</span>
              <input
                type="range"
                min={1}
                max={20}
                value={slices}
                onChange={(e) => setSlices(Number(e.target.value))}
                onMouseUp={run}
                onTouchEnd={run}
                onKeyUp={run}
              />
              <span className="metric-value">{slices}</span>
            </label>

            <div className="side-toggle">
              <span className="metric-label">Side</span>
              <div>
                {(["BUY", "SELL"] as const).map((s) => (
                  <button
                    key={s}
                    type="button"
                    className="example-chip"
                    data-active={side === s || undefined}
                    onClick={() => {
                      setSide(s);
                      setTimeout(run, 0);
                    }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </section>

        {error ? (
          <div className="notice" data-tone="error" role="alert">
            {error}
          </div>
        ) : null}

        {data ? (
          <div className="exec-grid">
            <section className="card">
              <div className="card-title">
                <h2>Order book</h2>
                <span className="count">{int(depth)} shares offered</span>
              </div>
              <BookLadder
                bids={data.bids}
                asks={data.asks}
                mid={data.mid}
                spreadBps={data.spread_bps}
              />
              <div className="metrics">
                <div>
                  <div className="metric-label">Capacity</div>
                  <div className="metric-value">{int(data.capacity_5bps)}</div>
                  <div className="metric-note">shares within 5 bps</div>
                </div>
              </div>
            </section>

            <section className="card">
              <div className="card-title">
                <h2>What it costs to trade</h2>
                <span className="count">{loading ? "running…" : "C++ engine"}</span>
              </div>

              <ul className="outcomes">
                {data.outcomes.map((o, i) => (
                  <li className="outcome" key={`${o.mode}-${i}`}>
                    <div className="outcome-head">
                      <span className="outcome-mode">
                        {MODE_LABEL[o.mode] ?? o.mode}
                      </span>
                      <span
                        className="badge"
                        data-tone={
                          o.filled === 0
                            ? "negative"
                            : o.complete
                              ? "positive"
                              : "warning"
                        }
                      >
                        {o.filled === 0
                          ? "no fill"
                          : o.complete
                            ? "filled"
                            : "partial"}
                      </span>
                    </div>

                    <div className="outcome-nums">
                      <span>
                        {/* "Shares", not "Filled": the badge above already
                            says whether it filled, and two labels saying
                            "filled" above "0 / 2,500" reads as a contradiction. */}
                        <span className="metric-label">Shares</span>
                        <span className="mono">
                          {int(o.filled)} / {int(o.requested)}
                        </span>
                      </span>
                      <span>
                        <span className="metric-label">Avg price</span>
                        <span className="mono">{num(o.avg_price, 4)}</span>
                      </span>
                      <span>
                        <span className="metric-label">Slippage</span>
                        <span className="mono">{num(o.slippage_bps, 2)} bps</span>
                      </span>
                      <span>
                        <span className="metric-label">Levels</span>
                        <span className="mono">{o.levels_consumed}</span>
                      </span>
                    </div>

                    <p className="outcome-note">{o.note}</p>
                  </li>
                ))}
              </ul>
            </section>
          </div>
        ) : null}

        <section className="card">
          <div className="card-title">
            <h2>Why this is not in the backtest</h2>
          </div>
          <p className="rationale">
            The backtest charges slippage with a participation-rate model in
            Python. This engine is more faithful, and it is built and tested, but
            it is not wired into the backtest path. Saying it was would mean the
            performance figures elsewhere on this site came from something they
            did not come from.
          </p>
        </section>
      </div>
    </main>
  );
}

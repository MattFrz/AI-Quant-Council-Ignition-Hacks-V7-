"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { buildPortfolio } from "../../lib/api";
import type { PortfolioResponse } from "../../lib/api";
import type { ResearchResponse } from "../../lib/types";
import { loadLastRun } from "../../lib/lastRun";
import { Nav } from "../../components/nav/Nav";
import { PositionTable } from "../../components/portfolio/PositionTable";
import { AllocationView } from "../../components/portfolio/AllocationView";

/**
 * Sizing the book, and refusing to.
 *
 * The exclusions are the point of this page, not a footnote to it. A name can
 * clear research and survive the backtest and still be turned down here on risk
 * grounds, and without that reason stated an empty book just looks broken.
 */
export default function PortfolioPage() {
  const [run, setRun] = useState<ResearchResponse | null>(null);
  const [portfolio, setPortfolio] = useState<PortfolioResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setRun(loadLastRun());
    setLoaded(true);
  }, []);

  const build = useCallback(
    async (result: ResearchResponse) => {
      const candidates = [result.top_idea, ...(result.runners_up ?? [])].filter(
        (i): i is NonNullable<typeof i> => Boolean(i),
      );
      if (!candidates.length) return;

      setLoading(true);
      setError(null);
      try {
        setPortfolio(await buildPortfolio(candidates));
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (run) void build(run);
  }, [run, build]);

  return (
    <main className="page">
      <div className="container">
        <header className="masthead">
          <Nav />
          <h1>Portfolio</h1>
          <p className="lede">
            What the risk layer will actually fund, and what it turned down.
          </p>
        </header>

        {!loaded ? null : !run ? (
          <section className="card">
            <p className="empty">
              No run in this session yet.{" "}
              <Link href="/">Run a thesis on the Research tab</Link> and the sized
              book will appear here.
            </p>
          </section>
        ) : (
          <>
            {error ? (
              <div className="notice" data-tone="error" role="alert">
                {error}
              </div>
            ) : null}

            {loading ? (
              <section className="card">
                <p className="empty">Sizing the book…</p>
              </section>
            ) : portfolio ? (
              <>
                <AllocationView portfolio={portfolio} />
                <PositionTable portfolio={portfolio} />
              </>
            ) : null}
          </>
        )}
      </div>
    </main>
  );
}

"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import type { ResearchResponse } from "../../lib/types";
import { loadLastRun } from "../../lib/lastRun";
import { UniverseFunnel } from "../../components/scanner/UniverseFunnel";
import { CandidateTable } from "../../components/scanner/CandidateTable";
import { Nav } from "../../components/nav/Nav";
import { int, num, pct } from "../../lib/format";

/**
 * How the universe narrowed, and what was left standing.
 *
 * Reads the last run rather than triggering its own, so the numbers here are
 * always the same ones the Research tab showed. A scan launched from this page
 * could disagree with the recommendation next door, which is the kind of
 * inconsistency that makes a reader distrust both.
 */
export default function OpportunityPage() {
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setResult(loadLastRun());
    setLoaded(true);
  }, []);

  const scan = result?.scan ?? null;
  const idea = result?.top_idea ?? null;
  const runners = result?.runners_up ?? [];

  return (
    <main className="page">
      <div className="container">
        <header className="masthead">
          <Nav />
          <h1>Opportunities</h1>
          <p className="lede">
            Every name the model considered, and the filters that removed them.
          </p>
        </header>

        {!loaded ? null : !result ? (
          <section className="card">
            <p className="empty">
              No run in this session yet.{" "}
              <Link href="/">Run a thesis on the Research tab</Link> and the scan
              will appear here.
            </p>
          </section>
        ) : (
          <>
            <section className="card">
              <div className="card-title">
                <h2>Thesis</h2>
                <span className="count">as of {result.as_of}</span>
              </div>
              <p className="rationale">{result.thesis}</p>
            </section>

            <section className="card">
              <div className="card-title">
                <h2>Universe funnel</h2>
                <span className="count">
                  {scan?.funnel?.length ? `${scan.funnel.length} stages` : ""}
                </span>
              </div>
              <UniverseFunnel funnel={scan?.funnel ?? []} />
            </section>

            <section className="card">
              <div className="card-title">
                <h2>Ranked candidates</h2>
                <span className="count">
                  {int(scan?.candidates?.length ?? 0)} researched
                </span>
              </div>
              <CandidateTable candidates={scan?.candidates ?? []} />
            </section>

            {scan?.criteria?.length ? (
              <section className="card">
                <div className="card-title">
                  <h2>Screening criteria</h2>
                </div>
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Criterion</th>
                        <th>Metric</th>
                        <th>Test</th>
                        <th>Why</th>
                      </tr>
                    </thead>
                    <tbody>
                      {scan.criteria.map((c) => (
                        <tr key={`${c.metric}-${c.value}`}>
                          <td>{c.label}</td>
                          <td className="mono text-secondary">{c.metric}</td>
                          <td className="mono">
                            {c.operator} {c.value}
                          </td>
                          <td className="text-secondary">{c.rationale ?? "–"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            ) : null}

            <section className="card">
              <div className="card-title">
                <h2>Survived validation</h2>
                <span className="count">{runners.length + (idea ? 1 : 0)} ideas</span>
              </div>
              {idea ? (
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Company</th>
                        <th>Side</th>
                        <th className="num">Alpha</th>
                        <th className="num">Confidence</th>
                        <th>Verdict</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[idea, ...runners].map((i) => (
                        <tr key={i.idea_id}>
                          <td className="mono strong">{i.ticker}</td>
                          <td>{i.company_name}</td>
                          <td className="mono">{i.side}</td>
                          <td className="num mono">{num(i.alpha_score, 1)}</td>
                          <td className="num mono">{pct(i.confidence, 0)}</td>
                          <td className="text-secondary">{i.validator_verdict}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="empty">No idea survived validation on this run.</p>
              )}
            </section>
          </>
        )}
      </div>
    </main>
  );
}

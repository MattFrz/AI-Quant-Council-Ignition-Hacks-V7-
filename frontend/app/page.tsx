"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { submitThesis, fetchResearchResult } from "../lib/api";
import { subscribeToResearchEvents } from "../lib/stream";
import type { ResearchEvent, ResearchResponse } from "../lib/types";
import type { ResearchStreamHandle } from "../lib/stream";
import { Timeline } from "../components/timeline/Timeline";
import { IdeaCard } from "../components/idea/IdeaCard";
import { AuditTrail } from "../components/audit/AuditTrail";
import { DebatePanel } from "../components/debate/DebatePanel";
import { QuantPanel } from "../components/quant/QuantPanel";
import { RiskPanel } from "../components/risk/RiskMetrics";
import { int } from "../lib/format";
import { saveLastRun } from "../lib/lastRun";
import { Nav } from "../components/nav/Nav";

/**
 * Theses with a warmed cache entry behind them.
 *
 * Two reasons these are offered rather than left to free text. A cached thesis
 * replays in under a second, so a first-time visitor sees the whole pipeline
 * immediately instead of waiting seventy seconds. And on a public deployment
 * every uncached run bills a real LLM call, so steering people here is what
 * keeps the demo from costing money per visitor.
 */
const EXAMPLES: { label: string; thesis: string }[] = [
  {
    label: "AI data-center buildout",
    thesis:
      "Find companies benefiting from accelerating AI data-center spending that the market may be underpricing.",
  },
  ...["AMD", "DELL", "KLAC", "NVDA", "META"].map((t) => ({
    label: t,
    thesis: `Analyze ${t} and whether the market is underpricing its exposure to accelerating AI data-center spending.`,
  })),
];

type Phase = "idle" | "running" | "done" | "error";

/**
 * How many names the alpha model actually scored.
 *
 * NOT the first funnel row - that is everything with price data (504), which
 * is a larger number than the one the percentile was computed against and so
 * overstates the rank. The scored universe is the last stage before the
 * retrieval corpus narrows things to companies we hold filings for, found by
 * position rather than by matching a label string.
 */
function scoredUniverseSize(result: ResearchResponse | null): number | undefined {
  const funnel = result?.scan?.funnel;
  if (!funnel?.length) return undefined;
  const cut = funnel.findIndex((f) => /filings/i.test(f.label));
  const stage = cut > 0 ? funnel[cut - 1] : funnel[funnel.length - 1];
  return stage?.count;
}

export default function Home() {
  const [thesis, setThesis] = useState(EXAMPLES[0].thesis);
  const [phase, setPhase] = useState<Phase>("idle");
  const [events, setEvents] = useState<Record<string, ResearchEvent>>({});
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<ResearchStreamHandle | null>(null);

  // A stream left open after the component goes away keeps an EventSource
  // reconnecting against a backend nobody is watching any more.
  useEffect(() => () => streamRef.current?.close(), []);

  const run = useCallback(async () => {
    if (!thesis.trim() || phase === "running") return;

    streamRef.current?.close();
    setPhase("running");
    setEvents({});
    setResult(null);
    setError(null);

    try {
      const job = await submitThesis({ thesis: thesis.trim() });

      streamRef.current = subscribeToResearchEvents(
        job.stream_url,
        (event) => {
          // Keyed by step so a step that reports running then done replaces
          // itself rather than appearing twice.
          setEvents((prev) => ({ ...prev, [event.step_id]: event }));
        },
        async () => {
          try {
            const finished = await fetchResearchResult(job.job_id);
            setResult(finished);
            saveLastRun(finished); // Opportunities and Portfolio read this
            setPhase("done");
          } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
            setPhase("error");
          }
        },
        () => {
          // The stream dropped. The job itself may well have finished, so try
          // the result endpoint before calling the whole run a failure.
          fetchResearchResult(job.job_id)
            .then((r) => {
              setResult(r);
              saveLastRun(r);
              setPhase("done");
            })
            .catch(() => {
              setError("Lost the connection to the research stream.");
              setPhase("error");
            });
        },
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPhase("error");
    }
  }, [thesis, phase]);

  const idea = result?.top_idea ?? null;
  const universeSize = scoredUniverseSize(result);

  return (
    <main className="page">
      <div className="container">
        <header className="masthead">
          <Nav />
          <span className="eyebrow">
            <i className="live-dot" data-state={phase === "running" ? "running" : phase === "idle" ? "idle" : "done"} />
            {phase === "running"
              ? "Research running"
              : phase === "done"
                ? "Run complete"
                : "No run yet · submit a thesis to begin"}
          </span>
          <h1>AI Quant Council</h1>
          <p className="lede">
            Turn an investment thesis into evidence, debate, quantitative validation,
            and an auditable decision. The language model reads filings and argues
            both sides; every number comes from a separate quantitative engine that
            is allowed to reject its idea.
          </p>
        </header>

        <section className="card thesis-form">
          <div className="card-title">
            <h2>Investment thesis</h2>
          </div>
          <label className="sr-only" htmlFor="thesis">
            Investment thesis
          </label>
          <textarea
            id="thesis"
            value={thesis}
            onChange={(e) => setThesis(e.target.value)}
            placeholder="Describe your investment thesis..."
            spellCheck={false}
          />
          <div className="form-row">
            <div className="examples">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.label}
                  type="button"
                  className="example-chip"
                  onClick={() => setThesis(ex.thesis)}
                  disabled={phase === "running"}
                >
                  {ex.label}
                </button>
              ))}
            </div>
            <button
              type="button"
              className="btn-run"
              onClick={run}
              disabled={phase === "running" || !thesis.trim()}
            >
              {phase === "running" ? "Researching…" : "▶ Run autonomous research"}
            </button>
          </div>
        </section>

        {error ? (
          <div className="notice" data-tone="error" style={{ marginTop: "var(--space-5)" }} role="alert">
            {error}
          </div>
        ) : null}

        <div className="results-grid">
          <div className="rail">
            <Timeline events={events} running={phase === "running"} />
          </div>

          <div>
            {idea ? (
              <>
                <IdeaCard idea={idea} universeSize={universeSize} />
                <AuditTrail catalysts={idea.catalysts ?? []} />
                <DebatePanel bullCase={idea.bull_case} bearCase={idea.bear_case} />
                <QuantPanel backtest={idea.backtest} />
                <RiskPanel risk={idea.risk} />
                <section className="card" aria-label="Portfolio manager decision">
                  <div className="card-title">
                    <h2>Portfolio manager</h2>
                  </div>
                  <p className="rationale">{idea.pm_rationale}</p>
                </section>
              </>
            ) : (
              <section className="card">
                <div className="card-title">
                  <h2>Result</h2>
                </div>
                <p className="empty">
                  {phase === "running"
                    ? "Working through the pipeline…"
                    : phase === "done"
                      ? "This run produced no fundable idea."
                      : "Nothing yet. Pick an example above or write your own thesis."}
                </p>
              </section>
            )}
          </div>
        </div>

        <footer className="footer">
          <span>
            {result?.scan?.funnel?.length
              ? result.scan.funnel.map((f) => `${f.label} ${int(f.count)}`).join("  →  ")
              : "Research, evidence, debate and validation. Not investment advice."}
          </span>
          <span>{result?.as_of ? `as of ${result.as_of}` : null}</span>
        </footer>
      </div>
    </main>
  );
}

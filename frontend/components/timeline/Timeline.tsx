"use client";

import { PIPELINE_STEPS, STEP_LABELS } from "../../lib/types";
import type { ResearchEvent, StepStatus } from "../../lib/types";

/**
 * The 13-step live timeline.
 *
 * Renders every step up front in `pending` and upgrades them as events land,
 * rather than appending rows as they arrive. A list that grows makes a run
 * look open-ended; a list that fills in shows the reader how much is left.
 */
export function Timeline({
  events,
  running,
}: {
  events: Record<string, ResearchEvent>;
  running: boolean;
}) {
  const done = PIPELINE_STEPS.filter((s) => events[s]?.status === "done").length;

  return (
    <section className="card" aria-label="Research timeline">
      <div className="card-title">
        <h2>Council research timeline</h2>
        <span className="count">
          {done} / {PIPELINE_STEPS.length}
        </span>
      </div>

      <ol className="timeline">
        {PIPELINE_STEPS.map((stepId) => {
          const event = events[stepId];
          const status: StepStatus = event?.status ?? "pending";
          return (
            <li className="step" key={stepId} data-status={status}>
              <span className="step-marker" aria-hidden="true">
                {status === "done" ? "✓" : status === "failed" ? "!" : ""}
              </span>
              <div className="step-body">
                <div className="step-label">{event?.label ?? STEP_LABELS[stepId] ?? stepId}</div>
                {event?.detail ? <div className="step-detail">{event.detail}</div> : null}
              </div>
            </li>
          );
        })}
      </ol>

      {!running && done === 0 ? (
        <p className="empty" style={{ marginTop: "var(--space-4)" }}>
          Submit a thesis to begin.
        </p>
      ) : null}
    </section>
  );
}

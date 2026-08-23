import type { FunnelStage } from "../../lib/types";
import { int } from "../../lib/format";

/**
 * How a universe of hundreds becomes one recommendation.
 *
 * Bars are scaled against the FIRST stage, not the largest remaining, so the
 * collapse from 504 names to a handful is visible rather than flattened into
 * nine near-equal rows. The steep drop is the honest shape of the process.
 */
export function UniverseFunnel({ funnel }: { funnel: FunnelStage[] }) {
  if (!funnel?.length) {
    return <p className="empty">No funnel recorded for this run.</p>;
  }

  const top = Math.max(...funnel.map((f) => f.count)) || 1;

  return (
    <ol className="funnel">
      {funnel.map((stage) => {
        const width = Math.max((stage.count / top) * 100, 1.5);
        return (
          <li className="funnel-row" key={stage.label}>
            <div className="funnel-head">
              <span className="funnel-label">{stage.label}</span>
              <span className="funnel-count mono">{int(stage.count)}</span>
            </div>
            <div className="funnel-track">
              <div className="funnel-bar" style={{ width: `${width}%` }} />
            </div>
            {stage.description ? (
              <div className="funnel-desc">{stage.description}</div>
            ) : null}
          </li>
        );
      })}
    </ol>
  );
}

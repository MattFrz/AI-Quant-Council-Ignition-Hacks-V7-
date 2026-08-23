import type { Catalyst } from "../../lib/types";
import { CatalystCard } from "./CatalystCard";

export function AuditTrail({ catalysts }: { catalysts: Catalyst[] }) {
  return (
    <section className="card" aria-label="Evidence">
      <div className="card-title">
        <h2>Evidence</h2>
        <span className="count">
          {catalysts.length} catalyst{catalysts.length === 1 ? "" : "s"}, every one source-linked
        </span>
      </div>

      {catalysts.length === 0 ? (
        <p className="empty">
          No catalysts survived extraction. The recommendation below rests on the
          quantitative signal alone.
        </p>
      ) : (
        // Scrolls in its own box rather than growing the page.
        //
        // A run can return forty-plus catalysts, each with a full quote, which
        // pushes the debate and the backtest so far down that a reader never
        // reaches them. Keeping the evidence in a fixed frame means the shape
        // of the page stays the same whether a run found three or forty.
        <div className="scroll-pane" tabIndex={0} role="group" aria-label="Catalyst list">
          {catalysts.map((c) => (
            <CatalystCard key={c.catalyst_id} catalyst={c} />
          ))}
        </div>
      )}
    </section>
  );
}

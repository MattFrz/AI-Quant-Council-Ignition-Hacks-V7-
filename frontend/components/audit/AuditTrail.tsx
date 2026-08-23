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
        catalysts.map((c) => <CatalystCard key={c.catalyst_id} catalyst={c} />)
      )}
    </section>
  );
}

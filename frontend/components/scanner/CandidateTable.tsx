import type { CandidateSummary } from "../../lib/types";
import { num } from "../../lib/format";

/**
 * The shortlist the alpha model produced, with the headline evidence for each.
 *
 * The score column is a percentile rank, so it is bounded 0 to 10 and a bar
 * behind the number is meaningful. Sorting is left as the backend returned it,
 * which is already rank order.
 */
export function CandidateTable({ candidates }: { candidates: CandidateSummary[] }) {
  if (!candidates?.length) {
    return <p className="empty">No candidates recorded for this run.</p>;
  }

  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Company</th>
            <th>Sector</th>
            <th className="num">Alpha</th>
            <th>Headline catalyst</th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((c) => (
            <tr key={c.ticker}>
              <td className="mono strong">{c.ticker}</td>
              <td>{c.company_name}</td>
              <td className="text-secondary">{c.sector ?? "-"}</td>
              <td className="num">
                <span className="score-cell">
                  <span className="mono">{num(c.alpha_score, 1)}</span>
                  <span className="score-track">
                    <span
                      className="score-bar"
                      style={{ width: `${Math.min(c.alpha_score * 10, 100)}%` }}
                    />
                  </span>
                </span>
              </td>
              <td className="text-secondary catalyst-cell">
                {c.headline_catalyst ?? "-"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

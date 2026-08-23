/**
 * Bull and bear, side by side.
 *
 * Both are rendered at equal size on purpose. Shrinking the bear case, or
 * hiding it behind a toggle, would quietly turn a debate into a pitch - and
 * the bear is the half that has seen the quant results.
 */
export function DebatePanel({
  bullCase,
  bearCase,
}: {
  bullCase: string;
  bearCase: string;
}) {
  return (
    <section className="card" aria-label="Bull and bear cases">
      <div className="card-title">
        <h2>The debate</h2>
        <span className="count">same evidence, opposing briefs</span>
      </div>

      <div className="debate">
        <div className="case" data-side="bull">
          <h3>Bull case</h3>
          <div className="case-body">{bullCase || "No bull case generated."}</div>
        </div>
        <div className="case" data-side="bear">
          <h3>Bear case</h3>
          <div className="case-body">{bearCase || "No bear case generated."}</div>
        </div>
      </div>
    </section>
  );
}

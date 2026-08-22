// Placeholder home page so `npm run dev` runs and the Phase 0 gate passes.
// Cecile replaces this in step D5 with the thesis input screen.
export default function Home() {
  return (
    <main style={{ padding: "3rem", maxWidth: "42rem" }}>
      <h1>Autonomous Alpha</h1>
      <p>Frontend scaffold is running. Phase 0 gate passed.</p>
      <p>
        Next: step D1 in the build plan - point <code>lib/api.ts</code> at{" "}
        <code>data/fixtures/sample_trade_idea.json</code> and start building
        against the contract.
      </p>
    </main>
  );
}

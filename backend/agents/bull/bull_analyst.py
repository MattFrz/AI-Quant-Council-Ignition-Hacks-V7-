from __future__ import annotations

from backend.agents.base import Agent
from backend.agents.state import ResearchState
from backend.agents.prompts import BULL_ANALYST_PROMPT


class BullAnalyst(Agent):
    """
    Builds the strongest long case grounded ONLY in retrieved evidence.
    No outside knowledge injected — if the case doesn't trace back to a
    retrieved chunk, that's a prompt problem, not something to patch after.
    """

    def run(self, state: ResearchState) -> ResearchState:
        state.emit("bull_case", "Building bull case", "in_progress")

        evidence_block = self._format_evidence(state)
        messages = [
            {"role": "system", "content": BULL_ANALYST_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Thesis: {state.thesis}\n\n"
                    f"Retrieved evidence:\n{evidence_block}\n\n"
                    "Build the strongest possible long case using ONLY the "
                    "evidence above. Cite the source URL inline whenever you "
                    "reference a specific claim."
                ),
            },
        ]
        response = self.llm.complete(messages)
        state.bull_case = response.text

        state.emit("bull_case", "Bull case complete", "done")
        return state

    def _format_evidence(self, state: ResearchState) -> str:
        """Evidence lines, each LABELLED WITH ITS COMPANY.

        Without the ticker the model cannot tell whose filing it is reading,
        so it attributes freely - we saw it write a paragraph about NVIDIA and
        cite AMD's 10-K. The ticker is the fix: it is on every chunk, it just
        was not being passed through.
        """
        chunks = getattr(state, "retrieved_chunks", []) or []
        if chunks:
            seen = set()
            lines = []
            for c in chunks:
                if c.chunk_id in seen:
                    continue
                seen.add(c.chunk_id)
                lines.append(
                    f"- [{c.ticker}] [{c.form_type.value if hasattr(c.form_type, 'value') else c.form_type}, "
                    f"{c.filed_date}] {c.text[:300]} (source: {c.source_url})"
                )
            return chr(10).join(lines[:40])

        citations = getattr(state, "citations", [])
        if not citations:
            return "(no evidence retrieved)"
        return chr(10).join(
            f"- [{c.form_type}, {c.filed_date}] {c.text[:300]} (source: {c.source_url})"
            for c in citations
        )

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
        citations = getattr(state, "citations", [])
        if not citations:
            return "(no evidence retrieved)"

        lines = []
        for c in citations:
            lines.append(f"- [{c.form_type}, {c.filed_date}] {c.text[:300]} (source: {c.source_url})")
        return "\n".join(lines)
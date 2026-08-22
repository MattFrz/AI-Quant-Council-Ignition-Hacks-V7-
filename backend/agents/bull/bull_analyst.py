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

    @staticmethod
    def _subject_block(state) -> str:
        """Name the company under discussion.

        Without it both analysts argue the thesis in general - one run
        produced a VRT recommendation whose bull case was about NVDA and AMD,
        and which called Vertiv "Verint" (a different company entirely).
        """
        if not state.primary_ticker:
            return ""
        return (
            f"SUBJECT: {state.primary_name} ({state.primary_ticker}).\n"
            "Argue about THIS company. Use its name exactly as written above - "
            "do not substitute a similar-sounding company. Evidence from other "
            "companies may be cited only as context for this one.\n\n"
        )

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
            subject, context = [], []
            for c in chunks:
                if c.chunk_id in seen:
                    continue
                seen.add(c.chunk_id)
                form = c.form_type.value if hasattr(c.form_type, "value") else c.form_type
                line = f"- [{c.ticker}] [{form}, {c.filed_date}] {c.text[:300]} (source: {c.source_url})"
                if state.primary_ticker and c.ticker == state.primary_ticker:
                    subject.append(line)
                else:
                    context.append(line)

            # Subject evidence first and in bulk; other companies capped.
            #
            # A one-line "focus on VRT" instruction cannot outweigh forty lines
            # about NVDA and AMD - the model writes about whatever it was given
            # most of. Weighting the evidence is what actually moves it.
            out = []
            if subject:
                out.append(f"EVIDENCE ON {state.primary_ticker}:")
                out.extend(subject[:25])
            if context:
                out.append("")
                out.append("CONTEXT FROM OTHER COMPANIES (supporting only):")
                out.extend(context[:8])
            return chr(10).join(out) if out else "(no evidence retrieved)"

        citations = getattr(state, "citations", [])
        if not citations:
            return "(no evidence retrieved)"
        return chr(10).join(
            f"- [{c.form_type}, {c.filed_date}] {c.text[:300]} (source: {c.source_url})"
            for c in citations
        )

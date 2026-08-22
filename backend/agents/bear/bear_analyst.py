from __future__ import annotations

from backend.agents.base import Agent
from backend.agents.state import ResearchState
from backend.agents.prompts import BEAR_ANALYST_PROMPT

# Fixed attack checklist from the §4 spec — kept as a constant here (not
# buried in the prompt string) so it's easy to extend without hunting
# through prompts.py.
BEAR_ATTACK_CHECKLIST = [
    "Valuation — is this already expensive relative to peers/history?",
    "Already priced in — does the market already reflect this thesis?",
    "Margin pressure — are costs or competition eroding margins?",
    "Crowding — is this a consensus trade with a crowded-exit risk?",
    "Historical false positives — have similar setups failed before?",
]


class BearAnalyst(Agent):
    """
    Gets the SAME evidence as BullAnalyst, plus the quant stats (backtest +
    risk, populated by QuantValidator) and the fixed attack checklist.
    Requires QuantValidator (C20) to have already run — sequence it after
    C20 in the orchestrator so state.backtest_result / risk_metrics exist.
    """

    def run(self, state: ResearchState) -> ResearchState:
        state.emit("bear_case", "Building bear case", "in_progress")

        evidence_block = self._format_evidence(state)
        quant_block = self._format_quant(state)
        checklist_block = "\n".join(f"- {item}" for item in BEAR_ATTACK_CHECKLIST)

        messages = [
            {"role": "system", "content": BEAR_ANALYST_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Thesis: {state.thesis}\n\n"
                    + self._subject_block(state)
                    + f"Retrieved evidence:\n{evidence_block}\n\n"
                    f"Quant results:\n{quant_block}\n\n"
                    f"Attack checklist (address each explicitly if relevant):\n{checklist_block}\n\n"
                    "Build the strongest bear case, engaging directly with "
                    "the quant numbers above — a bear case that ignores the "
                    "numbers reads as generic and will be discounted."
                ),
            },
        ]
        response = self.llm.complete(messages)
        state.bear_case = response.text

        state.emit("bear_case", "Bear case complete", "done")
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
    def _format_quant(self, state: ResearchState) -> str:
        if state.backtest_result is None:
            return "(quant results not yet available — QuantValidator must run before BearAnalyst)"
        bt = state.backtest_result
        risk = state.risk_metrics or {}

        def num(value, pct: bool = False) -> str:
            # Rounded before the model sees it. Raw floats get quoted back
            # verbatim, and "a Sharpe of 1.2176655690477185" on screen reads
            # as nobody having looked at the output.
            if value is None:
                return "n/a"
            try:
                return f"{float(value):.1%}" if pct else f"{float(value):.2f}"
            except (TypeError, ValueError):
                return "n/a"

        return (
            f"Sharpe: {num(getattr(bt, 'sharpe', None))}, "
            f"Max drawdown: {num(getattr(bt, 'max_drawdown', None), pct=True)}, "
            f"Win rate: {num(getattr(bt, 'win_rate', None), pct=True)}, "
            f"Beta: {num(risk.get('beta'))}, "
            f"VaR (95%): {num(risk.get('var_95'), pct=True)}"
        )
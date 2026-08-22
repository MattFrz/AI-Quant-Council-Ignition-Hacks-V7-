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
                    f"Retrieved evidence:\n{evidence_block}\n\n"
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
    def _format_quant(self, state: ResearchState) -> str:
        if state.backtest_result is None:
            return "(quant results not yet available — QuantValidator must run before BearAnalyst)"
        bt = state.backtest_result
        risk = state.risk_metrics or {}
        return (
            f"Sharpe: {getattr(bt, 'sharpe', 'n/a')}, "
            f"Max drawdown: {getattr(bt, 'max_drawdown', 'n/a')}, "
            f"Win rate: {getattr(bt, 'win_rate', 'n/a')}, "
            f"Beta: {risk.get('beta', 'n/a')}, "
            f"VaR (95%): {risk.get('var_95', 'n/a')}"
        )
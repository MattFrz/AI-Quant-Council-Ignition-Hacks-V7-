from __future__ import annotations

from backend.agents.base import Agent
from backend.agents.state import ResearchState
from backend.agents.prompts import PORTFOLIO_MANAGER_PROMPT
from data.schemas.trade_idea import TradeIdea  # adjust import if your schema module differs


class PortfolioManager(Agent):
    """
    Final synthesis. Same boundary as QuantValidator: the LLM only writes
    pm_rationale (narrative). Every numeric field on TradeIdea comes from
    state, populated by earlier agents/lanes — never from this LLM call.
    """

    def run(self, state: ResearchState) -> TradeIdea:
        state.emit("synthesis", "Synthesizing final trade idea", "in_progress")

        if state.backtest_result is None or state.risk_metrics is None:
            raise ValueError(
                "PortfolioManager requires backtest_result and risk_metrics "
                "on state — run QuantValidator first."
            )

        messages = [
            {"role": "system", "content": PORTFOLIO_MANAGER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Thesis: {state.thesis}\n\n"
                    f"Bull case:\n{state.bull_case}\n\n"
                    f"Bear case:\n{state.bear_case}\n\n"
                    f"Quant results: Sharpe={getattr(state.backtest_result, 'sharpe', 'n/a')}, "
                    f"max_drawdown={getattr(state.backtest_result, 'max_drawdown', 'n/a')}\n\n"
                    "Write a final rationale (3-5 sentences) synthesizing the "
                    "evidence, the debate, and the quant results into a single "
                    "recommendation. Do not invent any numbers — reference only "
                    "the figures given above."
                ),
            },
        ]
        response = self.llm.complete(messages)

        ticker = self._infer_primary_ticker(state)
        direction = self._infer_direction(state)

        trade_idea = TradeIdea(
            ticker=ticker,
            direction=direction,
            alpha_score=self._top_alpha_score(state, ticker),
            confidence=self._confidence_from_agreement(state),
            expected_alpha=self._top_expected_alpha(state, ticker),
            catalysts=state.catalysts,
            signals=self._top_signal_contributions(state, ticker),
            backtest=state.backtest_result,
            risk=state.risk_metrics,
            bull_case=state.bull_case,
            bear_case=state.bear_case,
            pm_rationale=response.text,
        )

        state.emit("synthesis", "Trade idea complete", "done")
        return trade_idea

    # --- helpers pulling structured values off state; no LLM involvement below ---

    def _infer_primary_ticker(self, state: ResearchState) -> str:
        if state.catalysts:
            return state.catalysts[0].ticker
        raise ValueError("Cannot determine primary ticker — no catalysts on state")

    def _infer_direction(self, state: ResearchState) -> str:
        # naive default; replace with real logic once factor_scores/backtest
        # give a clear directional signal (e.g. sign of alpha_score)
        return "long"

    def _top_alpha_score(self, state: ResearchState, ticker: str):
        return (state.factor_scores or {}).get(ticker, {}).get("alpha_score")

    def _top_expected_alpha(self, state: ResearchState, ticker: str):
        return (state.factor_scores or {}).get(ticker, {}).get("expected_alpha")

    def _top_signal_contributions(self, state: ResearchState, ticker: str):
        return (state.factor_scores or {}).get(ticker, {}).get("signal_contributions", [])

    def _confidence_from_agreement(self, state: ResearchState) -> float:
        # placeholder heuristic — tune once you have real bull/bear/quant output
        # to calibrate against
        return 0.6
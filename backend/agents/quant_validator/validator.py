from __future__ import annotations

from backend.agents.base import Agent
from backend.agents.state import ResearchState

# These imports point at Matt's (Lane A) and Nalin's (Lane B) actual code.
# Adjust the exact function names/paths once their modules are finalized —
# the point of this file is that it ONLY calls into quant/, it never
# computes or estimates a number on its own. If Lane A/B isn't ready yet,
# this should raise or return None — never a fabricated placeholder number.
from quant.backtest.engine import run_backtest        # Matt, A11
from quant.backtest.metrics import compute_metrics    # Matt, A12
from quant.risk.metrics import compute_risk_metrics   # Matt, A15
from quant.risk.var import compute_var                # Matt, A16


class QuantValidator(Agent):
    """
    DO NOT FAKE. Every number returned by this agent comes from a real
    function call into quant/. No LLM call happens in this file. If you
    find yourself typing a literal float for a Sharpe ratio, risk score,
    or anything numeric — stop, that's the one thing this file must never do.
    """

    def run(self, state: ResearchState) -> ResearchState:
        state.emit("quant_validation", "Running backtest and risk checks", "in_progress")

        if state.universe is None or state.factor_scores is None:
            raise ValueError(
                "QuantValidator requires state.universe and state.factor_scores "
                "to already be populated by upstream lanes — it does not compute "
                "these itself."
            )

        raw_backtest = run_backtest(
            universe=state.universe,
            factor_scores=state.factor_scores,
            as_of=state.as_of,
        )
        state.backtest_result = compute_metrics(raw_backtest)

        state.risk_metrics = {
            **compute_risk_metrics(state.backtest_result),
            "var_95": compute_var(state.backtest_result, confidence=0.95),
            "var_99": compute_var(state.backtest_result, confidence=0.99),
        }

        state.emit("quant_validation", "Backtest and risk checks complete", "done")
        return state
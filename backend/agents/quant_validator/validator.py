from __future__ import annotations

from backend.agents.base import Agent
from backend.agents.state import ResearchState

# These imports point at Matt's (Lane A) and Nalin's (Lane B) actual code.
# Adjust the exact function names/paths once their modules are finalized -
# the point of this file is that it ONLY calls into quant/, it never
# computes or estimates a number on its own. If Lane A/B isn't ready yet,
# this should raise or return None - never a fabricated placeholder number.
# Phase 3: these now come from the quant/api.py facade (Matt), which wraps
# A11/A12/A15/A16. The facade exists so this file's contract - "only calls
# into quant/, never computes a number" - holds without either lane having to
# rename its internals.
from quant.api import (            # noqa: F401
    run_backtest,                  # Matt, A11
    compute_metrics,               # Matt, A12
    compute_risk_metrics,          # Matt, A15
    compute_var,                   # Matt, A16
)


class QuantValidator(Agent):
    """
    DO NOT FAKE. Every number returned by this agent comes from a real
    function call into quant/. No LLM call happens in this file. If you
    find yourself typing a literal float for a Sharpe ratio, risk score,
    or anything numeric - stop, that's the one thing this file must never do.
    """

    def run(self, state: ResearchState) -> ResearchState:
        state.emit("quant_validation", "Running backtest and risk checks", "in_progress")

        if state.universe is None or state.factor_scores is None:
            raise ValueError(
                "QuantValidator requires state.universe and state.factor_scores "
                "to already be populated by upstream lanes - it does not compute "
                "these itself."
            )

        raw_backtest = run_backtest(
            universe=state.universe,
            factor_scores=state.factor_scores,
            as_of=state.as_of,
        )
        state.backtest_result = compute_metrics(raw_backtest)

        state.risk_metrics = {
            **compute_risk_metrics(raw_backtest),  # the RUN, not the result - it carries weights
            "var_95": compute_var(raw_backtest, confidence=0.95),
            "var_99": compute_var(raw_backtest, confidence=0.99),
        }

        state.emit("quant_validation", "Backtest and risk checks complete", "done")
        return state
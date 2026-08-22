"""Portfolio-level and per-position constraints. Step D14 (section 13).

Transparent and explainable over optimal: every rule here is a plain
threshold check, not a black-box optimizer output. If a candidate is
excluded, constraint_violations() says exactly why - that reason is what
gets shown in the D15 portfolio/risk panel.
"""
from __future__ import annotations

from typing import List

from data.schemas.trade_idea import Side, TradeIdea, Verdict

# Minimum confidence to size a position at all - below this the signal isn't
# strong enough to act on regardless of alpha score.
MIN_CONFIDENCE = 0.55

# A risk band worse than this needs a confidence override to still qualify.
MAX_RISK_BAND_FOR_STANDARD_SIZE = "medium"
HIGH_RISK_CONFIDENCE_OVERRIDE = 0.75

_RISK_BAND_ORDER = {"low": 0, "medium": 1, "high": 2}


def constraint_violations(idea: TradeIdea) -> List[str]:
    """Every constraint this candidate fails, in plain language. Empty list
    means it's eligible for the portfolio."""
    violations: List[str] = []

    if idea.side == Side.NO_TRADE:
        violations.append("side is NO_TRADE - no directional signal to size")

    if idea.validator_verdict == Verdict.REJECTED:
        violations.append("quant validator rejected the idea")

    if idea.confidence < MIN_CONFIDENCE:
        violations.append(f"confidence {idea.confidence:.2f} below minimum {MIN_CONFIDENCE}")

    if not idea.has_audit_trail():
        violations.append("no clickable, cited catalyst - fails the audit trail requirement")

    if idea.risk is None:
        violations.append("no risk metrics attached - cannot size safely")
    else:
        band_rank = _RISK_BAND_ORDER.get(idea.risk.risk_band.value, 2)
        max_rank = _RISK_BAND_ORDER[MAX_RISK_BAND_FOR_STANDARD_SIZE]
        if band_rank > max_rank and idea.confidence < HIGH_RISK_CONFIDENCE_OVERRIDE:
            violations.append(
                f"risk_band={idea.risk.risk_band.value} requires confidence >= "
                f"{HIGH_RISK_CONFIDENCE_OVERRIDE}, got {idea.confidence:.2f}"
            )

    if idea.backtest is None:
        violations.append("no backtest attached - unvalidated signal")
    elif idea.backtest.sharpe < 0:
        violations.append(f"negative backtested Sharpe ({idea.backtest.sharpe:.2f})")

    return violations


def passes_constraints(idea: TradeIdea) -> bool:
    return len(constraint_violations(idea)) == 0


def eligible_candidates(candidates: List[TradeIdea]) -> List[TradeIdea]:
    """Everything that survives the constraint screen. Order is preserved
    from the input list - construction.py handles alpha ranking before
    calling this, so eligibility filtering doesn't need to re-sort."""
    return [c for c in candidates if passes_constraints(c)]

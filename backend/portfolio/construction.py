"""Portfolio construction. Step D14 (section 13).

    Candidate Universe -> Alpha Ranking -> Risk Filtering -> Correlation
    Analysis -> Position Sizing -> Portfolio

Each stage is a plain function here or in the sibling modules
(constraints.py, sizing.py) - nothing in this pipeline is a black-box
optimizer. Every excluded candidate carries its reasons; every sized
position traces back to alpha_score, confidence, volatility, and
correlation. That's the whole point per the plan: "transparent and
explainable over optimal."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from backend.portfolio.constraints import constraint_violations, eligible_candidates
from backend.portfolio.sizing import size_positions
from data.schemas.trade_idea import TradeIdea


@dataclass
class PortfolioPosition:
    ticker: str
    company_name: str
    side: str
    position_size_pct: float
    alpha_score: float
    confidence: float
    risk_band: str


@dataclass
class ExcludedCandidate:
    ticker: str
    reasons: List[str]


@dataclass
class Portfolio:
    positions: List[PortfolioPosition]
    excluded: List[ExcludedCandidate]
    total_invested_pct: float
    cash_pct: float


def rank_by_alpha(candidates: List[TradeIdea]) -> List[TradeIdea]:
    """Highest alpha_score first. Ties broken by confidence, then by lower
    volatility - prefer the safer of two equally-strong ideas."""

    def _vol(idea: TradeIdea) -> float:
        return idea.risk.volatility if (idea.risk and idea.risk.volatility) else 999.0

    return sorted(
        candidates,
        key=lambda idea: (-idea.alpha_score, -idea.confidence, _vol(idea)),
    )


def build_portfolio(
    candidates: List[TradeIdea],
    returns: Optional[pd.DataFrame] = None,
    max_position_pct: Optional[float] = None,
) -> Portfolio:
    """The full section 13 pipeline: rank -> filter -> size.

    `returns` is an optional wide DataFrame (date index, one column per
    ticker). When provided, sizing uses the real pairwise correlation
    matrix from quant/risk/correlation.py; when omitted, it falls back to
    each idea's own risk.avg_correlation summary stat, so this runs fine
    even before Matt's return-series pipeline is wired up. Either way the
    result is fully explainable - see excluded[i].reasons for every drop.
    """
    ranked = rank_by_alpha(candidates)

    excluded = [
        ExcludedCandidate(ticker=idea.ticker, reasons=violations)
        for idea in ranked
        if (violations := constraint_violations(idea))
    ]
    eligible = eligible_candidates(ranked)

    weights = size_positions(eligible, returns=returns, max_position_pct=max_position_pct)

    positions = [
        PortfolioPosition(
            ticker=idea.ticker,
            company_name=idea.company_name,
            side=idea.side.value,
            position_size_pct=round(weights.get(idea.ticker, 0.0), 2),
            alpha_score=idea.alpha_score,
            confidence=idea.confidence,
            risk_band=idea.risk.risk_band.value if idea.risk else "unknown",
        )
        for idea in eligible
        if weights.get(idea.ticker, 0.0) > 0
    ]

    total_invested = round(sum(p.position_size_pct for p in positions), 2)
    return Portfolio(
        positions=positions,
        excluded=excluded,
        total_invested_pct=total_invested,
        cash_pct=round(100.0 - total_invested, 2),
    )

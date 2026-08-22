"""Position sizing. Step D14 (section 13).

Volatility-scaled sizing capped by portfolio rules, per the plan's own
guidance: "a transparent methodology is preferable to an unnecessarily
complicated optimizer." Every weight here traces back to alpha_score,
confidence, volatility, and an explicit correlation penalty - no black box,
and every number is one a judge can be walked through by hand.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from backend.config import settings
from data.schemas.trade_idea import TradeIdea

# Floor applied to volatility before dividing, so a near-zero-vol candidate
# doesn't blow the sizing up toward an absurd weight.
MIN_VOLATILITY_FLOOR = 0.05

# Floor on the correlation penalty multiplier - even a maximally-correlated
# name still gets some allocation, just heavily discounted, rather than
# being zeroed out by sizing alone (constraints.py is where hard exclusions
# belong, not sizing).
MIN_CORRELATION_MULTIPLIER = 0.4


def _raw_score(idea: TradeIdea) -> float:
    """alpha_score (0-10) scaled by confidence, divided by volatility.
    Higher conviction + lower risk => bigger raw score before caps apply."""
    vol = idea.risk.volatility if (idea.risk and idea.risk.volatility) else 0.20
    vol = max(vol, MIN_VOLATILITY_FLOOR)
    return (idea.alpha_score * idea.confidence) / vol


def _correlation_penalty(idea: TradeIdea, returns: Optional[pd.DataFrame]) -> float:
    """Multiplier in (MIN_CORRELATION_MULTIPLIER, 1.0]. 1.0 = no penalty.

    Prefers a real pairwise correlation matrix when a wide returns
    DataFrame is available (see quant/risk/correlation.py); falls back to
    the idea's own summary risk.avg_correlation field otherwise - coarser,
    but still transparent, and works before Matt's return-series pipeline
    is wired up."""
    if returns is not None and idea.ticker in returns.columns:
        from quant.risk.correlation import average_pairwise_correlation

        avg_corr = average_pairwise_correlation(returns)
        if avg_corr is not None:
            return max(MIN_CORRELATION_MULTIPLIER, 1.0 - avg_corr)

    if idea.risk and idea.risk.avg_correlation is not None:
        return max(MIN_CORRELATION_MULTIPLIER, 1.0 - idea.risk.avg_correlation)

    return 1.0


def size_positions(
    candidates: List[TradeIdea],
    returns: Optional[pd.DataFrame] = None,
    max_position_pct: Optional[float] = None,
) -> Dict[str, float]:
    """ticker -> position_size_pct, each capped at max_position_pct.

    `candidates` should already be constraint-screened (see
    constraints.eligible_candidates) - this function only sizes, it doesn't
    decide who's eligible. Capping a name at the max can leave total weights
    under 100%; the remainder is left as cash rather than redistributed,
    since silently renormalizing past a stated cap defeats the point of
    having one."""
    cap = max_position_pct if max_position_pct is not None else settings.max_position_pct

    if not candidates:
        return {}

    scored = {
        idea.ticker: _raw_score(idea) * _correlation_penalty(idea, returns)
        for idea in candidates
    }
    total = sum(scored.values())
    if total <= 0:
        return {ticker: 0.0 for ticker in scored}

    weights = {ticker: (score / total) * 100.0 for ticker, score in scored.items()}
    return {ticker: min(w, cap) for ticker, w in weights.items()}

"""Portfolio construction endpoint. Step 3.3 (Cecile's piece - she owns
backend/portfolio per section 13 of the plan).

Builds a portfolio from a set of already-validated TradeIdea candidates
using the section 13 pipeline in backend/portfolio/construction.py. This
takes candidates directly in the request body, so it's testable in
isolation before job_runner.py (3.6, Matt) exists - once jobs are wired up,
a job_id-based lookup can be layered on top of the same build_portfolio()
call without changing this endpoint's contract.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.portfolio.construction import build_portfolio
from data.schemas.trade_idea import TradeIdea

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


class PortfolioPositionOut(BaseModel):
    ticker: str
    company_name: str
    side: str
    position_size_pct: float
    alpha_score: float
    confidence: float
    risk_band: str


class ExcludedCandidateOut(BaseModel):
    ticker: str
    reasons: List[str]


class PortfolioResponse(BaseModel):
    positions: List[PortfolioPositionOut]
    excluded: List[ExcludedCandidateOut]
    total_invested_pct: float
    cash_pct: float


class BuildPortfolioRequest(BaseModel):
    candidates: List[TradeIdea] = Field(..., min_length=1)
    max_position_pct: Optional[float] = Field(
        None, description="Override the default per-name cap for this run"
    )


@router.post("", response_model=PortfolioResponse)
def build_portfolio_route(payload: BuildPortfolioRequest) -> PortfolioResponse:
    """POST a list of validated TradeIdea candidates, get back a sized,
    explainable portfolio. See backend/portfolio/construction.py for the
    actual rank -> filter -> size pipeline, and constraints.py for exactly
    why any candidate got excluded."""
    portfolio = build_portfolio(
        payload.candidates,
        max_position_pct=payload.max_position_pct,
    )

    return PortfolioResponse(
        positions=[
            PortfolioPositionOut(
                ticker=p.ticker,
                company_name=p.company_name,
                side=p.side,
                position_size_pct=p.position_size_pct,
                alpha_score=p.alpha_score,
                confidence=p.confidence,
                risk_band=p.risk_band,
            )
            for p in portfolio.positions
        ],
        excluded=[
            ExcludedCandidateOut(ticker=e.ticker, reasons=e.reasons)
            for e in portfolio.excluded
        ],
        total_invested_pct=portfolio.total_invested_pct,
        cash_pct=portfolio.cash_pct,
    )

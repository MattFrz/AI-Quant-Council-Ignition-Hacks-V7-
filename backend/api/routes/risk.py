"""Risk endpoints. Step 3.3 (Nalin)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.routes.backtest import get_engine, get_panel
from backend.config import settings
from backend.core.logging import get_logger
from data.schemas.risk import RiskMetrics
from quant.optimization.vol_scaling import size_positions
from quant.risk.metrics import build_risk_metrics
from quant.risk.var import cvar_historical, var_historical

log = get_logger(__name__)
router = APIRouter(prefix="/risk", tags=["risk"])


class Position(BaseModel):
    ticker: str
    weight: float = Field(..., description="Fraction of portfolio, e.g. 0.05")


class RiskRequest(BaseModel):
    positions: List[Position] = Field(..., min_length=1)
    lookback_days: int = Field(252, ge=30, le=2520)
    portfolio_value: float = Field(1_000_000.0, gt=0)
    as_of: Optional[str] = None


class SizingRequest(BaseModel):
    """Size a book from the alpha model, then measure its risk."""

    max_names: int = Field(10, ge=1, le=50)
    max_position: float = Field(0.05, gt=0, le=1.0)
    target_vol: float = Field(0.10, gt=0, le=1.0)
    as_of: Optional[str] = None


class SizedPosition(BaseModel):
    ticker: str
    weight: float
    weight_pct: float
    realized_vol: Optional[float] = None
    alpha_score: Optional[float] = None


class SizedBookResponse(BaseModel):
    as_of: str
    positions: List[SizedPosition] = Field(default_factory=list)
    gross_exposure: float
    est_portfolio_vol: float
    target_vol: float
    n_capped: int
    risk: RiskMetrics
    note: Optional[str] = None


def _as_of_timestamp(raw: Optional[str]) -> pd.Timestamp:
    panel = get_panel()
    if raw is None:
        return panel.dates[-1]
    ts = pd.Timestamp(raw)
    pos = int(panel.dates.searchsorted(ts, side="right")) - 1
    if pos < 0:
        raise HTTPException(400, f"as_of {raw} precedes the price history.")
    return panel.dates[pos]


def _sectors() -> Dict[str, str]:
    panel = get_panel()
    if panel.securities is None or "sector" not in panel.securities:
        return {}
    return panel.securities["sector"].dropna().to_dict()


@router.post("", response_model=RiskMetrics)
@router.post("/", response_model=RiskMetrics, include_in_schema=False)
def portfolio_risk(request: RiskRequest) -> RiskMetrics:
    """The section 14 panel for a given set of positions."""
    panel = get_panel()
    as_of = _as_of_timestamp(request.as_of)

    weights = pd.Series({p.ticker: p.weight for p in request.positions}, dtype=float)
    unknown = [t for t in weights.index if t not in panel.tickers]
    if unknown:
        raise HTTPException(400, f"No price history for {unknown}")

    history = panel.as_of(as_of, lag_days=0)
    asset_returns = history.returns().iloc[-request.lookback_days:][list(weights.index)]
    if asset_returns.empty:
        raise HTTPException(400, "Not enough price history for that lookback.")

    portfolio_returns = (asset_returns * weights).sum(axis=1)

    benchmark = None
    bench_ticker = settings.benchmark_ticker
    if bench_ticker in history.tickers:
        benchmark = history.returns()[bench_ticker].iloc[-request.lookback_days:]

    adv = None
    if panel.adv is not None:
        adv = float(panel.adv.loc[:as_of].iloc[-1].reindex(weights.index).mean())

    metrics = build_risk_metrics(
        returns=portfolio_returns,
        benchmark_returns=benchmark,
        weights=weights,
        sectors=_sectors(),
        asset_returns=asset_returns,
        position_notional=request.portfolio_value * float(weights.abs().max()),
        adv_usd=adv,
        max_participation=settings.max_adv_participation,
    )
    log.info("risk: %d positions, vol %.2f%%, band %s",
             len(weights), (metrics.volatility or 0) * 100, metrics.risk_band.value)
    return metrics


@router.post("/sized-book", response_model=SizedBookResponse)
def sized_book(request: SizingRequest) -> SizedBookResponse:
    """Alpha ranking -> vol-scaled positions -> risk panel, in one call.

    This is the section 13 -> 14 handoff the opportunity screen renders.
    """
    panel = get_panel()
    engine = get_engine(0.6, False)
    as_of = _as_of_timestamp(request.as_of)

    scores = engine.scores()
    row = scores.reindex(scores.index[scores.index <= as_of])
    if row.empty:
        raise HTTPException(400, "No alpha scores on or before that date.")
    alpha_date = row.index[-1]
    alpha = row.iloc[-1]

    book = size_positions(
        alpha, panel, alpha_date,
        target_vol=request.target_vol,
        max_position=request.max_position,
        max_names=request.max_names,
    )
    if book.n_positions == 0:
        raise HTTPException(400, "No positive-alpha names on that date.")

    weights = book.weights
    history = panel.as_of(as_of, lag_days=0)
    asset_returns = history.returns().iloc[-252:][list(weights.index)]
    portfolio_returns = (asset_returns * weights).sum(axis=1)

    benchmark = None
    if settings.benchmark_ticker in history.tickers:
        benchmark = history.returns()[settings.benchmark_ticker].iloc[-252:]

    metrics = build_risk_metrics(
        returns=portfolio_returns,
        benchmark_returns=benchmark,
        weights=weights,
        sectors=_sectors(),
        asset_returns=asset_returns,
    )

    alpha_scores = engine.model.alpha_scores(engine.panels).loc[alpha_date]
    positions = [
        SizedPosition(
            ticker=t,
            weight=float(w),
            weight_pct=float(w) * 100.0,
            realized_vol=_clean(book.realized_vol.get(t)),
            alpha_score=_clean(alpha_scores.get(t)),
        )
        for t, w in weights.sort_values(ascending=False).items()
    ]

    note = None
    if book.capped:
        note = (f"{len(book.capped)} position(s) hit the {request.max_position:.0%} "
                f"cap, so the book sits under its {request.target_vol:.0%} vol target "
                f"at {book.est_portfolio_vol:.1%}.")

    return SizedBookResponse(
        as_of=str(alpha_date.date()),
        positions=positions,
        gross_exposure=float(book.gross_exposure),
        est_portfolio_vol=float(book.est_portfolio_vol),
        target_vol=float(book.target_vol),
        n_capped=len(book.capped),
        risk=metrics,
        note=note,
    )


def _clean(value):
    if value is None:
        return None
    try:
        return None if value != value else float(value)
    except (TypeError, ValueError):
        return None


@router.post("/tail")
def tail_risk(request: RiskRequest) -> dict:
    """Historical VaR and CVaR for a live book.

    Calls var_historical / cvar_historical directly — the same functions
    quant/api.py wraps for Zain's C20 validator, so the agent's number and the
    UI's number come from one implementation. The facade's compute_var takes a
    BacktestRun; at portfolio level there is no backtest to hand it, and
    manufacturing one just to have it unpicked again would be worse code, not
    better provenance.

    Returns null rather than a fabricated figure when history is thin.
    """
    panel = get_panel()
    as_of = _as_of_timestamp(request.as_of)
    weights = pd.Series({p.ticker: p.weight for p in request.positions}, dtype=float)

    unknown = [t for t in weights.index if t not in panel.tickers]
    if unknown:
        raise HTTPException(400, f"No price history for {unknown}")

    history = panel.as_of(as_of, lag_days=0)
    asset_returns = history.returns().iloc[-request.lookback_days:][list(weights.index)]
    portfolio_returns = (asset_returns * weights).sum(axis=1).dropna()

    var95 = var_historical(portfolio_returns, 0.95)
    return {
        "as_of": str(as_of.date()),
        "n_positions": int(len(weights)),
        "lookback_days": int(len(portfolio_returns)),
        "var_95": _clean(var95),
        "cvar_95": _clean(cvar_historical(portfolio_returns, 0.95)),
        "var_99": _clean(var_historical(portfolio_returns, 0.99)),
        "cvar_99": _clean(cvar_historical(portfolio_returns, 0.99)),
        "var_95_dollar": (
            None if var95 is None else _clean(var95 * request.portfolio_value)
        ),
    }


def _clean(value):
    if value is None:
        return None
    try:
        return None if value != value else float(value)
    except (TypeError, ValueError):
        return None

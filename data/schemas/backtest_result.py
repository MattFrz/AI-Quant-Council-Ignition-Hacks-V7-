"""Backtest output. Step 1.7 of the data contract.

Every field here is something a judge may ask about. If a number is not
computed, leave it None rather than inventing it.
"""
from __future__ import annotations

from datetime import date as Date
from typing import List, Optional

from pydantic import BaseModel, Field


class CurvePoint(BaseModel):
    date: Date
    value: float


class BacktestWindow(BaseModel):
    train_start: Date
    train_end: Date
    test_start: Date
    test_end: Date

    def is_clean(self) -> bool:
        """Train must end before test begins. No overlap, ever."""
        return self.train_end < self.test_start


class BacktestResult(BaseModel):
    strategy_name: str
    universe_size: int
    window: BacktestWindow

    # Curves - all three share an x-axis so the chart can overlay them directly.
    equity_curve: List[CurvePoint] = Field(default_factory=list)
    drawdown_curve: List[CurvePoint] = Field(default_factory=list)
    benchmark_curve: List[CurvePoint] = Field(default_factory=list)

    # Returns (fractions, not percents)
    total_return: float
    annualized_return: float
    benchmark_annualized_return: Optional[float] = None
    excess_return: Optional[float] = Field(None, description="Annualized, vs benchmark")

    # Risk-adjusted
    sharpe: float
    sortino: Optional[float] = None
    volatility: Optional[float] = None
    max_drawdown: float = Field(..., description="Negative fraction, e.g. -0.084")

    # Trading
    turnover: Optional[float] = Field(None, description="Annualized, 1.0 = full portfolio once")
    win_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    n_trades: Optional[int] = None

    # Assumptions - state them so the number can be argued with honestly.
    commission_bps: float = 1.0
    slippage_model: str = "participation_rate"
    max_adv_participation: float = 0.05
    is_walk_forward: bool = False
    notes: Optional[str] = None

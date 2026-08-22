"""Performance metrics. Step A12.

Every function takes a daily return series and returns a plain float. Nothing
here knows about the engine, so the same functions score a backtest, a
benchmark, or a single position.
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd

from data.schemas.backtest_result import BacktestResult, BacktestWindow, CurvePoint

TRADING_DAYS = 252

#: Standard deviations below this are treated as zero. A "constant" return
#: series does not have std exactly 0 in floating point - [0.001]*100 has
#: std 2.2e-19 - and dividing by that produces a Sharpe of 7e16. Any near-zero
#: denominator here means the strategy has no meaningful variation, so the
#: honest answer is 0, not a headline number no judge would believe.
_MIN_STD = 1e-12


def annualized_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    total = float((1.0 + r).prod())
    years = len(r) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def total_return(returns: pd.Series) -> float:
    r = returns.dropna()
    return float((1.0 + r).prod() - 1.0) if len(r) else 0.0


def volatility(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0


def sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd < _MIN_STD or not np.isfinite(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Downside deviation only. Rewards strategies whose volatility is upside."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    dd = downside.std(ddof=1)
    if dd < _MIN_STD or not np.isfinite(dd):
        return 0.0
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def equity_curve(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    return start_value * (1.0 + returns.fillna(0.0)).cumprod()


def drawdown_curve(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak. Always <= 0."""
    peak = equity.cummax()
    return (equity / peak) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    return float(drawdown_curve(equity).min())


def win_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    r = r[r != 0]
    return float((r > 0).mean()) if len(r) else 0.0


def turnover(weights: pd.DataFrame) -> float:
    """Annualized one-way turnover. 1.0 means the book is replaced once a year."""
    if len(weights) < 2:
        return 0.0
    changes = weights.fillna(0.0).diff().abs().sum(axis=1) / 2.0
    daily = float(changes.mean())
    return daily * TRADING_DAYS


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    joined = pd.concat([returns, benchmark], axis=1).dropna()
    if len(joined) < 2:
        return 0.0
    var = joined.iloc[:, 1].var(ddof=1)
    if var < _MIN_STD or not np.isfinite(var):
        return 0.0
    return float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / var)


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    joined = pd.concat([returns, benchmark], axis=1).dropna()
    if len(joined) < 2:
        return 0.0
    active = joined.iloc[:, 0] - joined.iloc[:, 1]
    sd = active.std(ddof=1)
    if sd < _MIN_STD or not np.isfinite(sd):
        return 0.0
    return float(active.mean() / sd * np.sqrt(TRADING_DAYS))


def _points(series: pd.Series) -> List[CurvePoint]:
    return [
        CurvePoint(date=_as_date(idx), value=float(val))
        for idx, val in series.dropna().items()
    ]


def _as_date(idx) -> date:
    ts = pd.Timestamp(idx)
    return ts.date()


def build_result(
    strategy_name: str,
    returns: pd.Series,
    weights: pd.DataFrame,
    window: BacktestWindow,
    universe_size: int,
    benchmark_returns: Optional[pd.Series] = None,
    n_trades: Optional[int] = None,
    commission_bps: float = 1.0,
    slippage_model: str = "participation_rate",
    max_adv_participation: float = 0.05,
    is_walk_forward: bool = False,
    notes: Optional[str] = None,
) -> BacktestResult:
    """Assemble the schema object the API and UI consume.

    Everything here is derived from the return series - no number is passed in
    by hand, so the result cannot disagree with the curve it ships with.
    """
    eq = equity_curve(returns)
    dd = drawdown_curve(eq)

    bench_curve: List[CurvePoint] = []
    bench_ann: Optional[float] = None
    excess: Optional[float] = None
    if benchmark_returns is not None and len(benchmark_returns.dropna()):
        aligned = benchmark_returns.reindex(returns.index)
        bench_eq = equity_curve(aligned)
        bench_curve = _points(bench_eq)
        bench_ann = annualized_return(aligned)
        excess = annualized_return(returns) - bench_ann

    return BacktestResult(
        strategy_name=strategy_name,
        universe_size=universe_size,
        window=window,
        equity_curve=_points(eq),
        drawdown_curve=_points(dd),
        benchmark_curve=bench_curve,
        total_return=total_return(returns),
        annualized_return=annualized_return(returns),
        benchmark_annualized_return=bench_ann,
        excess_return=excess,
        sharpe=sharpe(returns),
        sortino=sortino(returns),
        volatility=volatility(returns),
        max_drawdown=max_drawdown(eq),
        turnover=turnover(weights),
        win_rate=win_rate(returns),
        n_trades=n_trades,
        commission_bps=commission_bps,
        slippage_model=slippage_model,
        max_adv_participation=max_adv_participation,
        is_walk_forward=is_walk_forward,
        notes=notes,
    )

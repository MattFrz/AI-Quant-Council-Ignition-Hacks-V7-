"""Stable facade over the Lane A quant engine. Phase 3 integration seam.

Zain's QuantValidator (C20) imports four functions by name. Those names did not
match Lane A's internals, so rather than rewrite either side this module
provides them, implemented over the real engine.

The point of C20 is that the validator NEVER computes a number itself - it only
calls into quant/. This file preserves that: every value returned here comes out
of the backtester, the metrics module or the risk module. Nothing is estimated.

    run_backtest(universe, factor_scores, as_of)  -> BacktestRun
    compute_metrics(run)                          -> BacktestResult
    compute_risk_metrics(result)                  -> dict
    compute_var(result, confidence)               -> float
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from backend.core.logging import get_logger
from data.pipelines.prices import load_wide
from data.schemas.backtest_result import BacktestResult, BacktestWindow
from quant.backtest.engine import Backtester, BacktestConfig, BacktestRun
from quant.risk.metrics import build_risk_metrics
from quant.risk.var import cvar_historical, var_historical

log = get_logger(__name__)

DateLike = Union[str, date, pd.Timestamp]


# ------------------------------------------------------------------ helpers

def _ticker_list(universe: Sequence[Any]) -> list:
    """Accept list[str] or list[Security] and return plain tickers."""
    out = []
    for item in universe or []:
        out.append(item if isinstance(item, str) else getattr(item, "ticker", str(item)))
    return out


def _to_signal_frame(
    factor_scores: Any,
    dates: pd.DatetimeIndex,
    tickers: Sequence[str],
) -> pd.DataFrame:
    """Normalise whatever Lane B hands over into a wide signal frame.

    Accepts a wide DataFrame (date x ticker), a per-date Series, or a plain
    {ticker: score} mapping. The last two are broadcast across all dates, which
    is correct for a single-date decision but means the backtest is evaluating
    a static cross-section - stated here so nobody mistakes it for a
    time-varying signal.
    """
    if isinstance(factor_scores, pd.DataFrame):
        return factor_scores.reindex(index=dates, columns=tickers)

    if isinstance(factor_scores, pd.Series):
        row = factor_scores.reindex(tickers)
        return pd.DataFrame([row.values] * len(dates), index=dates, columns=tickers)

    if isinstance(factor_scores, Mapping):
        row = pd.Series({k: float(v) for k, v in factor_scores.items()
                         if isinstance(v, (int, float, np.floating))})
        if row.empty:
            raise ValueError("factor_scores mapping contained no numeric scores")
        row = row.reindex(tickers)
        return pd.DataFrame([row.values] * len(dates), index=dates, columns=tickers)

    raise TypeError(
        f"factor_scores must be a DataFrame, Series or mapping, got {type(factor_scores)}"
    )


def _returns_from_result(result: BacktestResult) -> pd.Series:
    """Recover the daily return series from a result's equity curve.

    The curve is the authoritative record of what the backtest produced, so
    deriving from it guarantees the risk numbers describe the same run.
    """
    if not result.equity_curve:
        return pd.Series(dtype=float)
    eq = pd.Series(
        [p.value for p in result.equity_curve],
        index=pd.to_datetime([p.date for p in result.equity_curve]),
    ).sort_index()
    return eq.pct_change(fill_method=None).dropna()


# -------------------------------------------------------------------- API

def run_backtest(
    universe: Sequence[Any],
    factor_scores: Any,
    as_of: DateLike,
    lookback_years: float = 5.0,
    config: Optional[BacktestConfig] = None,
) -> BacktestRun:
    """Backtest a scored universe using cached prices.

    Only data on or before `as_of` is used, so a historical decision date
    produces a historically honest result.
    """
    tickers = _ticker_list(universe)
    if not tickers:
        raise ValueError("run_backtest: universe is empty")

    panel = load_wide(tickers=tickers)
    stamp = pd.Timestamp(as_of)
    start = stamp - pd.Timedelta(days=int(365.25 * lookback_years))

    mask = (panel.close.index <= stamp) & (panel.close.index >= start)
    close = panel.close.loc[mask]
    adv = panel.adv.loc[mask]

    if close.empty:
        raise ValueError(f"run_backtest: no cached prices on or before {stamp.date()}")

    available = [t for t in tickers if t in close.columns]
    missing = sorted(set(tickers) - set(available))
    if missing:
        log.warning("run_backtest: %d universe names have no prices (%s%s)",
                    len(missing), ", ".join(missing[:5]),
                    "..." if len(missing) > 5 else "")
    if not available:
        raise ValueError("run_backtest: none of the universe has cached prices")

    close = close[available]
    adv = adv[available]
    signal = _to_signal_frame(factor_scores, close.index, available)

    cfg = config or BacktestConfig(
        rebalance_freq="ME", max_names=min(10, len(available)), strategy_name="composite_alpha"
    )

    return Backtester(cfg).run(
        signal=signal,
        close=close,
        adv=adv,
        window=BacktestWindow(
            train_start=close.index[0].date(),
            train_end=close.index[0].date(),
            test_start=close.index[0].date(),
            test_end=close.index[-1].date(),
        ),
    )


def compute_metrics(raw: Union[BacktestRun, BacktestResult]) -> BacktestResult:
    """Extract the schema object. Metrics are computed inside the engine, so
    this is a projection, not a second calculation."""
    if isinstance(raw, BacktestResult):
        return raw
    if isinstance(raw, BacktestRun):
        return raw.result
    raise TypeError(f"compute_metrics expected BacktestRun or BacktestResult, got {type(raw)}")


def compute_risk_metrics(
    result: Union[BacktestRun, BacktestResult],
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Risk panel as a plain dict, for state.risk_metrics."""
    if isinstance(result, BacktestRun):
        returns = result.returns
        result = result.result
    else:
        returns = _returns_from_result(result)

    rm = build_risk_metrics(returns, benchmark_returns)

    return {
        "beta": rm.beta,
        "volatility": rm.volatility,
        "max_drawdown": result.max_drawdown,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "risk_band": rm.risk_band.value,
        "var_95": rm.var_95,
        "cvar_95": rm.cvar_95,
    }


def compute_var(
    result: Union[BacktestRun, BacktestResult],
    confidence: float = 0.95,
) -> Optional[float]:
    """Historical VaR as a negative fraction. None if history is too thin -
    never a fabricated number."""
    returns = result.returns if isinstance(result, BacktestRun) else _returns_from_result(result)
    return var_historical(returns, confidence)


def compute_cvar(
    result: Union[BacktestRun, BacktestResult],
    confidence: float = 0.95,
) -> Optional[float]:
    returns = result.returns if isinstance(result, BacktestRun) else _returns_from_result(result)
    return cvar_historical(returns, confidence)

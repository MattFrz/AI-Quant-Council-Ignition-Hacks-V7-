"""Benchmark comparison. Step A13.

Judges discount an absolute return and respect an excess one. Every headline
number the demo shows should be relative to this.
"""
from __future__ import annotations

from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from backend.config import settings
from backend.core.logging import get_logger
from data.sources.yahoo import YahooSource
from quant.backtest.metrics import (
    annualized_return,
    beta,
    information_ratio,
    sharpe,
    volatility,
)

log = get_logger(__name__)


def load_benchmark(
    ticker: Optional[str] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.Series:
    """Daily returns of the benchmark, indexed by date."""
    ticker = ticker or settings.benchmark_ticker
    start = start or settings.backtest_start
    end = end or settings.backtest_end

    panel = YahooSource().fetch_prices([ticker], start, end)
    if panel.empty:
        log.warning("benchmark %s returned no data", ticker)
        return pd.Series(dtype=float)

    s = panel.set_index("date")["adj_close"].sort_index()
    s.index = pd.to_datetime(s.index)
    return s.pct_change(fill_method=None).rename(ticker)


def excess_returns(strategy: pd.Series, benchmark: pd.Series) -> pd.Series:
    joined = pd.concat([strategy, benchmark], axis=1).dropna()
    return joined.iloc[:, 0] - joined.iloc[:, 1]


def alpha_beta(strategy: pd.Series, benchmark: pd.Series) -> Tuple[float, float]:
    """CAPM alpha (annualized) and beta, by OLS on daily returns."""
    joined = pd.concat([strategy, benchmark], axis=1).dropna()
    if len(joined) < 30:
        return 0.0, 0.0

    b = beta(joined.iloc[:, 0], joined.iloc[:, 1])
    daily_alpha = joined.iloc[:, 0].mean() - b * joined.iloc[:, 1].mean()
    return float(daily_alpha * 252), float(b)


def compare(strategy: pd.Series, benchmark: pd.Series) -> dict:
    """Side-by-side table for the UI and for the quant validator agent."""
    joined = pd.concat([strategy, benchmark], axis=1).dropna()
    if joined.empty:
        return {}

    s, b = joined.iloc[:, 0], joined.iloc[:, 1]
    ann_alpha, beta_val = alpha_beta(s, b)

    return {
        "strategy_annualized": annualized_return(s),
        "benchmark_annualized": annualized_return(b),
        "excess_annualized": annualized_return(s) - annualized_return(b),
        "strategy_sharpe": sharpe(s),
        "benchmark_sharpe": sharpe(b),
        "strategy_volatility": volatility(s),
        "benchmark_volatility": volatility(b),
        "capm_alpha": ann_alpha,
        "beta": beta_val,
        "information_ratio": information_ratio(s, b),
        "n_days": int(len(joined)),
    }

"""Value at Risk and Conditional VaR. Step A16.

Historical simulation, not a normal assumption. Return distributions have fat
tails and a Gaussian VaR understates exactly the days you care about.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def var_historical(returns: pd.Series, confidence: float = 0.95) -> Optional[float]:
    """1-day VaR as a NEGATIVE fraction: -0.021 means a 2.1% loss.

    The empirical quantile of realised returns - no distributional assumption.
    """
    r = returns.dropna()
    if len(r) < 30:
        return None
    return float(np.quantile(r, 1.0 - confidence))


def cvar_historical(returns: pd.Series, confidence: float = 0.95) -> Optional[float]:
    """Expected loss GIVEN the VaR threshold is breached. Always <= VaR."""
    r = returns.dropna()
    if len(r) < 30:
        return None
    threshold = np.quantile(r, 1.0 - confidence)
    tail = r[r <= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(tail.mean())


def var_dollar(returns: pd.Series, portfolio_value: float, confidence: float = 0.95) -> Optional[float]:
    v = var_historical(returns, confidence)
    return None if v is None else float(v * portfolio_value)


def rolling_var(returns: pd.Series, window: int = 252, confidence: float = 0.95) -> pd.Series:
    """VaR through time - shows whether tail risk is rising."""
    return returns.rolling(window, min_periods=60).quantile(1.0 - confidence)


def stress_scenarios(returns: pd.Series) -> dict:
    """Worst realised days and windows. Concrete beats a percentile in a demo."""
    r = returns.dropna()
    if r.empty:
        return {}
    cum5 = (1 + r).rolling(5).apply(np.prod, raw=True) - 1
    cum20 = (1 + r).rolling(20).apply(np.prod, raw=True) - 1
    return {
        "worst_day": float(r.min()),
        "worst_day_date": str(r.idxmin().date()) if hasattr(r.idxmin(), "date") else str(r.idxmin()),
        "worst_week": float(cum5.min()) if cum5.notna().any() else None,
        "worst_month": float(cum20.min()) if cum20.notna().any() else None,
        "pct_days_negative": float((r < 0).mean()),
    }

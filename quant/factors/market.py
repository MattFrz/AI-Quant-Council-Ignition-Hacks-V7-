"""Price-and-volume factors. Step B2."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from data.schemas.signal import FactorCategory
from quant.factors.base import Factor, Panel

TRADING_DAYS = 252


class Momentum12_1(Factor):
    """Return from t-lookback to t-skip."""

    category = FactorCategory.MOMENTUM

    def __init__(self, lookback: int = 252, skip: int = 21, min_lag_days: int = 1) -> None:
        if lookback <= skip:
            raise ValueError(f"lookback ({lookback}) must exceed skip ({skip})")
        self.lookback = lookback
        self.skip = skip
        self.min_lag_days = min_lag_days
        self.required_history = lookback + 1
        self.name = f"momentum_{round(lookback / 21)}_{round(skip / 21)}"

    def _compute(self, window: Panel) -> pd.Series:
        px = window.adj_close.where(window.adj_close > 0)
        end = px.iloc[-1 - self.skip]
        start = px.iloc[-1 - self.lookback]
        return (end / start) - 1.0


class RealizedVolatility(Factor):
    """Annualized standard deviation of daily log returns."""

    category = FactorCategory.RISK

    def __init__(self, window: int = 60, annualize: bool = True, min_lag_days: int = 1) -> None:
        self.window = window
        self.annualize = annualize
        self.min_lag_days = min_lag_days
        self.required_history = window + 1
        self.name = f"realized_vol_{window}d"

    def _compute(self, window: Panel) -> pd.Series:
        px = window.adj_close.where(window.adj_close > 0).iloc[-(self.window + 1):]
        rets = np.log(px).diff().iloc[1:]
        vol = rets.std(ddof=1)
        return vol * np.sqrt(TRADING_DAYS) if self.annualize else vol


class VolumeTrend(Factor):
    """Recent dollar volume against its own baseline."""

    category = FactorCategory.EVENT

    def __init__(self, short: int = 20, long: int = 60, min_lag_days: int = 1) -> None:
        if short >= long:
            raise ValueError(f"short ({short}) must be less than long ({long})")
        self.short = short
        self.long = long
        self.min_lag_days = min_lag_days
        self.required_history = long
        self.name = f"volume_trend_{short}_{long}"

    def _compute(self, window: Panel) -> pd.Series:
        dv = window.dollar_volume()
        recent = dv.iloc[-self.short:].mean()
        baseline = dv.iloc[-self.long:].mean()
        baseline = baseline.where(baseline > 0)
        return (recent / baseline) - 1.0


class RelativeStrength(Factor):
    """Trailing return minus the universe's average trailing return."""

    category = FactorCategory.MOMENTUM

    def __init__(self, window: int = 126, min_lag_days: int = 1) -> None:
        self.window = window
        self.min_lag_days = min_lag_days
        self.required_history = window + 1
        self.name = f"relative_strength_{window}d"

    def _compute(self, window: Panel) -> pd.Series:
        px = window.adj_close.where(window.adj_close > 0)
        total = (px.iloc[-1] / px.iloc[-1 - self.window]) - 1.0
        return total - total.mean(skipna=True)


def default_market_factors() -> List[Factor]:
    """The B2 set, in the order they get reported on the scoreboard."""
    return [
        Momentum12_1(),
        RealizedVolatility(),
        VolumeTrend(),
        RelativeStrength(),
    ]

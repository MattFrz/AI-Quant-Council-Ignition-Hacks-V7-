"""Market regime detection. Step B14 — optional, done after the lane's core."""
from __future__ import annotations

from enum import Enum
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from quant.alpha.statistical_tests import cross_sectional_ic, summarize_ic
from quant.factors.base import Panel
from quant.signals.cross_sectional import forward_returns

TRADING_DAYS = 252


class Regime(str, Enum):
    CALM_UPTREND = "calm_uptrend"
    VOLATILE_UPTREND = "volatile_uptrend"
    CALM_DOWNTREND = "calm_downtrend"
    VOLATILE_DOWNTREND = "volatile_downtrend"
    UNKNOWN = "unknown"


def market_index(panel: Panel) -> pd.Series:
    """Equal-weighted universe index. A benchmark when there isn't one."""
    rets = panel.returns().mean(axis=1)
    return (1.0 + rets.fillna(0.0)).cumprod()


def classify_regimes(
    panel: Panel,
    trend_window: int = 200,
    vol_window: int = 60,
    lag_days: int = 1,
    min_history: int = 252,
) -> pd.Series:
    """One `Regime` per date, using only data available on that date."""
    index = market_index(panel)
    rets = np.log(index).diff()

    trend_ma = index.rolling(trend_window, min_periods=trend_window // 2).mean()
    uptrend = index > trend_ma

    vol = rets.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(TRADING_DAYS)
    # Expanding threshold: what counted as "high vol" using only the past.
    vol_threshold = vol.expanding(min_periods=min_history).median()
    volatile = vol > vol_threshold

    labels = pd.Series(Regime.UNKNOWN.value, index=index.index, dtype=object)
    known = uptrend.notna() & volatile.notna()
    labels[known & uptrend & ~volatile] = Regime.CALM_UPTREND.value
    labels[known & uptrend & volatile] = Regime.VOLATILE_UPTREND.value
    labels[known & ~uptrend & ~volatile] = Regime.CALM_DOWNTREND.value
    labels[known & ~uptrend & volatile] = Regime.VOLATILE_DOWNTREND.value

    if lag_days:
        labels = labels.shift(lag_days).fillna(Regime.UNKNOWN.value)
    return labels.rename("regime")


def regime_summary(regimes: pd.Series) -> pd.DataFrame:
    """How much time was spent in each regime. Sanity check before trusting"""
    counts = regimes.value_counts()
    return pd.DataFrame({
        "days": counts,
        "share": counts / len(regimes),
    }).sort_values("days", ascending=False)


def ic_by_regime(
    factor_df: pd.DataFrame,
    panel: Panel,
    regimes: Optional[pd.Series] = None,
    horizon: int = 21,
    method: str = "spearman",
    name: str = "factor",
    min_periods: int = 6,
) -> pd.DataFrame:
    """IC conditioned on the regime in force at each rebalance date."""
    if regimes is None:
        regimes = classify_regimes(panel)

    fwd = forward_returns(panel, horizon=horizon, dates=factor_df.index)
    ic = cross_sectional_ic(factor_df, fwd, method=method)
    if ic.empty:
        return pd.DataFrame()

    labels = regimes.reindex(ic.index).fillna(Regime.UNKNOWN.value)

    rows = []
    for label, group in ic.groupby(labels):
        if len(group) < min_periods:
            continue  # too thin to report rather than reported with a caveat
        summary = summarize_ic(group, factor=name, horizon=horizon, method=method)
        row = summary.as_row()
        row["regime"] = label
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("regime").sort_values("mean_ic", ascending=False)


def regime_at(regimes: pd.Series, as_of) -> Regime:
    """The regime in force on a date — what the PM agent quotes."""
    ts = pd.Timestamp(as_of)
    pos = int(regimes.index.searchsorted(ts, side="right")) - 1
    if pos < 0:
        return Regime.UNKNOWN
    return Regime(regimes.iloc[pos])

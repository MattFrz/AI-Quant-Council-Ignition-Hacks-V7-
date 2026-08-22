"""Running a factor across every date. Step B4."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from quant.factors.base import Factor, Panel
from quant.signals.normalization import MIN_OBS, normalize, rank_transform


def rebalance_dates(
    panel: Panel,
    freq: str = "ME",
    warmup: int = 0,
) -> pd.DatetimeIndex:
    """Actual trading dates on a rebalance schedule."""
    dates = panel.dates[warmup:] if warmup else panel.dates
    if len(dates) == 0:
        return pd.DatetimeIndex([])
    marks = pd.Series(dates, index=dates).groupby(pd.Grouper(freq=freq)).last().dropna()
    return pd.DatetimeIndex(marks.values)


def factor_panel(
    factor: Factor,
    panel: Panel,
    dates: Optional[Sequence] = None,
    restrict_to_universe: bool = True,
) -> pd.DataFrame:
    """Compute one factor on every date. Returns raw values, `date x ticker`."""
    if dates is None:
        dates = rebalance_dates(panel, warmup=factor.required_history)

    rows = {}
    for d in pd.DatetimeIndex(dates):
        values = factor.compute(panel, d)
        if restrict_to_universe and panel.universe is not None:
            members = panel.members(d)
            values = values.where(values.index.isin(members))
        rows[d] = values

    out = pd.DataFrame(rows).T
    out.index.name = "date"
    out.columns.name = "ticker"
    return out.reindex(columns=panel.tickers)


def normalize_panel(raw: pd.DataFrame, min_obs: int = MIN_OBS, **kwargs) -> pd.DataFrame:
    """Apply the B3 pipeline to each date independently."""
    return raw.apply(lambda row: normalize(row, min_obs=min_obs, **kwargs), axis=1)


def rank_panel(raw: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """Cross-sectional rank per date."""
    return raw.rank(axis=1, pct=pct, na_option="keep")


def demean_by_group(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Subtract the group mean within each date — sector neutralization."""
    g = groups.reindex(df.columns)
    if g.isna().all():
        return df
    group_means = df.T.groupby(g).transform("mean").T
    return df - group_means


def forward_returns(
    panel: Panel,
    horizon: int = 21,
    dates: Optional[Sequence] = None,
) -> pd.DataFrame:
    """Return from each date to `horizon` trading days later. EVALUATION ONLY."""
    px = panel.adj_close.where(panel.adj_close > 0)
    fwd = px.shift(-horizon) / px - 1.0
    if dates is not None:
        fwd = fwd.reindex(pd.DatetimeIndex(dates))
    return fwd


def align(factor_df: pd.DataFrame, returns_df: pd.DataFrame) -> tuple:
    """Restrict two panels to their shared dates and tickers."""
    dates = factor_df.index.intersection(returns_df.index)
    tickers = factor_df.columns.intersection(returns_df.columns)
    return factor_df.loc[dates, tickers], returns_df.loc[dates, tickers]


def build_factor_panels(
    factors: Iterable[Factor],
    panel: Panel,
    dates: Optional[Sequence] = None,
    normalize_each: bool = True,
    sector_neutral: bool = False,
) -> dict:
    """Every factor at once, keyed by factor name. The input to B9's composite."""
    if dates is None:
        longest = max((f.required_history for f in factors), default=0)
        dates = rebalance_dates(panel, warmup=longest)

    sectors = None
    if sector_neutral and panel.securities is not None and "sector" in panel.securities:
        sectors = panel.securities["sector"]

    out = {}
    for f in factors:
        raw = factor_panel(f, panel, dates)
        if sectors is not None:
            raw = demean_by_group(raw, sectors)
        out[f.name] = normalize_panel(raw) if normalize_each else raw
    return out

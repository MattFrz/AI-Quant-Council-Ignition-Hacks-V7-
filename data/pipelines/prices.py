"""Price panel: download, adjust, store, reshape. Step A3.

The quant engine works in wide DataFrames (index=date, columns=ticker), not in
Bar objects. Bar exists for API responses; this module is what factors and the
backtester actually consume.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from backend.config import settings
from backend.core.logging import get_logger
from data.sources.yahoo import YahooSource

log = get_logger(__name__)

PANEL_FILE = "price_panel.parquet"
PROFILE_FILE = "profiles.parquet"


# --------------------------------------------------------------------- build

def build_price_panel(
    tickers: Sequence[str],
    start: Optional[date] = None,
    end: Optional[date] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download the long-format panel and persist it. Returns the panel."""
    start = start or settings.backtest_start
    end = end or settings.backtest_end

    source = YahooSource()
    panel = source.fetch_prices(tickers, start, end, refresh=refresh)

    out = settings.cache_path / PANEL_FILE
    panel.to_parquet(out, index=False)
    log.info("price panel: %d rows, %d tickers -> %s",
             len(panel), panel["ticker"].nunique() if len(panel) else 0, out)
    return panel


def build_profiles(tickers: Sequence[str], refresh: bool = False) -> pd.DataFrame:
    source = YahooSource()
    profiles = source.fetch_profiles(tickers, refresh=refresh)
    out = settings.cache_path / PROFILE_FILE
    profiles.to_parquet(out, index=False)
    log.info("profiles: %d rows -> %s", len(profiles), out)
    return profiles


# ---------------------------------------------------------------------- load

def load_prices(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or (settings.cache_path / PANEL_FILE)
    if not p.exists():
        raise FileNotFoundError(
            f"No price panel at {p}. Run:  python scripts/seed_data.py"
        )
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_profiles(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or (settings.cache_path / PROFILE_FILE)
    if not p.exists():
        raise FileNotFoundError(
            f"No profiles at {p}. Run:  python scripts/seed_data.py"
        )
    return pd.read_parquet(p)


# ------------------------------------------------------------------- reshape

def to_wide(panel: pd.DataFrame, field: str = "adj_close") -> pd.DataFrame:
    """Long panel -> wide frame, index=date, columns=ticker."""
    if field not in panel.columns:
        raise KeyError(f"{field} not in panel columns {list(panel.columns)}")
    wide = panel.pivot_table(index="date", columns="ticker", values=field, aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def daily_returns(close_wide: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from adjusted closes."""
    return close_wide.sort_index().pct_change(fill_method=None)


def dollar_volume(close_wide: pd.DataFrame, volume_wide: pd.DataFrame) -> pd.DataFrame:
    return close_wide * volume_wide


def average_dollar_volume(
    close_wide: pd.DataFrame, volume_wide: pd.DataFrame, window: int = 20
) -> pd.DataFrame:
    """Rolling ADV in dollars. The liquidity input to universe filters, position
    limits and the days-to-liquidate risk metric."""
    return dollar_volume(close_wide, volume_wide).rolling(window, min_periods=max(2, window // 2)).mean()


def align(*frames: pd.DataFrame) -> tuple:
    """Restrict every frame to the shared dates and tickers, in the same order.

    Misaligned frames are a silent source of wrong backtest numbers, so every
    entry point into the engine goes through this.
    """
    frames = [f for f in frames if f is not None]
    if not frames:
        return ()

    idx = frames[0].index
    cols = frames[0].columns
    for f in frames[1:]:
        idx = idx.intersection(f.index)
        cols = cols.intersection(f.columns)

    idx = idx.sort_values()
    cols = cols.sort_values()
    return tuple(f.reindex(index=idx, columns=cols) for f in frames)


def winsorize_returns(returns: pd.DataFrame, limit: float = 0.5) -> pd.DataFrame:
    """Clip absurd single-day moves that are almost always bad data.

    A genuine +/-50% daily move exists, but in a hackathon universe an unclipped
    data error will dominate the backtest and produce a fake Sharpe.
    """
    return returns.clip(lower=-limit, upper=limit)


def coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows and date range per ticker. Used by the data-completeness filter."""
    g = panel.groupby("ticker")["date"]
    out = pd.DataFrame({
        "n_days": g.count(),
        "first_date": g.min(),
        "last_date": g.max(),
    })
    return out.reset_index()

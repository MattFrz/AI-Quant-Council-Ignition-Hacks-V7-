"""The scoreboard. Step B5 — built BEFORE any more factors get written."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from quant.factors.base import Factor, Panel
from quant.signals.cross_sectional import (
    align,
    factor_panel,
    forward_returns,
    normalize_panel,
    rebalance_dates,
)

MIN_NAMES = 10


@dataclass
class ICSummary:
    """What one factor scored. Everything a judge might ask, in one object."""

    factor: str
    horizon: int
    method: str
    n_periods: int
    mean_ic: float
    std_ic: float
    ir: float                  # mean / std — IC's own Sharpe
    t_stat: float
    p_value: float
    hit_rate: float            # fraction of periods with IC > 0
    overlapping: bool = False  # if True the t-stat is optimistic

    def is_significant(self, alpha: float = 0.05) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < alpha)

    def as_row(self) -> dict:
        return {
            "factor": self.factor,
            "horizon": self.horizon,
            "method": self.method,
            "n": self.n_periods,
            "mean_ic": self.mean_ic,
            "std_ic": self.std_ic,
            "ir": self.ir,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "hit_rate": self.hit_rate,
            "overlapping": self.overlapping,
        }

    def __str__(self) -> str:
        flag = "  [OVERLAPPING - t-stat optimistic]" if self.overlapping else ""
        return (
            f"{self.factor:<26} h={self.horizon:<3} {self.method:<8} "
            f"IC={self.mean_ic:+.4f}  IR={self.ir:+.3f}  "
            f"t={self.t_stat:+.2f}  p={self.p_value:.4f}  "
            f"hit={self.hit_rate:.1%}  n={self.n_periods}{flag}"
        )


def cross_sectional_ic(
    factor_df: pd.DataFrame,
    fwd_df: pd.DataFrame,
    method: str = "pearson",
    min_names: int = MIN_NAMES,
) -> pd.Series:
    """IC per date. One correlation across tickers for each row."""
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")

    f, r = align(factor_df, fwd_df)
    out = {}
    for d in f.index:
        pair = pd.concat([f.loc[d], r.loc[d]], axis=1, keys=["f", "r"]).dropna()
        if len(pair) < min_names or pair["f"].nunique() < 2 or pair["r"].nunique() < 2:
            continue
        if method == "pearson":
            c, _ = stats.pearsonr(pair["f"], pair["r"])
        else:
            c, _ = stats.spearmanr(pair["f"], pair["r"])
        if np.isfinite(c):
            out[d] = c

    return pd.Series(out, dtype=float).sort_index()


def summarize_ic(
    ic: pd.Series,
    factor: str,
    horizon: int,
    method: str = "pearson",
) -> ICSummary:
    """Collapse an IC time series into the scoreboard row."""
    ic = ic.dropna()
    n = len(ic)
    if n < 2:
        return ICSummary(factor, horizon, method, n, np.nan, np.nan,
                         np.nan, np.nan, np.nan, np.nan)

    mean = float(ic.mean())
    std = float(ic.std(ddof=1))
    ir = mean / std if std > 0 else np.nan
    t = ir * np.sqrt(n) if np.isfinite(ir) else np.nan
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if np.isfinite(t) else np.nan

    overlapping = False
    if n >= 2 and isinstance(ic.index, pd.DatetimeIndex):
        spacing = np.median(np.diff(ic.index.values).astype("timedelta64[D]").astype(int))
        overlapping = bool(spacing < horizon * (7 / 5))  # horizon is trading days

    return ICSummary(
        factor=factor, horizon=horizon, method=method, n_periods=n,
        mean_ic=mean, std_ic=std, ir=float(ir), t_stat=float(t),
        p_value=p, hit_rate=float((ic > 0).mean()), overlapping=overlapping,
    )


def information_coefficient(
    factor_df: pd.DataFrame,
    panel: Panel,
    horizon: int = 21,
    method: str = "pearson",
    name: str = "factor",
) -> ICSummary:
    """Score a ready-made factor panel end to end."""
    fwd = forward_returns(panel, horizon=horizon, dates=factor_df.index)
    ic = cross_sectional_ic(factor_df, fwd, method=method)
    return summarize_ic(ic, factor=name, horizon=horizon, method=method)


def rank_ic(factor_df: pd.DataFrame, panel: Panel, horizon: int = 21, name: str = "factor") -> ICSummary:
    """Spearman IC. More robust than Pearson, and usually the one to quote."""
    return information_coefficient(factor_df, panel, horizon, method="spearman", name=name)


def signal_decay(
    factor_df: pd.DataFrame,
    panel: Panel,
    horizons: Sequence[int] = (1, 5, 10, 21, 63),
    method: str = "spearman",
    name: str = "factor",
) -> pd.DataFrame:
    """IC at several horizons — how fast the edge dies."""
    rows = [information_coefficient(factor_df, panel, h, method, name).as_row() for h in horizons]
    return pd.DataFrame(rows).set_index("horizon")


def factor_scoreboard(
    factors: Iterable[Factor],
    panel: Panel,
    horizon: int = 21,
    method: str = "spearman",
    dates: Optional[Sequence] = None,
    normalize_first: bool = True,
) -> pd.DataFrame:
    """Every factor, one row each, sorted by |IC|. The B5 deliverable."""
    factors = list(factors)
    if dates is None:
        longest = max((f.required_history for f in factors), default=0)
        dates = rebalance_dates(panel, warmup=longest)

    rows = []
    for f in factors:
        raw = factor_panel(f, panel, dates)
        scores = normalize_panel(raw) if normalize_first else raw
        summary = information_coefficient(scores, panel, horizon, method, name=f.name)
        row = summary.as_row()
        row["category"] = f.category.value
        rows.append(row)

    board = pd.DataFrame(rows)
    if board.empty:
        return board
    return board.reindex(board["mean_ic"].abs().sort_values(ascending=False).index).reset_index(drop=True)

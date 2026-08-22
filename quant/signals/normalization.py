"""Cross-sectional normalization. Step B3.

Every factor exits through here before it is combined. All functions take ONE
cross-section — the values for every ticker on a single date — and return a
Series on the same index, NaN preserved where the input was NaN.

Winsorize before z-scoring, never after. A single bad print (a missed split, a
100x price) moves the mean and inflates the standard deviation, which quietly
compresses every other name toward zero. Clipping first costs nothing and stops
one bad row from flattening a whole date.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

MIN_OBS = 5


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip to the given quantiles of the cross-section."""
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(f"require 0 <= lower < upper <= 1, got ({lower}, {upper})")
    valid = s.dropna()
    if valid.empty:
        return s.astype(float)
    lo, hi = valid.quantile(lower), valid.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def winsorize_mad(s: pd.Series, n_mad: float = 5.0) -> pd.Series:
    """Clip at n median-absolute-deviations from the median.

    Better than quantile clipping when the tail is a handful of names rather
    than a fixed fraction of the universe.
    """
    valid = s.dropna()
    if valid.empty:
        return s.astype(float)
    med = valid.median()
    mad = (valid - med).abs().median()
    if mad == 0 or not np.isfinite(mad):
        return s.astype(float)
    spread = n_mad * 1.4826 * mad  # 1.4826 puts MAD on the same scale as sigma
    return s.clip(lower=med - spread, upper=med + spread)


def zscore(s: pd.Series, robust: bool = False, min_obs: int = MIN_OBS) -> pd.Series:
    """Standardize the cross-section to mean 0, sd 1.

    Returns all-NaN when the date is too thin or has no dispersion, rather than
    dividing by ~0 and emitting enormous scores. A date with no information
    should contribute nothing, not noise amplified to look like conviction.
    """
    valid = s.dropna()
    if len(valid) < min_obs:
        return pd.Series(np.nan, index=s.index, name=s.name, dtype=float)

    if robust:
        center = valid.median()
        mad = (valid - center).abs().median()
        scale = 1.4826 * mad
    else:
        center = valid.mean()
        scale = valid.std(ddof=1)

    if scale == 0 or not np.isfinite(scale):
        return pd.Series(np.nan, index=s.index, name=s.name, dtype=float)

    return ((s - center) / scale).astype(float)


def rank_transform(s: pd.Series, pct: bool = True) -> pd.Series:
    """Rank across the cross-section. `pct=True` maps to (0, 1].

    Throws away magnitude and keeps order, which is what you want from a factor
    whose tails you do not trust.
    """
    return s.rank(pct=pct, na_option="keep").astype(float)


def to_percentile(s: pd.Series) -> pd.Series:
    """Rank on 0-100, matching FactorValue.percentile in the frozen schema."""
    return rank_transform(s, pct=True) * 100.0


def normalize(
    s: pd.Series,
    winsor: Optional[Tuple[float, float]] = (0.01, 0.99),
    robust: bool = False,
    min_obs: int = MIN_OBS,
) -> pd.Series:
    """The standard pipeline: winsorize, then z-score. This is the default path
    every factor takes on its way into the composite."""
    out = winsorize(s, *winsor) if winsor is not None else s
    return zscore(out, robust=robust, min_obs=min_obs)


def rank_normalize(s: pd.Series, min_obs: int = MIN_OBS) -> pd.Series:
    """Rank first, then z-score the ranks — a uniform mapped to roughly normal.

    Immune to outliers entirely. Use where a factor has a long tail that
    winsorizing only half-fixes, e.g. valuation ratios near zero earnings.
    """
    valid = s.dropna()
    if len(valid) < min_obs:
        return pd.Series(np.nan, index=s.index, name=s.name, dtype=float)
    return zscore(rank_transform(s, pct=True), min_obs=min_obs)

"""Correlation analysis. Step A18a.

Used by portfolio construction to avoid stacking the same bet under different
tickers - the most common way a "diversified" book turns out to hold one
position in five names.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def correlation_matrix(returns: pd.DataFrame, min_periods: int = 60) -> pd.DataFrame:
    return returns.corr(min_periods=min_periods)


def average_pairwise_correlation(returns: pd.DataFrame) -> Optional[float]:
    """Mean off-diagonal correlation. High values mean the book is one bet."""
    if returns is None or returns.shape[1] < 2:
        return None
    corr = correlation_matrix(returns)
    mask = ~np.eye(len(corr), dtype=bool)
    vals = corr.to_numpy()[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else None


def most_correlated_pairs(returns: pd.DataFrame, top_n: int = 5) -> List[tuple]:
    """Highest-correlation pairs, for the risk panel to call out by name."""
    if returns is None or returns.shape[1] < 2:
        return []
    corr = correlation_matrix(returns)
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if np.isfinite(val):
                pairs.append((cols[i], cols[j], float(val)))
    return sorted(pairs, key=lambda p: abs(p[2]), reverse=True)[:top_n]


def diversification_ratio(weights: pd.Series, returns: pd.DataFrame) -> Optional[float]:
    """Weighted average vol / portfolio vol. 1.0 = no diversification benefit."""
    common = weights.index.intersection(returns.columns)
    if len(common) < 2:
        return None

    w = weights[common].to_numpy()
    sub = returns[common].dropna(how="all")
    if len(sub) < 30:
        return None

    vols = sub.std(ddof=1).to_numpy()
    cov = sub.cov().to_numpy()
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return None
    return float((w @ vols) / np.sqrt(port_var))

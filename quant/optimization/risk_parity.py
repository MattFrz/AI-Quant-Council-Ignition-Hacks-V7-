"""Equal risk contribution sizing. Optional per the build order.

B13's inverse-vol sizing equalises risk only if the names are uncorrelated. Two
semiconductor suppliers at 5% each are not two 5% risks — they are closer to one
10% risk wearing two names, which is exactly the concentration the section 14
panel is supposed to catch.

This file does it properly: every position contributes an equal share of
portfolio variance, correlation included. It is a strictly better answer than
B13 and a strictly harder one to explain on a slide, which is why B13 remains
the default and this is the upgrade.

Also holds the covariance estimator both optimizers use, with shrinkage. On a
60-day window with 20 names the sample covariance is close to singular, and an
optimizer handed a near-singular matrix will happily produce enormous offsetting
positions that look precise and are noise.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from quant.factors.base import Panel

TRADING_DAYS = 252


def covariance_from_panel(
    panel: Panel,
    as_of,
    tickers: Sequence[str],
    window: int = 120,
    lag_days: int = 1,
    shrinkage: float = 0.2,
    annualize: bool = True,
) -> pd.DataFrame:
    """Shrunk sample covariance of daily returns, respecting the as-of rule.

    Shrinks toward a diagonal of the sample variances: keeps each name's own
    volatility, pulls the correlations toward zero. With a short window that is
    the part of the estimate you should not trust.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")

    hist = panel.as_of(as_of, lag_days)
    px = hist.adj_close.reindex(columns=list(tickers)).where(lambda d: d > 0)
    rets = np.log(px).diff().iloc[-window:].dropna(how="all")
    rets = rets.dropna(axis=1, how="any")

    if rets.shape[0] < 5 or rets.shape[1] == 0:
        n = len(tickers)
        return pd.DataFrame(np.eye(n) * 1e-4, index=list(tickers), columns=list(tickers))

    sample = rets.cov()
    target = pd.DataFrame(np.diag(np.diag(sample.values)),
                          index=sample.index, columns=sample.columns)
    cov = (1.0 - shrinkage) * sample + shrinkage * target
    return cov * TRADING_DAYS if annualize else cov


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each position's share of portfolio variance."""
    marginal = cov @ weights
    return weights * marginal


def equal_risk_contribution(
    cov: pd.DataFrame,
    max_iter: int = 2000,
    tol: float = 1e-10,
    long_only: bool = True,
) -> pd.Series:
    """Weights where every name contributes the same share of variance.

    Multiplicative fixed point with square-root damping: scale each weight by
    the square root of (target contribution / actual contribution) and
    renormalize. Damping matters — the undamped update oscillates on strongly
    correlated pairs instead of settling.
    """
    matrix = np.asarray(cov, dtype=float)
    n = matrix.shape[0]
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=cov.index)

    w = np.ones(n) / n
    for _ in range(max_iter):
        contrib = risk_contributions(w, matrix)
        total = contrib.sum()
        if total <= 0 or not np.isfinite(total):
            break
        target = total / n
        with np.errstate(divide="ignore", invalid="ignore"):
            adjustment = np.sqrt(np.where(contrib > 0, target / contrib, 1.0))
        w_new = w * adjustment
        if long_only:
            w_new = np.clip(w_new, 1e-12, None)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return pd.Series(w, index=cov.index, name="weight")


def risk_parity_book(
    alpha: pd.Series,
    panel: Panel,
    as_of,
    max_names: int = 10,
    max_position: float = 0.05,
    target_vol: float = 0.10,
    window: int = 120,
    shrinkage: float = 0.2,
) -> pd.DataFrame:
    """Top names by alpha, sized to equal risk contribution, capped and scaled."""
    scores = alpha.dropna()
    scores = scores[scores > 0].sort_values(ascending=False).head(max_names)
    if scores.empty:
        return pd.DataFrame(columns=["weight", "risk_contribution", "pct_of_risk"])

    cov = covariance_from_panel(panel, as_of, scores.index, window=window,
                                shrinkage=shrinkage)
    weights = equal_risk_contribution(cov)

    port_vol = float(np.sqrt(weights.values @ cov.values @ weights.values))
    if port_vol > 0:
        weights = weights * (target_vol / port_vol)
    weights = weights.clip(upper=max_position)

    contrib = risk_contributions(weights.values, np.asarray(cov, dtype=float))
    total = contrib.sum()
    return pd.DataFrame({
        "weight": weights,
        "risk_contribution": contrib,
        "pct_of_risk": contrib / total if total > 0 else np.nan,
    }).sort_values("weight", ascending=False)

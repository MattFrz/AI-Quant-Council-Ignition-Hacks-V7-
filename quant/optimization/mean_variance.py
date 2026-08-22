"""Constrained mean-variance optimization. Optional per the build order.

Section 13 prefers a transparent methodology over a clever one, and this file is
the clever one — included for completeness, not as the default. B13's vol
scaling is what should appear in the demo.

The reason to be wary is not that Markowitz is wrong, it is that MVO is a
maximiser of estimation error. Feed it noisy expected returns and it will find
whichever name's alpha was most overstated and put the whole book in it. Three
guards here push back on that:

  - expected returns come from cross-sectional alpha RANKS, not from a raw
    composite pretending to be a return forecast
  - covariance is shrunk (see risk_parity.covariance_from_panel)
  - position caps bind, and the risk-aversion term is high by default

If the optimizer's answer differs wildly from the equal-risk answer, trust the
equal-risk answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

from quant.factors.base import Panel
from quant.optimization.risk_parity import covariance_from_panel


@dataclass
class OptimizedBook:
    weights: pd.Series
    expected_return: float
    expected_vol: float
    sharpe_ex_ante: float
    converged: bool
    message: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"weight": self.weights}).sort_values(
            "weight", ascending=False
        )

    def __str__(self) -> str:
        status = "converged" if self.converged else f"DID NOT CONVERGE ({self.message})"
        return (
            f"{int((self.weights.abs() > 1e-6).sum())} positions, "
            f"E[r]={self.expected_return:.2%}, vol={self.expected_vol:.2%}, "
            f"ex-ante Sharpe={self.sharpe_ex_ante:.2f} — {status}"
        )


def alpha_to_expected_returns(
    alpha: pd.Series,
    spread: float = 0.12,
) -> pd.Series:
    """Turn composite alpha into a plausible return forecast via ranks.

    Deliberately crude. The composite is a score, not a percentage, and feeding
    it to an optimizer as if it were an annual return is how a backtest ends up
    claiming 200%. Ranks map to a fixed, stated spread — the best name in the
    book is assumed to beat the worst by `spread` a year, and nothing more
    precise is claimed.
    """
    scores = alpha.dropna()
    if scores.empty:
        return scores
    if len(scores) == 1:
        return pd.Series([spread / 2], index=scores.index)
    ranks = scores.rank(pct=True)
    return (ranks - 0.5) * spread


def mean_variance_weights(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = 8.0,
    max_position: float = 0.05,
    min_position: float = 0.0,
    budget: Optional[float] = 1.0,
) -> OptimizedBook:
    """Maximize w'mu - (lambda/2) w'Sigma w subject to caps and a budget."""
    tickers = list(expected_returns.index)
    mu = expected_returns.to_numpy(dtype=float)
    sigma = cov.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    n = len(tickers)
    if n == 0:
        empty = pd.Series(dtype=float)
        return OptimizedBook(empty, 0.0, 0.0, 0.0, False, "no candidates")

    def objective(w: np.ndarray) -> float:
        return -(w @ mu) + 0.5 * risk_aversion * (w @ sigma @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return -mu + risk_aversion * (sigma @ w)

    constraints = []
    if budget is not None:
        constraints.append({"type": "eq", "fun": lambda w: w.sum() - budget})

    result = optimize.minimize(
        objective,
        x0=np.full(n, (budget or 1.0) / n),
        jac=gradient,
        bounds=[(min_position, max_position)] * n,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-10},
    )

    w = pd.Series(result.x, index=tickers, name="weight")
    er = float(w.to_numpy() @ mu)
    vol = float(np.sqrt(max(w.to_numpy() @ sigma @ w.to_numpy(), 0.0)))
    return OptimizedBook(
        weights=w,
        expected_return=er,
        expected_vol=vol,
        sharpe_ex_ante=er / vol if vol > 0 else 0.0,
        converged=bool(result.success),
        message=str(result.message),
    )


def optimize_book(
    alpha: pd.Series,
    panel: Panel,
    as_of,
    max_names: int = 10,
    max_position: float = 0.05,
    risk_aversion: float = 8.0,
    window: int = 120,
    shrinkage: float = 0.2,
    spread: float = 0.12,
) -> OptimizedBook:
    """Alpha ranking -> mean-variance book, end to end."""
    scores = alpha.dropna()
    scores = scores[scores > 0].sort_values(ascending=False).head(max_names)
    if scores.empty:
        return OptimizedBook(pd.Series(dtype=float), 0.0, 0.0, 0.0, False, "no candidates")

    mu = alpha_to_expected_returns(scores, spread=spread)
    cov = covariance_from_panel(panel, as_of, scores.index, window=window,
                                shrinkage=shrinkage)
    # Cap total exposure at what the position limits can actually hold.
    budget = min(1.0, max_position * len(scores))
    return mean_variance_weights(
        mu, cov, risk_aversion=risk_aversion, max_position=max_position, budget=budget
    )

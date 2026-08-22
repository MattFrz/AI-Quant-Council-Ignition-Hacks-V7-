"""Volatility-scaled position sizing. Step B13.

The plan is explicit that a transparent methodology beats a clever one, so this
is deliberately the simplest thing that is still defensible: rank by alpha, size
inversely to volatility, cap the concentration, scale the book to a target.

Sizing inversely to volatility is the whole idea. Equal dollar weights across a
40%-vol name and a 15%-vol name is not a balanced book — the first position
drives nearly all the P&L and the ranking that picked them barely matters.
Equal RISK weights let the alpha ranking actually express itself.

`risk_parity.py` and `mean_variance.py` stay optional per the build order. This
file is the one that has to work.

Correlation is ignored here, which makes the portfolio-vol estimate an upper
bound rather than a forecast — real correlated names come in under it. That is
the honest direction to be wrong in, and the correlation step belongs to the
portfolio construction lane.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from quant.factors.base import Panel

TRADING_DAYS = 252


@dataclass
class SizedBook:
    """Positions plus the arithmetic that produced them."""

    weights: pd.Series               # fraction of portfolio per ticker
    realized_vol: pd.Series
    gross_exposure: float
    est_portfolio_vol: float
    target_vol: float
    max_position: float
    n_positions: int
    capped: list

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "weight": self.weights,
            "weight_pct": self.weights * 100.0,
            "realized_vol": self.realized_vol.reindex(self.weights.index),
            "risk_contribution": (self.weights.abs() *
                                  self.realized_vol.reindex(self.weights.index)),
        }).sort_values("weight", ascending=False)

    def __str__(self) -> str:
        note = f", {len(self.capped)} capped" if self.capped else ""
        return (
            f"{self.n_positions} positions, gross {self.gross_exposure:.1%}, "
            f"est vol {self.est_portfolio_vol:.1%} vs target {self.target_vol:.1%}{note}"
        )


def realized_vol(panel: Panel, as_of, window: int = 60, lag_days: int = 1) -> pd.Series:
    """Annualized volatility per ticker, respecting the as-of rule."""
    hist = panel.as_of(as_of, lag_days)
    px = hist.adj_close.where(hist.adj_close > 0).iloc[-(window + 1):]
    if len(px) < 2:
        return pd.Series(np.nan, index=panel.tickers, dtype=float)
    rets = np.log(px).diff().iloc[1:]
    return rets.std(ddof=1) * np.sqrt(TRADING_DAYS)


def size_positions(
    alpha: pd.Series,
    panel: Panel,
    as_of,
    target_vol: float = 0.10,
    max_position: float = 0.05,
    max_names: int = 10,
    vol_window: int = 60,
    min_vol: float = 0.05,
    long_only: bool = True,
) -> SizedBook:
    """Alpha ranking -> position sizes.

    `max_position` defaults to 5% to match MAX_POSITION_PCT in the config.
    Capped names have their excess redistributed across the rest, so the book
    still reaches its target exposure instead of quietly running light.
    """
    scores = alpha.dropna()
    if long_only:
        scores = scores[scores > 0]
    if scores.empty:
        empty = pd.Series(dtype=float)
        return SizedBook(empty, empty, 0.0, 0.0, target_vol, max_position, 0, [])

    scores = scores.reindex(scores.abs().sort_values(ascending=False).index).head(max_names)

    vols = realized_vol(panel, as_of, window=vol_window).reindex(scores.index)
    vols = vols.fillna(vols.median()).clip(lower=min_vol)

    # Inverse-vol, tilted by conviction.
    raw = scores.abs() / vols
    if raw.sum() == 0:
        raw = pd.Series(1.0, index=scores.index)
    weights = np.sign(scores) * (raw / raw.sum())

    # Scale the whole book so the (correlation-free) vol estimate hits target.
    book_vol = float(np.sqrt(((weights.abs() * vols) ** 2).sum()))
    if book_vol > 0:
        weights = weights * (target_vol / book_vol)

    # Cap, then push the excess back into the uncapped names.
    capped: list = []
    for _ in range(len(weights)):
        over = weights.abs() > max_position
        if not over.any():
            break
        capped = sorted(set(capped) | set(weights.index[over]))
        excess = float((weights.abs() - max_position)[over].sum())
        weights[over] = np.sign(weights[over]) * max_position
        free = ~weights.index.isin(capped)
        if not free.any() or excess <= 0:
            break
        room = weights[free].abs()
        weights[free] = weights[free] + np.sign(weights[free]) * excess * (room / room.sum())

    est_vol = float(np.sqrt(((weights.abs() * vols) ** 2).sum()))
    return SizedBook(
        weights=weights,
        realized_vol=vols,
        gross_exposure=float(weights.abs().sum()),
        est_portfolio_vol=est_vol,
        target_vol=target_vol,
        max_position=max_position,
        n_positions=int((weights.abs() > 1e-9).sum()),
        capped=capped,
    )

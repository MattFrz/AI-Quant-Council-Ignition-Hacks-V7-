"""Sector, concentration and factor exposures. Step A17.

Answers "what is this portfolio actually betting on?" - which is usually not
what the thesis says it is betting on.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def sector_exposure(weights: pd.Series, sectors: Dict[str, str]) -> Dict[str, float]:
    """Net weight per sector. Signed, so a long/short book can net to zero."""
    if weights is None or not sectors:
        return {}
    out: Dict[str, float] = {}
    for ticker, weight in weights.items():
        sector = sectors.get(ticker) or "Unknown"
        out[sector] = out.get(sector, 0.0) + float(weight)
    return {k: round(v, 6) for k, v in sorted(out.items(), key=lambda kv: -abs(kv[1]))}


def gross_sector_exposure(weights: pd.Series, sectors: Dict[str, str]) -> Dict[str, float]:
    """Absolute weight per sector - the real concentration picture."""
    if weights is None or not sectors:
        return {}
    out: Dict[str, float] = {}
    for ticker, weight in weights.items():
        sector = sectors.get(ticker) or "Unknown"
        out[sector] = out.get(sector, 0.0) + abs(float(weight))
    return {k: round(v, 6) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def concentration(weights: pd.Series) -> Optional[float]:
    """Largest single position weight."""
    if weights is None or weights.empty:
        return None
    return float(weights.abs().max())


def herfindahl(weights: pd.Series) -> Optional[float]:
    """Sum of squared weights. 1.0 = one position, 1/N = perfectly equal."""
    if weights is None or weights.empty:
        return None
    w = weights.abs()
    total = w.sum()
    if total == 0:
        return None
    return float(((w / total) ** 2).sum())


def effective_positions(weights: pd.Series) -> Optional[float]:
    """1 / Herfindahl. "You say 20 names, but you hold 4.2 positions."""
    h = herfindahl(weights)
    return None if not h else float(1.0 / h)


def factor_exposure(
    weights: pd.Series,
    factor_values: pd.DataFrame,
) -> Dict[str, float]:
    """Weighted average factor z-score of the book.

    factor_values: index=ticker, columns=factor name. Tells you the portfolio is
    short value or long momentum whether or not you intended it.
    """
    if weights is None or factor_values is None or factor_values.empty:
        return {}
    common = weights.index.intersection(factor_values.index)
    if len(common) == 0:
        return {}

    w = weights[common]
    total = w.abs().sum()
    if total == 0:
        return {}
    normalized = w / total

    exposures = factor_values.loc[common].mul(normalized, axis=0).sum()
    return {str(k): float(v) for k, v in exposures.items() if np.isfinite(v)}


def exposure_report(
    weights: pd.Series,
    sectors: Optional[Dict[str, str]] = None,
    factor_values: Optional[pd.DataFrame] = None,
) -> dict:
    """Everything the section 14 exposure block needs, in one call."""
    return {
        "sector_net": sector_exposure(weights, sectors or {}),
        "sector_gross": gross_sector_exposure(weights, sectors or {}),
        "concentration": concentration(weights),
        "herfindahl": herfindahl(weights),
        "effective_positions": effective_positions(weights),
        "factor": factor_exposure(weights, factor_values) if factor_values is not None else {},
        "n_positions": int((weights.abs() > 1e-6).sum()) if weights is not None else 0,
        "gross_exposure": float(weights.abs().sum()) if weights is not None else 0.0,
        "net_exposure": float(weights.sum()) if weights is not None else 0.0,
    }

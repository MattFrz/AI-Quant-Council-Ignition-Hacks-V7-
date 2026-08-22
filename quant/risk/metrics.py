"""Core risk metrics and the RiskMetrics assembler. Step A15.

Top of the section 14 panel: beta, volatility, max drawdown - then this module
pulls the other risk files together into the schema object TradeIdea carries.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from data.schemas.risk import RiskBand, RiskMetrics
from quant.backtest.metrics import beta as _beta
from quant.backtest.metrics import drawdown_curve, equity_curve, volatility as _vol
from quant.risk.correlation import average_pairwise_correlation
from quant.risk.exposures import concentration, sector_exposure
from quant.risk.liquidity import days_to_liquidate
from quant.risk.var import cvar_historical, var_historical

#: Annualized volatility thresholds separating the three risk bands.
BAND_THRESHOLDS = (0.18, 0.32)


def realized_volatility(returns: pd.Series) -> float:
    return _vol(returns)


def rolling_volatility(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Per-name annualized vol. Feeds the slippage model and position sizing."""
    return returns.rolling(window, min_periods=max(20, window // 3)).std(ddof=1) * np.sqrt(252)


def portfolio_beta(returns: pd.Series, benchmark: pd.Series) -> float:
    return _beta(returns, benchmark)


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_curve(equity_curve(returns)).min()) if len(returns) else 0.0


def risk_band(vol: Optional[float]) -> RiskBand:
    if vol is None or not np.isfinite(vol):
        return RiskBand.MEDIUM
    low, high = BAND_THRESHOLDS
    if vol < low:
        return RiskBand.LOW
    if vol > high:
        return RiskBand.HIGH
    return RiskBand.MEDIUM


def build_risk_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    weights: Optional[pd.Series] = None,
    sectors: Optional[Dict[str, str]] = None,
    asset_returns: Optional[pd.DataFrame] = None,
    position_notional: Optional[float] = None,
    adv_usd: Optional[float] = None,
    max_participation: float = 0.05,
) -> RiskMetrics:
    """Assemble the section 14 panel from whatever inputs are available.

    Every field is optional in the schema, so a partial input set produces a
    partial panel rather than a fabricated one.
    """
    vol = realized_volatility(returns) if len(returns) else None

    return RiskMetrics(
        beta=portfolio_beta(returns, benchmark_returns) if benchmark_returns is not None else None,
        volatility=vol,
        max_drawdown=max_drawdown(returns) if len(returns) else None,
        var_95=var_historical(returns, 0.95) if len(returns) else None,
        cvar_95=cvar_historical(returns, 0.95) if len(returns) else None,
        sector=_dominant_sector(weights, sectors),
        sector_exposure=sector_exposure(weights, sectors) if weights is not None and sectors else {},
        concentration=concentration(weights) if weights is not None else None,
        avg_correlation=(
            average_pairwise_correlation(asset_returns) if asset_returns is not None else None
        ),
        days_to_liquidate=(
            days_to_liquidate(position_notional, adv_usd, max_participation)
            if position_notional and adv_usd else None
        ),
        risk_band=risk_band(vol),
    )


def _dominant_sector(
    weights: Optional[pd.Series], sectors: Optional[Dict[str, str]]
) -> Optional[str]:
    if weights is None or not sectors:
        return None
    exposure = sector_exposure(weights, sectors)
    return max(exposure, key=exposure.get) if exposure else None

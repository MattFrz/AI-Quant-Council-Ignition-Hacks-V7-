"""Risk metrics for one position or portfolio.

NOTE: this file is not in the original section 21 tree - it was added because
TradeIdea needs somewhere to put the section 14 risk panel.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskMetrics(BaseModel):
    beta: Optional[float] = None
    volatility: Optional[float] = Field(None, description="Annualized, fraction")
    max_drawdown: Optional[float] = None

    var_95: Optional[float] = Field(None, description="1-day historical VaR, fraction")
    cvar_95: Optional[float] = None

    sector: Optional[str] = None
    sector_exposure: Dict[str, float] = Field(default_factory=dict)
    concentration: Optional[float] = Field(None, description="Largest position weight")
    avg_correlation: Optional[float] = None

    days_to_liquidate: Optional[float] = Field(
        None, description="At max_adv_participation of ADV"
    )

    risk_band: RiskBand = RiskBand.MEDIUM

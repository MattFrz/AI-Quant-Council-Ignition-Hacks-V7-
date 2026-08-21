"""THE convergence object. Step 1.8 of the data contract.

Every lane converges here:
  Zain   -> catalysts, bull_case, bear_case, pm_rationale
  Nalin  -> alpha
  Matt   -> backtest, risk
  Cecile -> renders the whole thing

If a lane's output does not fit in this object, that lane is building the wrong
thing. Change this file by agreement, never unilaterally.
"""
from __future__ import annotations

from datetime import date as Date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from data.schemas.backtest_result import BacktestResult
from data.schemas.catalyst import Catalyst
from data.schemas.risk import RiskMetrics
from data.schemas.signal import AlphaBreakdown


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"


class Verdict(str, Enum):
    """What the quant validator concluded. Set from real backtest output only."""

    SURVIVED = "survived"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class TradeIdea(BaseModel):
    idea_id: str
    ticker: str
    company_name: str
    side: Side
    as_of: Date = Field(..., description="Decision date. Nothing after this was visible.")

    # Headline numbers - all computed, never authored by the LLM.
    alpha_score: float = Field(..., ge=0.0, le=10.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    expected_alpha: Optional[float] = Field(None, description="Annualized fraction")
    position_size_pct: Optional[float] = Field(None, description="Percent of portfolio")

    # Evidence (Zain) - the section 3 audit trail
    catalysts: List[Catalyst] = Field(default_factory=list)

    # Quantification (Nalin)
    alpha: Optional[AlphaBreakdown] = None

    # Validation and risk (Matt)
    backtest: Optional[BacktestResult] = None
    risk: Optional[RiskMetrics] = None
    validator_verdict: Verdict = Verdict.INCONCLUSIVE

    # Reasoning (Zain)
    bull_case: str = ""
    bear_case: str = ""
    pm_rationale: str = ""

    def has_audit_trail(self) -> bool:
        """No idea ships without at least one clickable, cited catalyst."""
        return bool(self.catalysts) and all(c.source_url for c in self.catalysts)

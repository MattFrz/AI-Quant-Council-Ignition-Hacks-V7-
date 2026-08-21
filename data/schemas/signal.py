"""Factor values and their contribution to the composite. Step 1.6.

SignalContribution is exactly the breakdown table the UI renders (plan section 2).
Return the per-factor contributions, not just the total — the breakdown IS the
explainability story.
"""
from __future__ import annotations

from datetime import date as Date
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class FactorCategory(str, Enum):
    FUNDAMENTAL = "fundamental"
    EARNINGS_REVISION = "earnings_revision"
    EVENT = "event"
    NLP = "nlp"
    MOMENTUM = "momentum"
    RISK = "risk"
    VALUATION = "valuation"


class FactorValue(BaseModel):
    name: str = Field(..., description="e.g. 'momentum_12_1'")
    category: FactorCategory
    raw: float
    zscore: float = Field(..., description="Cross-sectional z-score on the as_of date")
    percentile: float = Field(..., ge=0.0, le=100.0)
    as_of: Date
    min_lag_days: int = Field(
        0, description="Days of staleness required on inputs. Enforced by leakage_guards."
    )


class SignalContribution(BaseModel):
    factor: str
    category: FactorCategory
    zscore: float
    weight: float = Field(..., description="Fitted on the TRAIN window only")
    contribution: float = Field(..., description="zscore * weight")


class AlphaBreakdown(BaseModel):
    """What quant/alpha/composite.py returns."""
    ticker: str
    as_of: Date
    contributions: List[SignalContribution] = Field(default_factory=list)
    composite_alpha: float
    alpha_score: float = Field(..., ge=0.0, le=10.0, description="Composite mapped to 0-10 for display")

    def check_sums(self, tolerance: float = 1e-6) -> bool:
        """The displayed total must equal the sum of the displayed parts."""
        return abs(sum(c.contribution for c in self.contributions) - self.composite_alpha) < tolerance

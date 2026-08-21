"""Daily price bars. Step 1.2 of the data contract."""
from __future__ import annotations

from datetime import date as Date
from typing import List, Optional

from pydantic import BaseModel, Field


class Bar(BaseModel):
    ticker: str
    date: Date
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: Optional[float] = Field(
        None, description="Split/dividend adjusted close. Use this for returns."
    )

    @property
    def dollar_volume(self) -> float:
        return self.close * self.volume


class PriceSeries(BaseModel):
    """Convenience wrapper. The quant engine works in pandas, not these objects —
    this exists for API responses and fixtures."""
    ticker: str
    bars: List[Bar] = Field(default_factory=list)

"""A tradeable name in the universe. Step 1.1 of the data contract."""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class Security(BaseModel):
    ticker: str = Field(..., description="Uppercase US listing symbol, e.g. NVDA")
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None

    market_cap: Optional[float] = Field(None, description="USD")
    adv_20d: Optional[float] = Field(None, description="20-day avg dollar volume, USD")

    # Survivorship: keep delisted names in the universe with is_active=False and a
    # last_date, rather than dropping them. Dropping them is how backtests lie.
    is_active: bool = True
    first_date: Optional[date] = None
    last_date: Optional[date] = None

    def __str__(self) -> str:
        return f"{self.ticker} ({self.name})"

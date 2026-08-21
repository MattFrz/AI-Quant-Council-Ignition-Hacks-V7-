"""Point-in-time fundamentals. Step 1.3 — the most important schema in the repo.

`period_end` is the fiscal period the numbers describe.
`report_date` is the date those numbers became public.

Every downstream join MUST key on report_date. Joining on period_end gives the
backtest knowledge of a quarter weeks before the market had it, which inflates
every metric and is the single most common way a hackathon backtest turns out
to be worthless.
"""
from __future__ import annotations

from datetime import date as Date
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class FundamentalSnapshot(BaseModel):
    ticker: str
    period_end: Date = Field(..., description="Fiscal period end the figures cover")
    report_date: Date = Field(..., description="Date the figures became PUBLIC")
    fiscal_period: str = Field("FY", description="Q1 | Q2 | Q3 | Q4 | FY")

    # Income statement
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    eps_diluted: Optional[float] = None

    # Margins (fractions, not percents: 0.42 not 42)
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None

    # Cash flow / returns
    free_cash_flow: Optional[float] = None
    capex: Optional[float] = None
    roic: Optional[float] = None

    # Balance sheet
    total_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    shares_diluted: Optional[float] = None

    source_url: Optional[str] = None

    @model_validator(mode="after")
    def _report_date_cannot_precede_period_end(self) -> "FundamentalSnapshot":
        if self.report_date < self.period_end:
            raise ValueError(
                f"{self.ticker}: report_date {self.report_date} precedes period_end "
                f"{self.period_end} — a company cannot publish results before the "
                f"period has ended. This is a look-ahead bug in the data pipeline."
            )
        return self

    def is_known_at(self, as_of: Date) -> bool:
        """The as-of test every factor must pass before using this row."""
        return self.report_date <= as_of

"""Fundamental factors, joined as-of report_date. Step B6."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from data.schemas.signal import FactorCategory
from quant.factors.base import Factor, Panel

QUARTERS_PER_YEAR = 4


def known_reports(window: Panel) -> pd.DataFrame:
    """Every fundamental row already public at the window's last date.

    Panel.as_of has already filtered on report_date, so this only has to pick
    the right rows out of what survived. Sorting by report_date rather than
    period_end matters: a 10-K restating an old quarter arrives late, and the
    market saw it late.
    """
    funds = window.fundamentals
    if funds is None or funds.empty:
        return pd.DataFrame()
    return funds.sort_values(["ticker", "report_date", "period_end"])


def latest_and_prior(
    window: Panel,
    lag_quarters: int = QUARTERS_PER_YEAR,
) -> tuple:
    """Per ticker: the newest known report, and the one `lag_quarters` earlier.

    Returns two frames indexed by ticker, aligned. Year-on-year by default,
    because quarterly figures are seasonal and a quarter-on-quarter comparison
    mostly measures the calendar.
    """
    funds = known_reports(window)
    if funds.empty:
        empty = pd.DataFrame(index=pd.Index([], name="ticker"))
        return empty, empty

    # de-duplicate restatements: keep the FIRST filing of each period, which is.
    first_filed = (
        funds.sort_values("report_date")
        .groupby(["ticker", "period_end"], as_index=False)
        .first()
    )
    ordered = first_filed.sort_values(["ticker", "period_end"])

    latest = ordered.groupby("ticker").nth(-1).set_index("ticker")
    prior = (
        ordered.groupby("ticker")
        .nth(-(lag_quarters + 1))
        .set_index("ticker")
        .reindex(latest.index)
    )
    return latest, prior


def _growth(latest: pd.Series, prior: pd.Series) -> pd.Series:
    """Year-on-year growth, guarding the sign flip."""
    denom = prior.abs().where(prior.abs() > 0)
    return (latest - prior) / denom


class _FundamentalFactor(Factor):
    """Shared plumbing: every subclass sees the as-of joined snapshots."""

    category = FactorCategory.FUNDAMENTAL
    min_lag_days = 1
    required_history = 1

    def _compute(self, window: Panel) -> pd.Series:
        latest, prior = latest_and_prior(window)
        if latest.empty:
            return pd.Series(np.nan, index=window.tickers, dtype=float)
        return self._from_snapshots(window, latest, prior)

    def _from_snapshots(self, window, latest, prior) -> pd.Series:
        raise NotImplementedError


class RevenueGrowth(_FundamentalFactor):
    """Year-on-year revenue growth."""

    def __init__(self, min_lag_days: int = 1) -> None:
        self.min_lag_days = min_lag_days
        self.name = "revenue_growth_yoy"

    def _from_snapshots(self, window, latest, prior) -> pd.Series:
        return _growth(latest["revenue"], prior["revenue"])


class EPSGrowth(_FundamentalFactor):
    """Year-on-year diluted EPS growth.

    Stands in for the plan's 'EPS revision', which needs analyst estimates that
    no free source provides. Growth is what is actually computable, and it is
    stated as growth rather than dressed up as a revision.
    """

    category = FactorCategory.EARNINGS_REVISION

    def __init__(self, min_lag_days: int = 1) -> None:
        self.min_lag_days = min_lag_days
        self.name = "eps_growth_yoy"

    def _from_snapshots(self, window, latest, prior) -> pd.Series:
        return _growth(latest["eps_diluted"], prior["eps_diluted"])


class MarginExpansion(_FundamentalFactor):
    """Change in operating margin, in percentage points."""

    def __init__(self, min_lag_days: int = 1) -> None:
        self.min_lag_days = min_lag_days
        self.name = "margin_expansion_yoy"

    def _from_snapshots(self, window, latest, prior) -> pd.Series:
        return latest["operating_margin"] - prior["operating_margin"]


class FreeCashFlowYield(_FundamentalFactor):
    """Trailing free cash flow over point-in-time market cap.

    Market cap is price on the decision date times the share count from the
    last filing - never `securities.market_cap`, which is today's value and
    would leak the future into every historical date.
    """

    category = FactorCategory.VALUATION

    def __init__(self, min_lag_days: int = 1) -> None:
        self.min_lag_days = min_lag_days
        self.name = "fcf_yield"

    def _from_snapshots(self, window, latest, prior) -> pd.Series:
        price = window.adj_close.iloc[-1].reindex(latest.index)
        market_cap = (price * latest["shares_diluted"]).where(lambda s: s > 0)
        return (latest["free_cash_flow"] * QUARTERS_PER_YEAR) / market_cap


class EarningsYield(_FundamentalFactor):
    """Trailing earnings over point-in-time market cap - inverted P/E."""

    category = FactorCategory.VALUATION

    def __init__(self, min_lag_days: int = 1) -> None:
        self.min_lag_days = min_lag_days
        self.name = "earnings_yield"

    def _from_snapshots(self, window, latest, prior) -> pd.Series:
        price = window.adj_close.iloc[-1].reindex(latest.index).where(lambda s: s > 0)
        return (latest["eps_diluted"] * QUARTERS_PER_YEAR) / price


class ReturnOnInvestedCapital(_FundamentalFactor):
    """ROIC as filed."""

    def __init__(self, min_lag_days: int = 1) -> None:
        self.min_lag_days = min_lag_days
        self.name = "roic"

    def _from_snapshots(self, window, latest, prior) -> pd.Series:
        return latest["roic"].astype(float)


class NetDebtToMarketCap(_FundamentalFactor):
    """Net debt over point-in-time market cap. Higher = more levered."""

    category = FactorCategory.RISK

    def __init__(self, min_lag_days: int = 1) -> None:
        self.min_lag_days = min_lag_days
        self.name = "net_debt_to_mktcap"

    def _from_snapshots(self, window, latest, prior) -> pd.Series:
        price = window.adj_close.iloc[-1].reindex(latest.index)
        market_cap = (price * latest["shares_diluted"]).where(lambda s: s > 0)
        net_debt = latest["total_debt"] - latest["cash_and_equivalents"]
        return net_debt / market_cap


def default_fundamental_factors() -> List[Factor]:
    """The B6 set."""
    return [
        RevenueGrowth(),
        EPSGrowth(),
        MarginExpansion(),
        FreeCashFlowYield(),
        EarningsYield(),
        ReturnOnInvestedCapital(),
        NetDebtToMarketCap(),
    ]

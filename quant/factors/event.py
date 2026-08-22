"""Event factors: earnings surprise and drift. Step B7."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from data.schemas.catalyst import Catalyst, Direction
from data.schemas.signal import FactorCategory
from quant.factors.base import Factor, Panel
from quant.factors.fundamental import QUARTERS_PER_YEAR, known_reports

MIN_HISTORY_QUARTERS = 6


def first_filed(window: Panel) -> pd.DataFrame:
    """One row per (ticker, period_end): the ORIGINAL filing, not a restatement."""
    funds = known_reports(window)
    if funds.empty:
        return funds
    return (
        funds.sort_values("report_date")
        .groupby(["ticker", "period_end"], as_index=False)
        .first()
        .sort_values(["ticker", "period_end"])
    )


def yoy_changes(reports: pd.DataFrame, column: str) -> pd.DataFrame:
    """Year-on-year change of `column`, per ticker, quarter by quarter."""
    out = reports[["ticker", "period_end", "report_date", column]].copy()
    out["yoy"] = out.groupby("ticker")[column].diff(QUARTERS_PER_YEAR)
    return out.dropna(subset=["yoy"])


class _SurpriseFactor(Factor):
    """Standardized unexpected value: this year-on-year change against the
    typical size of past ones.

    This is the plan's 'earnings surprise' built without analyst estimates,
    which no free data source provides. The expectation is a seasonal random
    walk - the market expects roughly last year's quarter - and the surprise is
    scaled by how variable that company's changes normally are, so a steady
    compounder and a cyclical are not judged on the same scale.
    """

    category = FactorCategory.EVENT
    min_lag_days = 1
    required_history = 1
    column = "eps_diluted"

    def __init__(self, min_history: int = MIN_HISTORY_QUARTERS, min_lag_days: int = 1) -> None:
        self.min_history = min_history
        self.min_lag_days = min_lag_days

    def _compute(self, window: Panel) -> pd.Series:
        reports = first_filed(window)
        if reports.empty:
            return pd.Series(np.nan, index=window.tickers, dtype=float)

        changes = yoy_changes(reports, self.column)
        if changes.empty:
            return pd.Series(np.nan, index=window.tickers, dtype=float)

        scores = {}
        for ticker, group in changes.groupby("ticker"):
            if len(group) < self.min_history:
                continue
            history = group["yoy"].to_numpy(dtype=float)
            latest, past = history[-1], history[:-1]
            spread = past.std(ddof=1)
            if not np.isfinite(spread) or spread == 0:
                continue
            scores[ticker] = float(latest / spread)

        if not scores:
            return pd.Series(np.nan, index=window.tickers, dtype=float)
        return pd.Series(scores, dtype=float).reindex(window.tickers)


class StandardizedUnexpectedEarnings(_SurpriseFactor):
    """SUE on diluted EPS."""

    column = "eps_diluted"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "sue_eps"


class RevenueSurprise(_SurpriseFactor):
    """SUE on revenue. Harder to manage than EPS, so often the cleaner signal."""

    column = "revenue"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "sue_revenue"


class PostEarningsDrift(Factor):
    """Return since the last earnings report, while the report is still recent.

    Post-earnings announcement drift is one of the most durable anomalies on
    record: prices keep moving in the direction of the surprise for weeks. The
    factor is deliberately NaN once a report is stale - after the drift window
    closes this is just momentum wearing an earnings label.
    """

    category = FactorCategory.EVENT

    def __init__(
        self,
        max_age_days: int = 60,
        min_age_days: int = 2,
        min_lag_days: int = 1,
    ) -> None:
        self.max_age_days = max_age_days
        self.min_age_days = min_age_days
        self.min_lag_days = min_lag_days
        self.required_history = 2
        self.name = f"pead_{max_age_days}d"

    def _compute(self, window: Panel) -> pd.Series:
        reports = first_filed(window)
        if reports.empty:
            return pd.Series(np.nan, index=window.tickers, dtype=float)

        as_of = window.dates[-1]
        latest = reports.groupby("ticker")["report_date"].max()
        px = window.adj_close.where(window.adj_close > 0)

        scores = {}
        for ticker, report_date in latest.items():
            if ticker not in px.columns:
                continue
            age = (as_of - pd.Timestamp(report_date)).days
            if age < self.min_age_days or age > self.max_age_days:
                continue
            pos = int(px.index.searchsorted(pd.Timestamp(report_date), side="left"))
            if pos >= len(px.index):
                continue
            start, end = px[ticker].iloc[pos], px[ticker].iloc[-1]
            if pd.isna(start) or pd.isna(end) or start <= 0:
                continue
            scores[ticker] = float(end / start - 1.0)

        if not scores:
            return pd.Series(np.nan, index=window.tickers, dtype=float)
        return pd.Series(scores, dtype=float).reindex(window.tickers)


class EarningsAcceleration(Factor):
    """Is growth speeding up? This quarter's YoY growth minus last quarter's."""

    category = FactorCategory.EVENT

    def __init__(self, min_lag_days: int = 1) -> None:
        self.min_lag_days = min_lag_days
        self.required_history = 1
        self.name = "earnings_acceleration"

    def _compute(self, window: Panel) -> pd.Series:
        reports = first_filed(window)
        if reports.empty:
            return pd.Series(np.nan, index=window.tickers, dtype=float)

        frame = reports[["ticker", "period_end", "eps_diluted"]].copy()
        prior = frame.groupby("ticker")["eps_diluted"].shift(QUARTERS_PER_YEAR)
        denom = prior.abs().where(prior.abs() > 0)
        frame["growth"] = (frame["eps_diluted"] - prior) / denom
        frame["accel"] = frame.groupby("ticker")["growth"].diff()

        latest = frame.dropna(subset=["accel"]).groupby("ticker")["accel"].last()
        if latest.empty:
            return pd.Series(np.nan, index=window.tickers, dtype=float)
        return latest.reindex(window.tickers).astype(float)


# --------------------------------------------------------------- catalyst-driven

DIRECTION_SIGN = {Direction.BULLISH: 1.0, Direction.BEARISH: -1.0, Direction.NEUTRAL: 0.0}


class _CatalystDrivenFactor(Factor):
    """Scores catalysts whose headline matches this factor's keywords.

    Guidance changes and corporate announcements are events, but they only
    exist as text, so they arrive from Zain's C15 rather than from a filing's
    numbers. Same contract as B8: ships inert, one call to go live, and NaN
    rather than 0.0 where nothing is known.
    """

    category = FactorCategory.EVENT
    keywords: tuple = ()

    def __init__(
        self,
        catalysts: Optional[List[Catalyst]] = None,
        half_life_days: int = 45,
        max_age_days: int = 180,
        min_lag_days: int = 1,
        enabled: Optional[bool] = None,
    ) -> None:
        self.catalysts = list(catalysts or [])
        self.half_life_days = half_life_days
        self.max_age_days = max_age_days
        self.min_lag_days = min_lag_days
        self.required_history = 1
        self.enabled = bool(self.catalysts) if enabled is None else bool(enabled)

    @property
    def is_stubbed(self) -> bool:
        return not self.enabled or not self.catalysts

    def _matches(self, catalyst: Catalyst) -> bool:
        text = f"{catalyst.headline}".lower()
        return any(k in text for k in self.keywords)

    def _compute(self, window: Panel) -> pd.Series:
        if self.is_stubbed:
            return pd.Series(np.nan, index=window.tickers, dtype=float)

        as_of = window.dates[-1].date()
        scores = {}
        for catalyst in self.catalysts:
            if not catalyst.is_known_at(as_of) or not self._matches(catalyst):
                continue
            age = (as_of - catalyst.source_date).days
            if age < 0 or age > self.max_age_days:
                continue
            decay = 0.5 ** (age / self.half_life_days)
            sign = DIRECTION_SIGN.get(catalyst.direction, 0.0)
            scores[catalyst.ticker] = scores.get(catalyst.ticker, 0.0) + sign * catalyst.confidence * decay

        if not scores:
            return pd.Series(np.nan, index=window.tickers, dtype=float)
        return pd.Series(scores, dtype=float).reindex(window.tickers)

    def with_catalysts(self, catalysts: List[Catalyst]) -> "_CatalystDrivenFactor":
        return type(self)(
            catalysts=catalysts, half_life_days=self.half_life_days,
            max_age_days=self.max_age_days, min_lag_days=self.min_lag_days, enabled=True,
        )


class GuidanceChange(_CatalystDrivenFactor):
    """Management raising or cutting guidance. WAITS ON ZAIN'S C15."""

    keywords = ("guidance", "outlook", "forecast", "raises", "lowers", "cuts")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "guidance_change"


class CorporateAnnouncements(_CatalystDrivenFactor):
    """Buybacks, dividends, M&A, capex commitments. WAITS ON ZAIN'S C15."""

    keywords = ("buyback", "repurchase", "dividend", "acquisition", "merger",
                "capex", "capital expenditure", "expansion", "restructuring")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "corporate_announcements"


def default_event_factors(catalysts: Optional[List[Catalyst]] = None) -> List[Factor]:
    """The B7 set. The last two ship inert until C15 lands."""
    return [
        StandardizedUnexpectedEarnings(),
        RevenueSurprise(),
        PostEarningsDrift(),
        EarningsAcceleration(),
        GuidanceChange(catalysts=catalysts),
        CorporateAnnouncements(catalysts=catalysts),
    ]

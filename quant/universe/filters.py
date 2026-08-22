"""Universe filters. Step A5.

Four filters from section 10: market cap, average dollar volume, listing status,
data availability. Each returns the survivors AND a count, because the counts
are what the UI funnel animates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import pandas as pd

from backend.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class FilterResult:
    label: str
    passed: List[str]
    description: str = ""
    dropped: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.passed)


def filter_market_cap(
    profiles: pd.DataFrame, min_market_cap: float, universe: Optional[Sequence[str]] = None
) -> FilterResult:
    df = _restrict(profiles, universe)
    keep = df[df["market_cap"].fillna(0) >= min_market_cap]["ticker"].tolist()
    return FilterResult(
        label="Market cap",
        passed=keep,
        description=f"market cap >= ${min_market_cap/1e9:.1f}B",
        dropped=sorted(set(df["ticker"]) - set(keep)),
    )


def filter_liquidity(
    adv: pd.DataFrame, min_adv_usd: float, universe: Optional[Sequence[str]] = None
) -> FilterResult:
    """adv: wide frame of average dollar volume (date x ticker).

    Uses the most recent observation per ticker. Names that never traded enough
    are dropped, because a signal on an illiquid name is untradeable and the
    backtest would credit us with fills we could never get.
    """
    cols = [c for c in adv.columns if universe is None or c in set(universe)]
    latest = adv[cols].ffill().iloc[-1] if len(adv) else pd.Series(dtype=float)
    keep = latest[latest.fillna(0) >= min_adv_usd].index.tolist()
    return FilterResult(
        label="Liquidity",
        passed=keep,
        description=f"20-day ADV >= ${min_adv_usd/1e6:.1f}M",
        dropped=sorted(set(cols) - set(keep)),
    )


def filter_listing(
    profiles: pd.DataFrame,
    allowed_exchanges: Sequence[str] = ("NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS"),
    universe: Optional[Sequence[str]] = None,
) -> FilterResult:
    """Keep primary US listings. Yahoo exchange codes: NMS/NGM/NCM = Nasdaq
    tiers, NYQ = NYSE, ASE = NYSE American, PCX = NYSE Arca, BTS = Cboe BZX."""
    df = _restrict(profiles, universe)
    allowed = {e.upper() for e in allowed_exchanges}
    keep = df[df["exchange"].astype(str).str.upper().isin(allowed)]["ticker"].tolist()
    return FilterResult(
        label="Listing",
        passed=keep,
        description="primary US listing",
        dropped=sorted(set(df["ticker"]) - set(keep)),
    )


def filter_data_completeness(
    coverage_df: pd.DataFrame,
    min_days: int = 500,
    universe: Optional[Sequence[str]] = None,
) -> FilterResult:
    """Require enough history to compute a 12-month momentum factor and still
    leave a test window. Thin history is the quiet cause of NaN-heavy factors."""
    df = _restrict(coverage_df, universe)
    keep = df[df["n_days"] >= min_days]["ticker"].tolist()
    return FilterResult(
        label="Data availability",
        passed=keep,
        description=f">= {min_days} trading days of history",
        dropped=sorted(set(df["ticker"]) - set(keep)),
    )


def _restrict(df: pd.DataFrame, universe: Optional[Sequence[str]]) -> pd.DataFrame:
    if universe is None:
        return df
    return df[df["ticker"].isin(set(universe))]

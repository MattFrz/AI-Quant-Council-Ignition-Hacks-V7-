"""Build the tradeable universe and the funnel the UI animates. Step A6.

Returns both the surviving tickers and the per-stage counts, because section 15
renders exactly those counts:

    1,247 companies scanned
    -> 183 passed liquidity filter
    -> 74 passed fundamental filter
    ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

from backend.config import settings
from backend.core.logging import get_logger
from data.pipelines.prices import average_dollar_volume, coverage, to_wide
from quant.universe.filters import (
    FilterResult,
    filter_data_completeness,
    filter_liquidity,
    filter_listing,
    filter_market_cap,
)

log = get_logger(__name__)


@dataclass
class UniverseResult:
    tickers: List[str]
    stages: List[FilterResult]
    scanned: int

    def funnel(self) -> List[dict]:
        """Shape matches backend.api.schemas.FunnelStage."""
        rows = [{"label": "Scanned", "count": self.scanned, "description": "all names with price data"}]
        rows += [
            {"label": s.label, "count": s.count, "description": s.description}
            for s in self.stages
        ]
        return rows

    def summary(self) -> str:
        parts = [f"{self.scanned} scanned"]
        parts += [f"{s.count} {s.label.lower()}" for s in self.stages]
        return " -> ".join(parts)


def build_universe(
    panel: pd.DataFrame,
    profiles: pd.DataFrame,
    min_market_cap: Optional[float] = None,
    min_adv_usd: Optional[float] = None,
    min_days: int = 500,
    max_size: Optional[int] = None,
) -> UniverseResult:
    """Apply the four filters in order, tracking survivors at each stage.

    Order matters for the story the funnel tells: listing, then liquidity, then
    size, then data quality. Each stage runs only on the previous survivors.
    """
    min_market_cap = min_market_cap if min_market_cap is not None else settings.min_market_cap
    min_adv_usd = min_adv_usd if min_adv_usd is not None else settings.min_adv_usd
    max_size = max_size or settings.universe_size

    all_tickers = sorted(panel["ticker"].unique().tolist())
    scanned = len(all_tickers)

    close = to_wide(panel, "adj_close")
    volume = to_wide(panel, "volume")
    adv = average_dollar_volume(close, volume, window=20)
    cov = coverage(panel)

    stages: List[FilterResult] = []
    current: Sequence[str] = all_tickers

    for stage in (
        filter_listing(profiles, universe=current),
    ):
        stages.append(stage)
        current = stage.passed

    stage = filter_liquidity(adv, min_adv_usd, universe=current)
    stages.append(stage)
    current = stage.passed

    stage = filter_market_cap(profiles, min_market_cap, universe=current)
    stages.append(stage)
    current = stage.passed

    stage = filter_data_completeness(cov, min_days=min_days, universe=current)
    stages.append(stage)
    current = stage.passed

    # Cap the universe by liquidity so we keep the most tradeable names.
    tickers = list(current)
    if max_size and len(tickers) > max_size:
        latest_adv = adv[[t for t in tickers if t in adv.columns]].ffill().iloc[-1]
        tickers = latest_adv.sort_values(ascending=False).head(max_size).index.tolist()
        stages.append(
            FilterResult(
                label="Top by liquidity",
                passed=tickers,
                description=f"most liquid {max_size} names",
            )
        )

    result = UniverseResult(tickers=sorted(tickers), stages=stages, scanned=scanned)
    log.info("universe: %s", result.summary())
    return result

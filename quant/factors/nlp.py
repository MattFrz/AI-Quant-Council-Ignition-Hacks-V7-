"""Catalyst and sentiment factor. Step B8 - WAITS ON ZAIN'S C15."""
from __future__ import annotations

from datetime import date as Date
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from data.schemas.catalyst import Catalyst, Direction
from data.schemas.signal import FactorCategory
from quant.factors.base import Factor, Panel

DIRECTION_SIGN: Dict[Direction, float] = {
    Direction.BULLISH: 1.0,
    Direction.BEARISH: -1.0,
    Direction.NEUTRAL: 0.0,
}


class CatalystSentiment(Factor):
    """Confidence-weighted, time-decayed net catalyst direction per ticker."""

    category = FactorCategory.NLP

    def __init__(
        self,
        catalysts: Optional[Sequence[Catalyst]] = None,
        half_life_days: int = 45,
        max_age_days: int = 365,
        min_lag_days: int = 1,
        enabled: Optional[bool] = None,
    ) -> None:
        self.catalysts: List[Catalyst] = list(catalysts or [])
        self.half_life_days = half_life_days
        self.max_age_days = max_age_days
        self.min_lag_days = min_lag_days
        self.required_history = 1
        self.name = "catalyst_sentiment"
        # Enabled only when there is something to score, unless forced.
        self.enabled = bool(self.catalysts) if enabled is None else bool(enabled)

    @property
    def is_stubbed(self) -> bool:
        return not self.enabled or not self.catalysts

    def _compute(self, window: Panel) -> pd.Series:
        if self.is_stubbed:
            return pd.Series(np.nan, index=window.tickers, dtype=float)

        as_of: Date = window.dates[-1].date()
        scores: Dict[str, float] = {}

        for catalyst in self.catalysts:
            if not catalyst.is_known_at(as_of):
                continue  # the as-of rule, enforced on evidence as well as prices
            age = (as_of - catalyst.source_date).days
            if age < 0 or age > self.max_age_days:
                continue
            decay = 0.5 ** (age / self.half_life_days)
            sign = DIRECTION_SIGN.get(catalyst.direction, 0.0)
            scores[catalyst.ticker] = scores.get(catalyst.ticker, 0.0) + (
                sign * catalyst.confidence * decay
            )

        if not scores:
            return pd.Series(np.nan, index=window.tickers, dtype=float)

        out = pd.Series(scores, dtype=float).reindex(window.tickers)
        # Names with no catalysts stay NaN rather than becoming 0.0: no evidence.
        return out

    def with_catalysts(self, catalysts: Sequence[Catalyst]) -> "CatalystSentiment":
        """Wire in Zain's C15 output. The one line that turns the stub live."""
        return CatalystSentiment(
            catalysts=catalysts,
            half_life_days=self.half_life_days,
            max_age_days=self.max_age_days,
            min_lag_days=self.min_lag_days,
            enabled=True,
        )

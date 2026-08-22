"""Catalyst and sentiment factor. Step B8 — WAITS ON ZAIN'S C15.

Ships disabled so the composite runs end to end today. The moment C15 starts
emitting `Catalyst` objects, hand them to the constructor and this factor joins
the model with no other change anywhere.

Disabled returns NaN, not 0.0. The build order says "returning zeros", and the
effect is the same — zero contribution to the composite — but the mechanism
matters. A 0.0 is a measurement: "this name has neutral sentiment". A NaN is an
absence: "nothing is known here". B9 rescales the surviving weights around a
NaN, so the other factors still add to a full-strength score. Feed it zeros
instead and every name gets silently diluted by a factor that measured nothing.

The as-of rule applies to catalysts exactly as it does to prices: a catalyst is
usable only once its SOURCE was published. `Catalyst.is_known_at` is the guard,
and it keys on `source_date` rather than `event_date` — the market learned about
the event when the filing appeared, not when the event happened.
"""
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
    """Confidence-weighted, time-decayed net catalyst direction per ticker.

    Older evidence counts for less. A guidance raise from three weeks ago is
    live information; the same raise from ten months ago is in the price. The
    half-life makes that explicit instead of letting a stale catalyst carry the
    same weight as today's.
    """

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
        # Names with no catalysts stay NaN rather than becoming 0.0: no evidence
        # is not the same finding as balanced evidence.
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

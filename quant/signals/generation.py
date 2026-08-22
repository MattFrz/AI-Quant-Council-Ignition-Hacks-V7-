"""The lane's single entry point."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from data.schemas.signal import AlphaBreakdown, FactorCategory
from quant.alpha.composite import CompositeModel, equal_weight_model
from quant.alpha.statistical_tests import factor_scoreboard, information_coefficient
from quant.alpha.weighting import FittedWeights, fit_weights, split_train_test
from quant.factors.base import Factor, Panel
from quant.factors.market import default_market_factors
from quant.factors.nlp import CatalystSentiment
from quant.signals.cross_sectional import build_factor_panels, rebalance_dates


@dataclass
class SignalEngine:
    """Factors in, ranked and explained trade candidates out."""

    factors: List[Factor] = field(default_factory=list)
    sector_neutral: bool = False
    horizon: int = 21
    rebalance_freq: str = "ME"

    _panels: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _dates: Optional[pd.DatetimeIndex] = field(default=None, repr=False)
    fitted: Optional[FittedWeights] = None
    model: Optional[CompositeModel] = None

    # ---- construction ------------------------------------------------------

    @classmethod
    def default(cls, catalysts: Optional[Sequence] = None, **kwargs) -> "SignalEngine":
        """The market factors, plus the NLP factor — live if catalysts are"""
        factors: List[Factor] = list(default_market_factors())
        factors.append(CatalystSentiment(catalysts=catalysts))
        return cls(factors=factors, **kwargs)

    @property
    def categories(self) -> Dict[str, FactorCategory]:
        return {f.name: f.category for f in self.factors}

    @property
    def factor_names(self) -> List[str]:
        return [f.name for f in self.factors]

    # ---- pipeline ----------------------------------------------------------

    def build(self, panel: Panel, dates: Optional[Sequence] = None) -> "SignalEngine":
        """Compute and normalize every factor. Everything else reuses this."""
        if dates is None:
            warmup = max((f.required_history for f in self.factors), default=0)
            dates = rebalance_dates(panel, freq=self.rebalance_freq, warmup=warmup)
        self._dates = pd.DatetimeIndex(dates)
        self._panels = build_factor_panels(
            self.factors, panel, self._dates, sector_neutral=self.sector_neutral
        )
        return self

    def fit(
        self,
        panel: Panel,
        dates: Optional[Sequence] = None,
        train_frac: float = 0.6,
        method: str = "ic",
        **kwargs,
    ) -> "SignalEngine":
        """Fit weights on the train window, leaving test untouched."""
        if not self._panels or dates is not None:
            self.build(panel, dates)

        train, _ = split_train_test(self._dates, train_frac=train_frac,
                                    embargo=self.horizon)
        self.fitted = fit_weights(self._panels, panel, train, method=method,
                                  horizon=self.horizon, **kwargs)
        self.model = self.fitted.to_model(categories=self.categories)
        return self

    def use_equal_weights(self, panel: Panel, dates: Optional[Sequence] = None) -> "SignalEngine":
        """Baseline model, no fitting. What a fitted model has to beat."""
        if not self._panels or dates is not None:
            self.build(panel, dates)
        self.model = equal_weight_model(self._panels, categories=self.categories)
        self.fitted = None
        return self

    # ---- outputs -----------------------------------------------------------

    @property
    def panels(self) -> Dict[str, pd.DataFrame]:
        self._require_built()
        return self._panels

    @property
    def dates(self) -> pd.DatetimeIndex:
        self._require_built()
        return self._dates

    def scores(self) -> pd.DataFrame:
        """Composite alpha, `date x ticker`. What Matt's backtester consumes."""
        self._require_model()
        return self.model.score_panel(self._panels)

    def rank(self, as_of, top_n: int = 10) -> List[AlphaBreakdown]:
        """The candidate list, best first, each with its factor breakdown."""
        self._require_model()
        return self.model.rank_date(self._panels, as_of, top_n=top_n)

    def scoreboard(self, panel: Panel) -> pd.DataFrame:
        """Per-factor IC table — the evidence behind the weights."""
        self._require_built()
        return factor_scoreboard(self.factors, panel, horizon=self.horizon,
                                 dates=self._dates)

    def evaluate(self, panel: Panel, dates: Optional[Sequence] = None, label: str = "composite"):
        """Score the composite itself. Pass the TEST dates for the honest number."""
        self._require_model()
        scores = self.scores()
        if dates is not None:
            scores = scores.reindex(pd.DatetimeIndex(dates))
        return information_coefficient(scores, panel, self.horizon, "spearman", name=label)

    def train_test_dates(self, train_frac: float = 0.6):
        self._require_built()
        return split_train_test(self._dates, train_frac=train_frac, embargo=self.horizon)

    # ---- guards ------------------------------------------------------------

    def _require_built(self) -> None:
        if not self._panels or self._dates is None:
            raise RuntimeError("Call build() or fit() before requesting output.")

    def _require_model(self) -> None:
        self._require_built()
        if self.model is None:
            raise RuntimeError(
                "No weights yet — call fit() to learn them, or use_equal_weights() "
                "for the baseline. Weights are never invented here."
            )

"""The composite alpha model. Step B9."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from data.schemas.signal import AlphaBreakdown, FactorCategory, SignalContribution


@dataclass
class CompositeModel:
    """Weighted sum of normalized factors, with the breakdown preserved."""

    weights: Dict[str, float]
    categories: Dict[str, FactorCategory] = field(default_factory=dict)
    renormalize_missing: bool = True
    min_factors: int = 1

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("CompositeModel needs at least one weighted factor.")
        if self.min_factors < 1:
            raise ValueError("min_factors must be >= 1")

    @property
    def factor_names(self) -> List[str]:
        return list(self.weights)

    @property
    def total_abs_weight(self) -> float:
        return float(sum(abs(w) for w in self.weights.values()))

    def category_of(self, factor: str) -> FactorCategory:
        return self.categories.get(factor, FactorCategory.MOMENTUM)

    # ---- panel-level scoring ----------------------------------------------

    def _scale_frame(self, panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Per-cell rescaling factor that compensates for missing factors."""
        available = None
        for name, w in self.weights.items():
            present = panels[name].notna().astype(float) * abs(w)
            available = present if available is None else available.add(present, fill_value=0.0)

        n_present = None
        for name in self.weights:
            got = panels[name].notna().astype(int)
            n_present = got if n_present is None else n_present.add(got, fill_value=0)

        scale = self.total_abs_weight / available.replace(0.0, np.nan)
        if not self.renormalize_missing:
            scale = scale.where(scale.isna(), 1.0)
        return scale.where(n_present >= self.min_factors)

    def contribution_panels(
        self, panels: Mapping[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Each factor's contribution to the composite, `date x ticker`."""
        self._check_inputs(panels)
        scale = self._scale_frame(panels)
        return {
            name: panels[name] * (w * scale)
            for name, w in self.weights.items()
        }

    def score_panel(self, panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Composite alpha, `date x ticker`. The sum of the contributions."""
        contribs = self.contribution_panels(panels)
        total = None
        for frame in contribs.values():
            total = frame.fillna(0.0) if total is None else total.add(frame.fillna(0.0))
        valid = self._scale_frame(panels).notna()
        return total.where(valid)

    def alpha_scores(self, panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """The 0-10 display score: cross-sectional percentile, divided by ten."""
        composite = self.score_panel(panels)
        return composite.rank(axis=1, pct=True, na_option="keep") * 10.0

    # ---- contract objects --------------------------------------------------

    def breakdown(
        self,
        panels: Mapping[str, pd.DataFrame],
        ticker: str,
        as_of,
        alpha_score: Optional[float] = None,
    ) -> AlphaBreakdown:
        """One `AlphaBreakdown` — the object that lands in `TradeIdea.alpha`."""
        self._check_inputs(panels)
        ts = pd.Timestamp(as_of)
        scale = self._scale_frame(panels).at[ts, ticker]

        contributions: List[SignalContribution] = []
        for name, w in self.weights.items():
            z = panels[name].at[ts, ticker]
            if pd.isna(z) or pd.isna(scale):
                continue
            effective = float(w * scale)
            contributions.append(
                SignalContribution(
                    factor=name,
                    category=self.category_of(name),
                    zscore=float(z),
                    weight=effective,
                    contribution=float(z * effective),
                )
            )

        # Derived from the parts, never computed separately — check_sums() is a
        # gate in scripts/verify_contract.py and it must hold exactly.
        composite = float(sum(c.contribution for c in contributions))

        if alpha_score is None:
            scores = self.alpha_scores(panels)
            raw = scores.at[ts, ticker] if ticker in scores.columns else np.nan
            alpha_score = 5.0 if pd.isna(raw) else float(raw)

        return AlphaBreakdown(
            ticker=ticker,
            as_of=ts.date(),
            contributions=contributions,
            composite_alpha=composite,
            alpha_score=float(np.clip(alpha_score, 0.0, 10.0)),
        )

    def rank_date(
        self,
        panels: Mapping[str, pd.DataFrame],
        as_of,
        top_n: Optional[int] = None,
    ) -> List[AlphaBreakdown]:
        """Every name on one date, best first. The scanner's candidate list."""
        ts = pd.Timestamp(as_of)
        composite = self.score_panel(panels).loc[ts].dropna().sort_values(ascending=False)
        scores = self.alpha_scores(panels)
        if top_n is not None:
            composite = composite.head(top_n)
        return [
            self.breakdown(panels, ticker, ts, alpha_score=float(scores.at[ts, ticker]))
            for ticker in composite.index
        ]

    # ---- guards ------------------------------------------------------------

    def _check_inputs(self, panels: Mapping[str, pd.DataFrame]) -> None:
        missing = set(self.weights) - set(panels)
        if missing:
            raise KeyError(
                f"CompositeModel is weighted on {sorted(missing)} but no panel was "
                "supplied for them. Every weighted factor needs a normalized panel."
            )


def equal_weight_model(
    panels: Mapping[str, pd.DataFrame],
    categories: Optional[Mapping[str, FactorCategory]] = None,
) -> CompositeModel:
    """Equal weights — the baseline any fitted model has to beat."""
    n = len(panels)
    return CompositeModel(
        weights={name: 1.0 / n for name in panels},
        categories=dict(categories or {}),
    )

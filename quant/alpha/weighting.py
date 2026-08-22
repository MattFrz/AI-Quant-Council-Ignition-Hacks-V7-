"""Fitting the composite weights. Step B10."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.schemas.signal import FactorCategory
from quant.alpha.composite import CompositeModel
from quant.alpha.statistical_tests import cross_sectional_ic, summarize_ic
from quant.factors.base import Panel
from quant.signals.cross_sectional import forward_returns


@dataclass
class FittedWeights:
    """Weights plus the evidence for them. Never just the numbers."""

    weights: Dict[str, float]
    method: str
    horizon: int
    train_start: Date
    train_end: Date
    n_train_periods: int
    embargo_days: int
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_model(
        self,
        categories: Optional[Mapping[str, FactorCategory]] = None,
        **kwargs,
    ) -> CompositeModel:
        return CompositeModel(
            weights=dict(self.weights),
            categories=dict(categories or {}),
            **kwargs,
        )

    def __str__(self) -> str:
        rows = "  ".join(f"{k}={v:+.3f}" for k, v in self.weights.items())
        return (
            f"{self.method} weights (train {self.train_start}..{self.train_end}, "
            f"n={self.n_train_periods}, horizon={self.horizon}d, "
            f"embargo={self.embargo_days}d)\n  {rows}"
        )


def split_train_test(
    dates: Sequence,
    train_frac: float = 0.6,
    embargo: int = 21,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Chronological split with an embargo gap between the windows."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    idx = pd.DatetimeIndex(dates).sort_values()
    split_at = int(len(idx) * train_frac)
    test = idx[split_at:]

    train = idx[:split_at]
    if embargo > 0 and len(train):
        cutoff = test[0] - pd.Timedelta(days=embargo * 7 / 5) if len(test) else train[-1]
        train = train[train <= cutoff]

    return train, test


def _stack(
    panels: Mapping[str, pd.DataFrame],
    fwd: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Flatten the train window into a design matrix and a return vector."""
    names = list(panels)
    cols = {}
    for name in names:
        cols[name] = panels[name].reindex(index=dates).stack(future_stack=True)
    X = pd.DataFrame(cols)
    y = fwd.reindex(index=dates).stack(future_stack=True)
    joined = X.join(y.rename("__fwd__"), how="inner").dropna()
    return joined[names], joined["__fwd__"]


def fit_ic_weights(
    panels: Mapping[str, pd.DataFrame],
    panel: Panel,
    train_dates: Sequence,
    horizon: int = 21,
    method: str = "spearman",
    use_ir: bool = True,
    min_abs_t: float = 0.0,
) -> FittedWeights:
    """Weight each factor by its information coefficient on the train window."""
    train = pd.DatetimeIndex(train_dates)
    fwd = forward_returns(panel, horizon=horizon)

    rows, raw = [], {}
    for name, scores in panels.items():
        ic = cross_sectional_ic(scores.reindex(index=train), fwd, method=method)
        summary = summarize_ic(ic, factor=name, horizon=horizon, method=method)
        strength = summary.ir if use_ir else summary.mean_ic
        if not np.isfinite(strength) or abs(summary.t_stat) < min_abs_t:
            strength = 0.0
        raw[name] = float(strength)
        row = summary.as_row()
        row["raw_strength"] = float(strength)
        rows.append(row)

    total = sum(abs(v) for v in raw.values())
    if total == 0:
        raise ValueError(
            "Every factor scored zero on the train window — nothing survived "
            f"min_abs_t={min_abs_t}. Lower the bar or fix the factors; do not "
            "ship an equal-weight fallback and call it fitted."
        )
    weights = {k: v / total for k, v in raw.items()}

    diagnostics = pd.DataFrame(rows).set_index("factor")
    diagnostics["weight"] = pd.Series(weights)

    return FittedWeights(
        weights=weights,
        method="ic" + ("_ir" if use_ir else "_mean"),
        horizon=horizon,
        train_start=train[0].date(),
        train_end=train[-1].date(),
        n_train_periods=len(train),
        embargo_days=horizon,
        diagnostics=diagnostics,
    )


def fit_ridge_weights(
    panels: Mapping[str, pd.DataFrame],
    panel: Panel,
    train_dates: Sequence,
    horizon: int = 21,
    ridge_alpha: float = 10.0,
) -> FittedWeights:
    """Penalized pooled regression of forward returns on factor scores."""
    train = pd.DatetimeIndex(train_dates)
    fwd = forward_returns(panel, horizon=horizon)

    # Ridge needs complete rows, and a factor that is entirely absent on the
    # train window (the B8 stub, before C15 lands) would drop EVERY row via the
    # dropna in _stack. Exclude those up front and give them zero weight — the
    # alternative is a fit that cannot run at all while any factor is stubbed.
    usable = {
        name: scores for name, scores in panels.items()
        if scores.reindex(index=train).notna().any().any()
    }
    stubbed = [name for name in panels if name not in usable]
    if not usable:
        raise ValueError(
            "Every factor is empty on the train window — nothing to fit."
        )

    X, y = _stack(usable, fwd, train)

    if X.empty:
        raise ValueError("No complete rows in the train window — cannot fit.")

    names = list(X.columns)
    Xv = X.values
    yv = y.values - y.values.mean()

    gram = Xv.T @ Xv + ridge_alpha * np.eye(len(names))
    beta = np.linalg.solve(gram, Xv.T @ yv)

    total = float(np.abs(beta).sum())
    if total == 0:
        raise ValueError("Ridge fit returned all-zero coefficients.")
    weights = {n: float(b / total) for n, b in zip(names, beta)}
    weights.update({n: 0.0 for n in stubbed})

    diagnostics = pd.DataFrame(
        {"coefficient": list(beta) + [0.0] * len(stubbed),
         "weight": [weights[n] for n in names + stubbed]},
        index=names + stubbed,
    )
    diagnostics.index.name = "factor"
    diagnostics["n_obs"] = len(X)

    return FittedWeights(
        weights=weights,
        method=f"ridge(alpha={ridge_alpha})",
        horizon=horizon,
        train_start=train[0].date(),
        train_end=train[-1].date(),
        n_train_periods=len(train),
        embargo_days=horizon,
        diagnostics=diagnostics,
    )


def fit_weights(
    panels: Mapping[str, pd.DataFrame],
    panel: Panel,
    train_dates: Sequence,
    method: str = "ic",
    **kwargs,
) -> FittedWeights:
    """Dispatcher. `method` is 'ic' or 'ridge'."""
    if method == "ic":
        return fit_ic_weights(panels, panel, train_dates, **kwargs)
    if method == "ridge":
        return fit_ridge_weights(panels, panel, train_dates, **kwargs)
    raise ValueError(f"unknown weighting method {method!r}; use 'ic' or 'ridge'")

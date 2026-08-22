"""Finding historical setups that resemble the current one. Step B12."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from quant.eventstudy.study import EventStudyResult, event_study
from quant.factors.base import Panel


@dataclass
class MatchedSetup:
    """One historical (ticker, date) that looked like the target."""

    ticker: str
    date: pd.Timestamp
    distance: float
    similarity: float          # 1 / (1 + distance), for display

    def as_row(self) -> dict:
        return {
            "ticker": self.ticker,
            "date": self.date.date(),
            "distance": round(self.distance, 4),
            "similarity": round(self.similarity, 4),
        }


@dataclass
class MatchResult:
    target_ticker: str
    target_date: pd.Timestamp
    target_features: Dict[str, float]
    matches: List[MatchedSetup] = field(default_factory=list)
    study: Optional[EventStudyResult] = None

    @property
    def n_matches(self) -> int:
        return len(self.matches)

    def to_events(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"ticker": m.ticker, "event_date": m.date} for m in self.matches],
            columns=["ticker", "event_date"],
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([m.as_row() for m in self.matches])

    def summary_line(self) -> str:
        if self.study is None:
            return (
                f"{self.n_matches} historical setups resembling {self.target_ticker} "
                f"on {self.target_date.date()}; outcomes not yet measured."
            )
        s = self.study
        verb = "gained" if s.car_post_event >= 0 else "lost"
        return (
            f"Found {s.n_events} historical setups resembling {self.target_ticker} on "
            f"{self.target_date.date()}. On average they {verb} "
            f"{abs(s.car_post_event):.2%} market-adjusted over the following "
            f"{s.post} trading days (t={s.t_post_event:+.2f}, p={s.p_post_event:.3f})."
        )

    def __str__(self) -> str:
        return self.summary_line()


def _feature_frame(panels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Factor panels -> one long frame indexed by (date, ticker)."""
    cols = {name: df.stack(future_stack=True) for name, df in panels.items()}
    return pd.DataFrame(cols).dropna()


def find_similar_setups(
    panels: Mapping[str, pd.DataFrame],
    target_ticker: str,
    target_date,
    k: int = 25,
    post: int = 40,
    min_gap_days: Optional[int] = None,
    min_separation_days: int = 60,
    exclude_same_ticker: bool = False,
    weights: Optional[Mapping[str, float]] = None,
) -> MatchResult:
    """The k closest historical setups, by weighted distance on factor scores."""
    target_ts = pd.Timestamp(target_date)
    features = _feature_frame(panels)
    names = list(panels)

    if (target_ts, target_ticker) not in features.index:
        raise KeyError(
            f"{target_ticker} has no complete factor vector on "
            f"{target_ts.date()} — cannot match a setup that was never scored."
        )
    target_vec = features.loc[(target_ts, target_ticker)].to_numpy(dtype=float)

    w = np.array([float((weights or {}).get(n, 1.0)) for n in names])
    w = np.abs(w)
    if w.sum() == 0:
        w = np.ones(len(names))

    # Hard cutoff: match date AND its whole outcome window must precede the
    # decision date. This is the line that keeps the study honest.
    gap = min_gap_days if min_gap_days is not None else int(np.ceil(post * 7 / 5))
    cutoff = target_ts - pd.Timedelta(days=gap)

    dates = features.index.get_level_values(0)
    tickers = features.index.get_level_values(1)
    mask = dates <= cutoff
    if exclude_same_ticker:
        mask = mask & (tickers != target_ticker)

    pool = features[mask]
    if pool.empty:
        return MatchResult(target_ticker, target_ts, dict(zip(names, target_vec)), [])

    diff = (pool.to_numpy(dtype=float) - target_vec) * np.sqrt(w)
    distance = np.sqrt((diff**2).sum(axis=1))

    order = np.argsort(distance)
    chosen: List[MatchedSetup] = []
    last_seen: Dict[str, List[pd.Timestamp]] = {}

    for i in order:
        if len(chosen) >= k:
            break
        d, t = pool.index[i]
        prior = last_seen.get(t, [])
        if any(abs((d - p).days) < min_separation_days for p in prior):
            continue  # same episode, already represented
        chosen.append(
            MatchedSetup(
                ticker=str(t),
                date=pd.Timestamp(d),
                distance=float(distance[i]),
                similarity=float(1.0 / (1.0 + distance[i])),
            )
        )
        last_seen.setdefault(t, []).append(pd.Timestamp(d))

    return MatchResult(
        target_ticker=target_ticker,
        target_date=target_ts,
        target_features=dict(zip(names, target_vec)),
        matches=chosen,
    )


def study_matched_setups(
    panel: Panel,
    result: MatchResult,
    pre: int = 10,
    post: int = 40,
    min_events: int = 5,
) -> MatchResult:
    """Run B11's event study over the matched dates and attach the outcome."""
    events = result.to_events()
    if len(events) < min_events:
        return result
    result.study = event_study(
        panel,
        events,
        pre=pre,
        post=post,
        label=f"setups resembling {result.target_ticker}",
        min_events=min_events,
    )
    return result


def analogue_study(
    panel: Panel,
    panels: Mapping[str, pd.DataFrame],
    target_ticker: str,
    target_date,
    k: int = 25,
    pre: int = 10,
    post: int = 40,
    **kwargs,
) -> MatchResult:
    """Match, then measure. The one call the research agent should make."""
    result = find_similar_setups(
        panels, target_ticker, target_date, k=k, post=post, **kwargs
    )
    return study_matched_setups(panel, result, pre=pre, post=post)

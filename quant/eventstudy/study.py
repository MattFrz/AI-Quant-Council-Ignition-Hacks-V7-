"""Cumulative abnormal return around events. Step B11.

This is the statistical evidence behind the audit trail's central claim —
"these events historically preceded positive returns". Zain's C16 assembles the
trail; the number it quotes has to come from here, because the whole section 23
differentiation rests on the LLM never producing a statistic itself.

Abnormal return is market-adjusted: the stock's return minus the universe's
equal-weighted return that day. Without that adjustment an event study run
through a bull market shows every event working, because everything worked.

What comes out is a curve and a t-stat, both honest about sample size. Twelve
events is an anecdote with error bars, and `min_events` refuses rather than
letting a three-event "pattern" reach a slide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from data.schemas.catalyst import Catalyst
from quant.factors.base import Panel

MIN_EVENTS = 5


@dataclass
class EventStudyResult:
    """The curve Cecile's EventStudyChart draws, plus its significance."""

    label: str
    n_events: int
    pre: int
    post: int
    offsets: np.ndarray
    mean_car: np.ndarray          # cumulative abnormal return, from window start
    se_car: np.ndarray
    t_stat: np.ndarray
    car_post_event: float         # CAR from the day before the event to +post
    t_post_event: float
    p_post_event: float

    def to_curve(self) -> List[Dict]:
        """Chart-ready points: offset in trading days, CAR as a fraction."""
        return [
            {"offset": int(o), "value": float(v), "t_stat": float(t)}
            for o, v, t in zip(self.offsets, self.mean_car, self.t_stat)
        ]

    def is_significant(self, alpha: float = 0.05) -> bool:
        return bool(np.isfinite(self.p_post_event) and self.p_post_event < alpha)

    def summary_line(self) -> str:
        """One sentence, quotable, with the sample size attached."""
        direction = "gained" if self.car_post_event >= 0 else "lost"
        return (
            f"Across {self.n_events} historical instances, {self.label} {direction} "
            f"{abs(self.car_post_event):.2%} market-adjusted over the following "
            f"{self.post} trading days (t={self.t_post_event:+.2f}, "
            f"p={self.p_post_event:.3f})."
        )

    def __str__(self) -> str:
        return self.summary_line()


def events_from_catalysts(
    catalysts: Iterable[Catalyst],
    use_event_date: bool = False,
) -> pd.DataFrame:
    """Zain's C15 output -> the (ticker, date) frame this module consumes.

    Defaults to `source_date`, not `event_date`: the study measures what
    happened after the market could KNOW about the event. Using event_date
    credits the strategy with moves that occurred before the filing appeared.
    """
    rows = []
    for c in catalysts:
        when = c.event_date if (use_event_date and c.event_date) else c.source_date
        rows.append({"ticker": c.ticker, "event_date": pd.Timestamp(when),
                     "direction": c.direction.value, "confidence": c.confidence})
    return pd.DataFrame(rows, columns=["ticker", "event_date", "direction", "confidence"])


def abnormal_returns(panel: Panel, market_adjust: bool = True) -> pd.DataFrame:
    """Daily returns less the equal-weighted universe return."""
    rets = panel.returns()
    if not market_adjust:
        return rets
    market = rets.mean(axis=1)
    return rets.sub(market, axis=0)


def event_study(
    panel: Panel,
    events: pd.DataFrame,
    pre: int = 20,
    post: int = 40,
    label: str = "this event",
    market_adjust: bool = True,
    min_events: int = MIN_EVENTS,
) -> EventStudyResult:
    """Average CAR path around a set of (ticker, event_date) pairs."""
    required = {"ticker", "event_date"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events frame is missing {sorted(missing)}")

    ar = abnormal_returns(panel, market_adjust=market_adjust)
    dates = panel.dates
    offsets = np.arange(-pre, post + 1)
    window_len = len(offsets)

    paths: List[np.ndarray] = []
    for _, row in events.iterrows():
        ticker = row["ticker"]
        if ticker not in ar.columns:
            continue
        pos = int(dates.searchsorted(pd.Timestamp(row["event_date"]), side="left"))
        start, end = pos - pre, pos + post + 1
        if start < 0 or end > len(dates):
            continue  # window runs off the edge of the data
        segment = ar[ticker].iloc[start:end].to_numpy(dtype=float)
        if len(segment) != window_len or np.isnan(segment).sum() > window_len * 0.2:
            continue
        paths.append(np.nan_to_num(segment, nan=0.0))

    n = len(paths)
    if n < min_events:
        raise ValueError(
            f"Only {n} usable events (need {min_events}). An event study on a "
            "handful of cases is an anecdote — report the count or do not report it."
        )

    car = np.cumsum(np.vstack(paths), axis=1)      # n_events x window
    mean_car = car.mean(axis=0)
    se_car = car.std(axis=0, ddof=1) / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(se_car > 0, mean_car / se_car, np.nan)

    # The event's own contribution: from the close before the event to +post.
    baseline_idx = pre - 1 if pre > 0 else 0
    per_event_move = car[:, -1] - car[:, baseline_idx]
    move = float(per_event_move.mean())
    move_se = float(per_event_move.std(ddof=1) / np.sqrt(n))
    t_move = move / move_se if move_se > 0 else np.nan
    p_move = float(2 * stats.t.sf(abs(t_move), df=n - 1)) if np.isfinite(t_move) else np.nan

    return EventStudyResult(
        label=label,
        n_events=n,
        pre=pre,
        post=post,
        offsets=offsets,
        mean_car=mean_car,
        se_car=se_car,
        t_stat=t_stat,
        car_post_event=move,
        t_post_event=float(t_move),
        p_post_event=p_move,
    )


def study_by_direction(
    panel: Panel,
    events: pd.DataFrame,
    **kwargs,
) -> Dict[str, EventStudyResult]:
    """Split bullish from bearish before averaging.

    Pooling them cancels the effect out and reports "no signal" for a catalyst
    type that works perfectly well in both directions.
    """
    out: Dict[str, EventStudyResult] = {}
    if "direction" not in events.columns:
        out["all"] = event_study(panel, events, **kwargs)
        return out

    for direction, group in events.groupby("direction"):
        try:
            out[str(direction)] = event_study(
                panel, group, label=f"{direction} catalysts", **kwargs
            )
        except ValueError:
            continue  # too few of this kind; leave it out rather than fake it
    return out

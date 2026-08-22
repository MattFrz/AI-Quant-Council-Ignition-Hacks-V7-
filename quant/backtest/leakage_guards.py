"""Point-in-time guards. Step A8 - written BEFORE the engine, on purpose.

Retrofitting leakage protection into a working backtest never happens: once the
numbers look good nobody wants to touch them. So the engine calls into here,
not the other way round.

Three rules this module enforces:
  1. A fundamental row is invisible until its report_date.
  2. A filing is invisible until its filed_date.
  3. A signal computed from data through the close of day t is traded at day
     t+1's price, never day t's.
"""
from __future__ import annotations

from datetime import date
from typing import Optional, Sequence, Union

import pandas as pd

from backend.core.logging import get_logger

log = get_logger(__name__)

DateLike = Union[str, date, pd.Timestamp]


class LeakageError(AssertionError):
    """Raised when future information reaches a historical decision."""


# ------------------------------------------------------------------- as-of

def available_at(
    df: pd.DataFrame,
    as_of: DateLike,
    date_col: str = "report_date",
) -> pd.DataFrame:
    """Rows whose information was public on or before as_of.

    This is the only correct way to read fundamentals or filings at a historical
    date. Filtering on period_end instead of report_date is the classic
    look-ahead bug.
    """
    if date_col not in df.columns:
        raise KeyError(
            f"{date_col} missing. Point-in-time filtering requires a publication "
            f"date column; available: {list(df.columns)}"
        )
    stamp = pd.Timestamp(as_of)
    return df[pd.to_datetime(df[date_col]) <= stamp]


def latest_available(
    df: pd.DataFrame,
    as_of: DateLike,
    key_col: str = "ticker",
    date_col: str = "report_date",
) -> pd.DataFrame:
    """Most recent publicly-known row per key as of a date."""
    visible = available_at(df, as_of, date_col)
    if visible.empty:
        return visible
    ordered = visible.sort_values(date_col)
    return ordered.groupby(key_col, as_index=False).last()


def asof_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str = "date",
    right_on: str = "report_date",
    by: str = "ticker",
) -> pd.DataFrame:
    """Backward as-of join: attach the most recent PUBLISHED fundamentals to
    each price date. Never forward-fills across the publication boundary."""
    l = left.copy()
    r = right.copy()
    l[left_on] = pd.to_datetime(l[left_on])
    r[right_on] = pd.to_datetime(r[right_on])

    return pd.merge_asof(
        l.sort_values(left_on),
        r.sort_values(right_on),
        left_on=left_on,
        right_on=right_on,
        by=by,
        direction="backward",
        allow_exact_matches=True,
    )


# ------------------------------------------------------------- execution lag

def enforce_execution_lag(signal: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """Shift a signal forward so it is traded after it could have been known.

    A signal computed from the close of day t cannot be traded at day t's close.
    Every signal entering the backtester passes through here.
    """
    if lag_days < 1:
        raise LeakageError(
            "execution lag must be at least 1 day - trading on the same bar the "
            "signal was computed from is look-ahead bias"
        )
    return signal.shift(lag_days)


# --------------------------------------------------------------- assertions

def assert_no_future_data(
    df: pd.DataFrame,
    as_of: DateLike,
    date_col: str = "date",
    label: str = "frame",
) -> None:
    """Hard stop if any row postdates the decision date."""
    stamp = pd.Timestamp(as_of)
    if df.empty:
        return
    latest = pd.to_datetime(df[date_col]).max()
    if latest > stamp:
        raise LeakageError(
            f"{label}: contains data dated {latest.date()} but the decision date "
            f"is {stamp.date()}. Future information has reached a historical decision."
        )


def assert_causal(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    label: str = "signal",
) -> None:
    """Signal and returns must share an index, and the signal must not be
    perfectly correlated with the same day's return - that is the fingerprint of
    a signal built from the return it is supposed to predict."""
    common = signal.index.intersection(returns.index)
    if len(common) == 0:
        raise LeakageError(f"{label}: no overlapping dates with returns")

    s = signal.loc[common].stack(future_stack=True)
    r = returns.loc[common].stack(future_stack=True)
    joined = pd.concat([s, r], axis=1).dropna()
    if len(joined) < 30:
        return

    corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    if pd.notna(corr) and abs(corr) > 0.95:
        raise LeakageError(
            f"{label}: same-day correlation with returns is {corr:.3f}. The "
            f"signal is almost certainly built from the return it predicts."
        )


def assert_window_clean(train_end: DateLike, test_start: DateLike) -> None:
    """Train must end strictly before test begins."""
    if pd.Timestamp(train_end) >= pd.Timestamp(test_start):
        raise LeakageError(
            f"train window ends {pd.Timestamp(train_end).date()} but test starts "
            f"{pd.Timestamp(test_start).date()} - windows overlap"
        )


def _values_match(a, b, tol: float = 1e-10) -> bool:
    """Elementwise equality that treats NaN-vs-number as a MISMATCH.

    This matters more than it looks. A look-ahead signal typically differs from
    its causal twin only at the final rows, where the truncated input produces
    NaN and the extended input produces a real number. Plain arithmetic
    comparison propagates NaN and pandas' .max() skips it, so the discrepancy
    disappears and the leak passes the check.
    """
    a = a.astype(float)
    b = b.astype(float)

    if not a.isna().equals(b.isna()):
        return False

    diff = (a.fillna(0.0) - b.fillna(0.0)).abs()
    worst = diff.to_numpy().max() if diff.size else 0.0
    return bool(worst <= tol)


def check_stability(
    compute_fn,
    panel: pd.DataFrame,
    as_of: DateLike,
    future_panel: Optional[pd.DataFrame] = None,
) -> bool:
    """The property test: a signal computed at date t must be unchanged when
    future rows are appended to the input.

    Pass a callable that takes a panel and returns a Series/DataFrame. Used by
    tests/quant/test_leakage.py.
    """
    stamp = pd.Timestamp(as_of)
    truncated = panel[pd.to_datetime(panel["date"]) <= stamp]

    baseline = compute_fn(truncated)
    extended = compute_fn(future_panel if future_panel is not None else panel)

    if isinstance(baseline, pd.DataFrame):
        extended = extended.reindex(index=baseline.index, columns=baseline.columns)
    else:
        extended = extended.reindex(baseline.index)

    return _values_match(baseline, extended)

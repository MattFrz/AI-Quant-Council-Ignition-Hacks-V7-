"""Leakage tests. Step A19 - written before test_backtest.py, on purpose.

The property that matters: a signal computed at date t must be IDENTICAL when
future rows are appended to the input. If that fails, every performance number
downstream is fiction.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from quant.backtest.leakage_guards import (
    LeakageError,
    asof_merge,
    assert_causal,
    assert_no_future_data,
    assert_window_clean,
    available_at,
    check_stability,
    enforce_execution_lag,
    latest_available,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=200)
    rows = []
    for ticker in ("AAA", "BBB", "CCC"):
        price = 100.0
        for d in dates:
            price *= 1 + rng.normal(0, 0.015)
            rows.append({"date": d, "ticker": ticker, "adj_close": price, "volume": 1e6})
    return pd.DataFrame(rows)


@pytest.fixture
def fundamentals() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": "AAA", "period_end": date(2023, 3, 31), "report_date": date(2023, 5, 5), "eps": 1.10},
        {"ticker": "AAA", "period_end": date(2023, 6, 30), "report_date": date(2023, 8, 4), "eps": 1.25},
        {"ticker": "BBB", "period_end": date(2023, 3, 31), "report_date": date(2023, 5, 10), "eps": 0.80},
    ])


# --------------------------------------------------------------- as-of rules

def test_available_at_hides_unpublished_rows(fundamentals):
    """The core rule: a quarter is invisible until its report_date."""
    visible = available_at(fundamentals, "2023-05-06")
    assert len(visible) == 1
    assert visible.iloc[0]["ticker"] == "AAA"


def test_available_at_uses_report_date_not_period_end(fundamentals):
    """On 2023-04-15 the Q1 period has ENDED but nothing has been published."""
    assert available_at(fundamentals, "2023-04-15").empty


def test_available_at_rejects_missing_date_column(fundamentals):
    with pytest.raises(KeyError):
        available_at(fundamentals.drop(columns=["report_date"]), "2023-06-01")


def test_latest_available_takes_most_recent_published(fundamentals):
    out = latest_available(fundamentals, "2023-09-01")
    aaa = out[out["ticker"] == "AAA"].iloc[0]
    assert aaa["eps"] == 1.25


def test_asof_merge_does_not_leak_future_fundamentals(fundamentals):
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2023-05-01", "2023-05-08", "2023-08-10"]),
        "ticker": ["AAA", "AAA", "AAA"],
        "adj_close": [100.0, 101.0, 105.0],
    })
    merged = asof_merge(prices, fundamentals[fundamentals["ticker"] == "AAA"])

    assert pd.isna(merged.iloc[0]["eps"])       # before any report
    assert merged.iloc[1]["eps"] == 1.10        # after Q1 published
    assert merged.iloc[2]["eps"] == 1.25        # after Q2 published


# ------------------------------------------------------------ execution lag

def test_enforce_execution_lag_shifts_forward():
    sig = pd.DataFrame(
        {"AAA": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]),
    )
    lagged = enforce_execution_lag(sig, 1)
    assert pd.isna(lagged.iloc[0]["AAA"])
    assert lagged.iloc[1]["AAA"] == 1.0


def test_zero_lag_is_rejected():
    sig = pd.DataFrame({"AAA": [1.0, 2.0]}, index=pd.to_datetime(["2023-01-03", "2023-01-04"]))
    with pytest.raises(LeakageError):
        enforce_execution_lag(sig, 0)


# --------------------------------------------------------------- assertions

def test_assert_no_future_data_catches_future_rows(panel):
    with pytest.raises(LeakageError):
        assert_no_future_data(panel, "2023-02-01", label="panel")


def test_assert_no_future_data_passes_when_truncated(panel):
    truncated = panel[panel["date"] <= "2023-02-01"]
    assert_no_future_data(truncated, "2023-02-01", label="panel")


def test_assert_window_clean_rejects_overlap():
    with pytest.raises(LeakageError):
        assert_window_clean("2023-06-30", "2023-06-01")


def test_assert_window_clean_accepts_gap():
    assert_window_clean("2023-06-30", "2023-07-01")


def test_assert_causal_catches_signal_built_from_same_day_return(panel):
    """The fingerprint of the worst bug: signal == the return it predicts."""
    wide = panel.pivot_table(index="date", columns="ticker", values="adj_close")
    returns = wide.pct_change(fill_method=None)
    with pytest.raises(LeakageError):
        assert_causal(returns, returns, label="cheating_signal")


def test_assert_causal_passes_for_a_lagged_signal(panel):
    wide = panel.pivot_table(index="date", columns="ticker", values="adj_close")
    returns = wide.pct_change(fill_method=None)
    momentum = wide.pct_change(20, fill_method=None).shift(1)
    assert_causal(momentum, returns, label="momentum")


# ------------------------------------------------- the property that matters

def test_signal_is_unchanged_when_future_data_is_appended(panel):
    """Compute momentum at t, then append the rest of history and recompute.
    A causal signal gives identical values for the overlapping dates."""

    def momentum(df: pd.DataFrame) -> pd.DataFrame:
        wide = df.pivot_table(index="date", columns="ticker", values="adj_close")
        return wide.pct_change(20, fill_method=None).shift(1)

    cutoff = panel["date"].quantile(0.5)
    assert check_stability(momentum, panel, cutoff, future_panel=panel)


def test_lookahead_signal_fails_the_stability_check(panel):
    """A signal peeking at tomorrow's price must NOT survive the same test."""

    def peeking(df: pd.DataFrame) -> pd.DataFrame:
        wide = df.pivot_table(index="date", columns="ticker", values="adj_close")
        return wide.shift(-1) / wide - 1.0   # tomorrow's return, known today

    cutoff = panel["date"].quantile(0.5)
    assert not check_stability(peeking, panel, cutoff, future_panel=panel)

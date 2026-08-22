"""Shared test fixtures — chiefly the synthetic market."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.factors.base import Panel  # noqa: E402

TRADING_DAYS = 252
SECTORS = ["Technology", "Industrials", "Healthcare", "Financials", "Energy", "Consumer"]


def make_factor_and_returns(
    n_dates: int = 120,
    n_tickers: int = 200,
    ic: float = 0.05,
    seed: int = 11,
    freq: str = "ME",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A factor panel and forward returns with a designed cross-sectional IC."""
    if not -1.0 < ic < 1.0:
        raise ValueError(f"ic must be in (-1, 1), got {ic}")

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-31", periods=n_dates, freq=freq)
    tickers = [f"SYN{i:04d}" for i in range(n_tickers)]

    x = rng.standard_normal((n_dates, n_tickers))
    e = rng.standard_normal((n_dates, n_tickers))
    y = ic * x + np.sqrt(1.0 - ic**2) * e

    factor_df = pd.DataFrame(x, index=dates, columns=tickers)
    returns_df = pd.DataFrame(y * 0.08, index=dates, columns=tickers)  # scale to plausible returns
    return factor_df, returns_df


def make_synthetic_panel(
    n_tickers: int = 120,
    n_days: int = 1600,
    seed: int = 7,
    momentum_strength: float = 0.012,
    lookback: int = 252,
    skip: int = 21,
    start: str = "2018-01-02",
) -> Tuple[Panel, Dict]:
    """A price panel with genuine 12-1 momentum in the return process."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    tickers = [f"SYN{i:03d}" for i in range(n_tickers)]

    ann_vol = rng.uniform(0.18, 0.50, n_tickers)
    daily_vol = ann_vol / np.sqrt(TRADING_DAYS)
    drift = rng.normal(0.06, 0.04, n_tickers) / TRADING_DAYS

    logp = np.zeros((n_days, n_tickers))
    logp[0] = np.log(rng.uniform(20.0, 300.0, n_tickers))

    first_signal = lookback + skip + 1
    for t in range(1, n_days):
        shock = rng.normal(0.0, daily_vol)
        if t - 1 - skip - lookback >= 0:
            past = logp[t - 1 - skip] - logp[t - 1 - skip - lookback]
            spread = past.std()
            z = (past - past.mean()) / spread if spread > 0 else np.zeros(n_tickers)
            shock = shock + momentum_strength * daily_vol * z
        logp[t] = logp[t - 1] + drift + shock

    prices = np.exp(logp)
    adj_close = pd.DataFrame(prices, index=dates, columns=tickers)

    # Volume: lognormal around a per-name base, nudged up on big move days.
    base_vol = rng.uniform(3e5, 8e6, n_tickers)
    daily_ret = np.vstack([np.zeros((1, n_tickers)), np.diff(logp, axis=0)])
    activity = 1.0 + 3.0 * np.abs(daily_ret) / daily_vol
    noise = np.exp(rng.normal(0.0, 0.35, (n_days, n_tickers)))
    volume = pd.DataFrame(base_vol * activity * noise, index=dates, columns=tickers)

    securities = pd.DataFrame(
        {
            "sector": [SECTORS[i % len(SECTORS)] for i in range(n_tickers)],
            "market_cap": prices[-1] * rng.uniform(5e6, 4e8, n_tickers),
            "adv_20d": (adj_close * volume).iloc[-20:].mean().values,
            "is_active": True,
        },
        index=pd.Index(tickers, name="ticker"),
    )

    universe = pd.DataFrame(True, index=dates, columns=tickers)

    # Expected IC from the design: signal contributes h*k*sigma of drift against
    # sigma*sqrt(h) of noise over an h-day horizon.
    h = 21
    a = h * momentum_strength
    expected_ic = a / np.sqrt(a**2 + h)

    truth = {
        "momentum_strength": momentum_strength,
        "lookback": lookback,
        "skip": skip,
        "n_tickers": n_tickers,
        "n_days": n_days,
        "first_signal_row": first_signal,
        "expected_ic_21d": float(expected_ic),
        "seed": seed,
    }

    panel = Panel(
        adj_close=adj_close,
        volume=volume,
        close=adj_close.copy(),
        securities=securities,
        fundamentals=None,
        universe=universe,
    )
    return panel, truth


@pytest.fixture(scope="session")
def synthetic() -> Tuple[Panel, Dict]:
    return make_synthetic_panel()


@pytest.fixture(scope="session")
def synthetic_panel(synthetic) -> Panel:
    return synthetic[0]


@pytest.fixture(scope="session")
def synthetic_truth(synthetic) -> Dict:
    return synthetic[1]


@pytest.fixture(scope="session")
def designed_ic() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sized so the standard error of the mean IC is ~0.003 — small enough that"""
    return make_factor_and_returns(n_dates=240, n_tickers=400, ic=0.05)

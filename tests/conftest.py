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
        "drift": drift,
        "tickers": tickers,
        "dates": dates,
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
    """Sized so the standard error of the mean IC is about 0.003."""
    return make_factor_and_returns(n_dates=240, n_tickers=400, ic=0.05)


def make_synthetic_fundamentals(
    truth: Dict,
    seed: int = 21,
    growth_link: float = 0.7,
    report_lag_days: int = 45,
) -> pd.DataFrame:
    """Quarterly point-in-time fundamentals for the synthetic market.

    Two things make this a real test rather than filler.

    `report_date` sits `report_lag_days` AFTER `period_end`, jittered, exactly
    as a real filing does. A factor that joins on period_end instead will look
    brilliant here and be worthless live, which is the bug the whole schema
    exists to prevent.

    Revenue growth is tied to each ticker's true return drift via `growth_link`,
    so a correctly built growth factor has something real to find and a broken
    one has nothing.
    """
    rng = np.random.default_rng(seed)
    tickers = truth["tickers"]
    dates = truth["dates"]
    drift = np.asarray(truth["drift"], dtype=float)

    # standardize drift -> the growth signal every ticker's revenue follows
    signal = (drift - drift.mean()) / (drift.std() or 1.0)

    start, end = dates[0], dates[-1]
    period_ends = pd.date_range(start - pd.Timedelta(days=400), end, freq="QE")

    base_revenue = rng.uniform(2e8, 4e10, len(tickers))
    base_margin = rng.uniform(0.08, 0.42, len(tickers))
    base_shares = rng.uniform(2e8, 6e9, len(tickers))

    rows = []
    for i, ticker in enumerate(tickers):
        growth = 0.02 + growth_link * 0.06 * signal[i]      # annual revenue growth
        revenue = base_revenue[i]
        margin = base_margin[i]
        shares = base_shares[i]

        for q, period_end in enumerate(period_ends):
            revenue *= (1.0 + growth / 4.0) * (1.0 + rng.normal(0, 0.02))
            margin = float(np.clip(margin + 0.004 * signal[i] + rng.normal(0, 0.006), 0.01, 0.6))
            shares *= (1.0 - rng.uniform(0.0, 0.004))        # buybacks
            operating_income = revenue * margin
            net_income = operating_income * rng.uniform(0.70, 0.85)

            lag = report_lag_days + int(rng.integers(-8, 15))
            rows.append({
                "ticker": ticker,
                "period_end": period_end.date(),
                "report_date": (period_end + pd.Timedelta(days=lag)).date(),
                "fiscal_period": f"Q{(q % 4) + 1}",
                "revenue": revenue,
                "gross_profit": revenue * (margin + 0.2),
                "operating_income": operating_income,
                "net_income": net_income,
                "eps_diluted": net_income / shares,
                "gross_margin": margin + 0.2,
                "operating_margin": margin,
                "free_cash_flow": operating_income * rng.uniform(0.5, 0.9),
                "capex": revenue * rng.uniform(0.02, 0.09),
                "roic": margin * rng.uniform(0.5, 1.4),
                "total_debt": revenue * rng.uniform(0.1, 1.2),
                "cash_and_equivalents": revenue * rng.uniform(0.05, 0.6),
                "shares_diluted": shares,
                "source_url": f"https://www.sec.gov/synthetic/{ticker}/{period_end.date()}",
            })

    frame = pd.DataFrame(rows)
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    return frame.sort_values(["ticker", "report_date"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic_fundamentals(synthetic_truth) -> pd.DataFrame:
    return make_synthetic_fundamentals(synthetic_truth)


@pytest.fixture(scope="session")
def panel_with_fundamentals(synthetic, synthetic_fundamentals) -> Panel:
    panel = synthetic[0]
    return Panel(
        adj_close=panel.adj_close, volume=panel.volume, close=panel.close,
        securities=panel.securities, universe=panel.universe,
        fundamentals=synthetic_fundamentals,
    )

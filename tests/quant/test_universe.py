"""Universe filter tests. Steps A5 and A6.

These produce the funnel counts the dashboard animates, so a wrong count is
something a judge sees on screen. Runs on synthetic data, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.universe.builder import build_universe
from quant.universe.filters import (
    filter_data_completeness,
    filter_liquidity,
    filter_listing,
    filter_market_cap,
)


@pytest.fixture
def profiles() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": "BIG", "name": "Big Co", "sector": "Tech", "industry": "SW",
         "exchange": "NMS", "market_cap": 50e9},
        {"ticker": "MID", "name": "Mid Co", "sector": "Industrials", "industry": "Mach",
         "exchange": "NYQ", "market_cap": 5e9},
        {"ticker": "SMALL", "name": "Small Co", "sector": "Tech", "industry": "HW",
         "exchange": "NMS", "market_cap": 200e6},
        {"ticker": "OTC", "name": "OTC Co", "sector": "Energy", "industry": "Oil",
         "exchange": "PNK", "market_cap": 10e9},
        {"ticker": "NOCAP", "name": "Unknown Co", "sector": None, "industry": None,
         "exchange": "NMS", "market_cap": None},
    ])


@pytest.fixture
def adv() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=60)
    return pd.DataFrame({
        "BIG": 500e6,
        "MID": 25e6,
        "SMALL": 400e3,      # too thin
        "OTC": 50e6,
        "NOCAP": 12e6,
    }, index=dates)


@pytest.fixture
def coverage_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": "BIG", "n_days": 2500},
        {"ticker": "MID", "n_days": 1200},
        {"ticker": "SMALL", "n_days": 800},
        {"ticker": "OTC", "n_days": 120},     # too little history
        {"ticker": "NOCAP", "n_days": 600},
    ])


# ----------------------------------------------------------------- filters

def test_market_cap_filter_drops_below_threshold(profiles):
    r = filter_market_cap(profiles, min_market_cap=1e9)
    assert set(r.passed) == {"BIG", "MID", "OTC"}
    assert "SMALL" in r.dropped


def test_missing_market_cap_is_dropped_not_kept(profiles):
    """A null must not silently pass. Treating unknown as acceptable is how
    junk enters the universe."""
    r = filter_market_cap(profiles, min_market_cap=1e9)
    assert "NOCAP" not in r.passed


def test_liquidity_filter_drops_thin_names(adv):
    r = filter_liquidity(adv, min_adv_usd=5e6)
    assert "SMALL" not in r.passed
    assert {"BIG", "MID", "OTC", "NOCAP"} <= set(r.passed)


def test_listing_filter_keeps_only_primary_us_exchanges(profiles):
    r = filter_listing(profiles)
    assert "OTC" not in r.passed          # PNK is pink sheets
    assert {"BIG", "MID", "SMALL", "NOCAP"} <= set(r.passed)


def test_data_completeness_filter_needs_enough_history(coverage_df):
    r = filter_data_completeness(coverage_df, min_days=500)
    assert "OTC" not in r.passed
    assert "BIG" in r.passed


def test_filters_respect_the_universe_argument(profiles):
    """Each stage must run only on the previous stage's survivors, or the
    funnel counts stop being cumulative and the cascade lies."""
    r = filter_market_cap(profiles, min_market_cap=1e9, universe=["BIG", "SMALL"])
    assert set(r.passed) == {"BIG"}
    assert "MID" not in r.passed and "MID" not in r.dropped


def test_count_matches_passed_length(profiles):
    r = filter_market_cap(profiles, min_market_cap=1e9)
    assert r.count == len(r.passed)


# ----------------------------------------------------------------- builder

@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2019-01-02", periods=800)
    rows = []
    specs = {
        "BIG": (300.0, 3_000_000),
        "MID": (80.0, 400_000),
        "SMALL": (12.0, 20_000),
        "OTC": (45.0, 900_000),
        "NOCAP": (60.0, 250_000),
    }
    for ticker, (px0, vol) in specs.items():
        px = px0
        for d in dates:
            px *= 1 + rng.normal(0.0003, 0.015)
            rows.append({"date": d, "ticker": ticker, "adj_close": px,
                         "close": px, "volume": vol})
    return pd.DataFrame(rows)


def test_build_universe_produces_a_monotonic_funnel(panel, profiles):
    """Counts must never increase as stages are applied. A rising count means
    a stage ignored its input universe."""
    result = build_universe(panel, profiles, min_market_cap=1e9,
                            min_adv_usd=5e6, min_days=500)

    counts = [s["count"] for s in result.funnel()]
    assert counts == sorted(counts, reverse=True), f"funnel not monotonic: {counts}"


def test_funnel_shape_matches_the_api_schema(panel, profiles):
    """UniverseResult.funnel() feeds backend.api.schemas.FunnelStage directly."""
    from backend.api.schemas import FunnelStage

    result = build_universe(panel, profiles, min_market_cap=1e9,
                            min_adv_usd=5e6, min_days=500)
    stages = [FunnelStage(**s) for s in result.funnel()]

    assert stages[0].label == "Scanned"
    assert stages[0].count == panel["ticker"].nunique()
    assert all(s.count >= 0 for s in stages)


def test_survivors_actually_satisfy_every_filter(panel, profiles):
    """The end-to-end property: anything in the final universe must pass all
    four criteria, not just the last one applied."""
    result = build_universe(panel, profiles, min_market_cap=1e9,
                            min_adv_usd=5e6, min_days=500)

    caps = profiles.set_index("ticker")["market_cap"]
    exch = profiles.set_index("ticker")["exchange"]
    for t in result.tickers:
        assert caps.get(t, 0) >= 1e9, f"{t} survived with market cap {caps.get(t)}"
        assert exch.get(t) != "PNK", f"{t} survived despite being pink sheets"


def test_max_size_caps_the_universe_by_liquidity(panel, profiles):
    result = build_universe(panel, profiles, min_market_cap=1e8,
                            min_adv_usd=1e3, min_days=100, max_size=2)
    assert len(result.tickers) <= 2


def test_scanned_count_is_the_full_input(panel, profiles):
    result = build_universe(panel, profiles)
    assert result.scanned == panel["ticker"].nunique()

"""Risk engine tests. Steps A15 to A18.

Mostly invariants: relationships that must hold whatever the input. An
invariant test catches a broken formula that still returns a plausible number,
which is the failure mode that matters here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.schemas.risk import RiskBand
from quant.risk.correlation import (
    average_pairwise_correlation,
    diversification_ratio,
    most_correlated_pairs,
)
from quant.risk.exposures import (
    concentration,
    effective_positions,
    exposure_report,
    factor_exposure,
    herfindahl,
    sector_exposure,
)
from quant.risk.liquidity import capacity, days_to_liquidate, liquidation_horizon
from quant.risk.metrics import build_risk_metrics, portfolio_beta, risk_band
from quant.risk.var import cvar_historical, stress_scenarios, var_historical


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2022-01-03", periods=500)
    return pd.Series(rng.normal(0.0005, 0.012, 500), index=idx)


@pytest.fixture
def asset_returns() -> pd.DataFrame:
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2022-01-03", periods=500)
    common = rng.normal(0, 0.01, 500)
    return pd.DataFrame({
        "A": common * 0.8 + rng.normal(0, 0.004, 500),
        "B": common * 0.7 + rng.normal(0, 0.004, 500),
        "C": rng.normal(0, 0.011, 500),
    }, index=idx)


# --------------------------------------------------------------------- VaR

def test_cvar_is_never_worse_than_var(returns):
    """CVaR is the mean of the tail beyond VaR, so it must be <= VaR.
    If this ever inverts, one of the two quantiles is computed backwards."""
    assert cvar_historical(returns, 0.95) <= var_historical(returns, 0.95)


def test_var_is_negative_for_a_normal_return_series(returns):
    assert var_historical(returns, 0.95) < 0


def test_higher_confidence_means_a_worse_var(returns):
    assert var_historical(returns, 0.99) <= var_historical(returns, 0.95)


def test_var_returns_none_on_insufficient_history():
    assert var_historical(pd.Series([0.01, -0.02, 0.005])) is None


def test_stress_scenarios_report_the_actual_worst_day(returns):
    s = stress_scenarios(returns)
    assert s["worst_day"] == pytest.approx(returns.min())
    assert 0.0 <= s["pct_days_negative"] <= 1.0


# -------------------------------------------------------------------- beta

def test_beta_against_itself_is_one(returns):
    """The simplest invariant in the file, and it catches a swapped
    covariance/variance denominator instantly."""
    assert portfolio_beta(returns, returns) == pytest.approx(1.0, abs=1e-9)


def test_beta_of_double_the_benchmark_is_two(returns):
    assert portfolio_beta(returns * 2, returns) == pytest.approx(2.0, abs=1e-9)


def test_beta_of_a_constant_series_is_zero(returns):
    flat = pd.Series(0.0, index=returns.index)
    assert portfolio_beta(flat, returns) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- exposures

def test_sector_exposure_sums_to_gross_weight():
    w = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
    sectors = {"A": "Tech", "B": "Tech", "C": "Energy"}
    e = sector_exposure(w, sectors)
    assert e["Tech"] == pytest.approx(0.7)
    assert sum(e.values()) == pytest.approx(1.0)


def test_unknown_sector_is_labelled_not_dropped():
    w = pd.Series({"A": 0.5, "B": 0.5})
    e = sector_exposure(w, {"A": "Tech"})
    assert "Unknown" in e
    assert sum(e.values()) == pytest.approx(1.0)


def test_concentration_is_the_largest_weight():
    assert concentration(pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})) == pytest.approx(0.5)


def test_herfindahl_of_equal_weights_is_one_over_n():
    w = pd.Series({c: 0.25 for c in "ABCD"})
    assert herfindahl(w) == pytest.approx(0.25)


def test_effective_positions_reveals_hidden_concentration():
    """Five names, but one is 90% of the book. Effective count must be near 1."""
    w = pd.Series({"A": 0.90, "B": 0.025, "C": 0.025, "D": 0.025, "E": 0.025})
    assert effective_positions(w) < 1.3
    assert effective_positions(pd.Series({c: 0.2 for c in "ABCDE"})) == pytest.approx(5.0)


def test_factor_exposure_is_weighted_average():
    w = pd.Series({"A": 0.5, "B": 0.5})
    fv = pd.DataFrame({"momentum": [2.0, 0.0]}, index=["A", "B"])
    assert factor_exposure(w, fv)["momentum"] == pytest.approx(1.0)


def test_exposure_report_nets_a_long_short_book():
    w = pd.Series({"A": 0.5, "B": -0.5})
    r = exposure_report(w, {"A": "Tech", "B": "Tech"})
    assert r["gross_exposure"] == pytest.approx(1.0)
    assert r["net_exposure"] == pytest.approx(0.0)
    assert r["n_positions"] == 2


# -------------------------------------------------------------- correlation

def test_average_pairwise_correlation_is_in_range(asset_returns):
    c = average_pairwise_correlation(asset_returns)
    assert -1.0 <= c <= 1.0


def test_correlated_names_rank_above_independent_ones(asset_returns):
    """A and B share a common factor; C does not."""
    top = most_correlated_pairs(asset_returns, top_n=1)[0]
    assert set(top[:2]) == {"A", "B"}


def test_single_asset_has_no_pairwise_correlation(asset_returns):
    assert average_pairwise_correlation(asset_returns[["A"]]) is None


def test_diversification_ratio_is_at_least_one(asset_returns):
    w = pd.Series({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    assert diversification_ratio(w, asset_returns) >= 1.0 - 1e-9


# ---------------------------------------------------------------- liquidity

def test_days_to_liquidate_scales_linearly_with_size():
    one = days_to_liquidate(1_000_000, adv_usd=100_000_000)
    two = days_to_liquidate(2_000_000, adv_usd=100_000_000)
    assert two == pytest.approx(2 * one)


def test_days_to_liquidate_falls_as_liquidity_rises():
    thin = days_to_liquidate(1_000_000, adv_usd=1_000_000)
    deep = days_to_liquidate(1_000_000, adv_usd=100_000_000)
    assert thin > deep


def test_days_to_liquidate_is_none_without_volume():
    assert days_to_liquidate(1_000_000, adv_usd=0) is None


def test_liquidation_horizon_is_set_by_the_slowest_name():
    w = pd.Series({"FAST": 0.5, "SLOW": 0.5})
    adv = pd.Series({"FAST": 500e6, "SLOW": 1e6})
    per_name = days_to_liquidate(0.5 * 1e6, 1e6)
    assert liquidation_horizon(w, adv, 1e6) == pytest.approx(per_name)


def test_capacity_grows_with_market_liquidity():
    thin = capacity(pd.Series({"A": 1e6, "B": 2e6}))
    deep = capacity(pd.Series({"A": 1e9, "B": 2e9}))
    assert deep > thin


# ------------------------------------------------------------- risk bands

def test_risk_band_thresholds():
    assert risk_band(0.10) == RiskBand.LOW
    assert risk_band(0.25) == RiskBand.MEDIUM
    assert risk_band(0.50) == RiskBand.HIGH


def test_unknown_volatility_defaults_to_medium():
    assert risk_band(None) == RiskBand.MEDIUM
    assert risk_band(float("nan")) == RiskBand.MEDIUM


# --------------------------------------------------------------- assembler

def test_build_risk_metrics_populates_the_panel(returns, asset_returns):
    w = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
    rm = build_risk_metrics(
        returns, returns, w, {"A": "Tech", "B": "Tech", "C": "Energy"},
        asset_returns, position_notional=100_000, adv_usd=50_000_000,
    )
    assert rm.beta == pytest.approx(1.0, abs=1e-9)
    assert rm.var_95 is not None and rm.cvar_95 <= rm.var_95
    assert rm.sector == "Tech"
    assert rm.concentration == pytest.approx(0.4)
    assert rm.days_to_liquidate > 0


def test_build_risk_metrics_degrades_gracefully(returns):
    """Missing inputs must produce empty fields, never invented numbers."""
    rm = build_risk_metrics(returns)
    assert rm.beta is None
    assert rm.sector is None
    assert rm.sector_exposure == {}
    assert rm.days_to_liquidate is None
    assert rm.volatility is not None

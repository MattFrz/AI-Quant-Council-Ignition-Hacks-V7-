"""Backtest engine tests. Runs on synthetic data - no network, no cache."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.costs import CostModel
from quant.backtest.engine import Backtester, BacktestConfig
from quant.backtest.metrics import (
    annualized_return,
    max_drawdown,
    equity_curve,
    sharpe,
    sortino,
    turnover,
    win_rate,
)
from quant.backtest.slippage import FixedBpsSlippage, ParticipationRateSlippage
from quant.backtest.walk_forward import make_splits


@pytest.fixture
def market():
    """20 names, 3 years of daily bars, one with a genuine upward drift."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-04", periods=750)
    tickers = [f"T{i:02d}" for i in range(20)]

    prices = {}
    for i, t in enumerate(tickers):
        drift = 0.0006 if i < 5 else 0.0
        steps = rng.normal(drift, 0.012, len(dates))
        prices[t] = 100.0 * np.exp(np.cumsum(steps))

    close = pd.DataFrame(prices, index=dates)
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=tickers)
    adv = close * volume
    return close, adv


# ------------------------------------------------------------------ metrics

def test_sharpe_of_constant_returns_is_zero():
    r = pd.Series([0.001] * 100)
    assert sharpe(r) == 0.0


def test_sharpe_positive_for_upward_drift():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0008, 0.01, 500))
    assert sharpe(r) > 0


def test_max_drawdown_is_negative_or_zero():
    eq = equity_curve(pd.Series([0.05, -0.10, 0.02, -0.03]))
    assert max_drawdown(eq) < 0


def test_max_drawdown_of_monotonic_rise_is_zero():
    eq = equity_curve(pd.Series([0.01] * 50))
    assert max_drawdown(eq) == pytest.approx(0.0, abs=1e-12)


def test_sortino_rewards_upside_volatility_over_downside():
    """Two series, same mean. One earns it through big up days with a mild
    downside; the other has the same dispersion pointed downward."""
    rng = np.random.default_rng(3)
    base = rng.normal(0.0005, 0.004, 400)

    upside = pd.Series(np.concatenate([base, [0.06, 0.05, 0.04]]))
    downside = pd.Series(np.concatenate([base, [-0.06, -0.05, -0.04]]))

    assert sortino(upside) > sortino(downside)
    assert sortino(upside) > sharpe(upside)


def test_sortino_of_degenerate_downside_is_zero():
    """Identical losses give zero downside deviation - report 0, not infinity."""
    assert sortino(pd.Series([0.001] * 200 + [-0.001] * 10)) == 0.0


def test_win_rate_ignores_flat_days():
    assert win_rate(pd.Series([0.01, -0.01, 0.0, 0.0])) == pytest.approx(0.5)


def test_turnover_zero_for_static_book():
    w = pd.DataFrame({"A": [0.5] * 10, "B": [0.5] * 10})
    assert turnover(w) == pytest.approx(0.0)


# ------------------------------------------------------------------- costs

def test_illiquid_names_cost_more_to_trade():
    cm = CostModel(commission_bps=1.0)
    assert cm.total(100_000, adv_usd=5_000) > cm.total(100_000, adv_usd=2_000_000_000)


def test_cost_scales_with_notional():
    cm = CostModel(commission_bps=1.0)
    assert cm.total(200_000, 1e9) == pytest.approx(2 * cm.total(100_000, 1e9))


# ---------------------------------------------------------------- slippage

def test_slippage_grows_with_participation():
    m = ParticipationRateSlippage()
    small = m.impact_bps(10_000, adv_usd=100_000_000)
    large = m.impact_bps(10_000_000, adv_usd=100_000_000)
    assert large > small


def test_slippage_is_sublinear():
    """Square-root impact: 100x the size costs less than 100x the bps."""
    m = ParticipationRateSlippage()
    small = m.impact_bps(100_000, adv_usd=1e9)
    big = m.impact_bps(10_000_000, adv_usd=1e9)
    assert big < small * 100


def test_slippage_always_works_against_you():
    m = ParticipationRateSlippage()
    buy = m.fill_price(100.0, quantity=1000, adv_usd=1e7)
    sell = m.fill_price(100.0, quantity=-1000, adv_usd=1e7)
    assert buy > 100.0 > sell


# ------------------------------------------------------------------ engine

def test_engine_runs_and_produces_a_valid_result(market):
    close, adv = market
    signal = close.pct_change(60, fill_method=None)

    run = Backtester(BacktestConfig(rebalance_freq="ME", max_names=5)).run(
        signal=signal, close=close, adv=adv
    )

    assert len(run.returns) == len(close)
    assert run.result.sharpe is not None
    assert run.result.max_drawdown <= 0
    assert len(run.result.equity_curve) > 0
    assert run.result.n_trades > 0


def test_position_cap_binds_at_rebalance(market):
    """The cap constrains target weights on rebalance days."""
    close, adv = market
    signal = close.pct_change(60, fill_method=None)

    run = Backtester(
        BacktestConfig(rebalance_freq="ME", max_names=3, max_position_pct=10.0)
    ).run(signal=signal, close=close, adv=adv)

    rebalance_days = sorted({f.timestamp for f in run.fills})
    on_rebalance = run.weights.loc[run.weights.index.isin(rebalance_days)]
    assert on_rebalance.abs().to_numpy().max() <= 0.1005


def test_positions_drift_only_modestly_between_rebalances(market):
    """Drift above the cap is expected and real; it should stay bounded.
    A blowout here means the rebalance schedule is too loose."""
    close, adv = market
    signal = close.pct_change(60, fill_method=None)

    run = Backtester(
        BacktestConfig(rebalance_freq="W-FRI", max_names=3, max_position_pct=10.0)
    ).run(signal=signal, close=close, adv=adv)

    assert run.weights.abs().to_numpy().max() <= 0.13


def test_liquidity_cap_limits_trade_size():
    """With near-zero ADV the engine must refuse to build a full position."""
    dates = pd.bdate_range("2022-01-03", periods=120)
    close = pd.DataFrame({"A": np.linspace(100, 150, 120)}, index=dates)
    tiny_adv = pd.DataFrame({"A": [1_000.0] * 120}, index=dates)
    signal = pd.DataFrame({"A": [1.0] * 120}, index=dates)

    run = Backtester(
        BacktestConfig(rebalance_freq="W-FRI", max_names=1, initial_capital=1_000_000)
    ).run(signal=signal, close=close, adv=tiny_adv)

    assert run.weights["A"].max() < 0.05
    assert any(f.ticker == "A" for f in run.fills)


def test_costs_reduce_returns(market):
    """The same signal must perform worse once trading costs are charged."""
    close, adv = market
    signal = close.pct_change(60, fill_method=None)
    cfg = BacktestConfig(rebalance_freq="W-FRI", max_names=5)

    free = Backtester(cfg, cost_model=CostModel(commission_bps=0.0),
                      slippage_model=FixedBpsSlippage(bps=0.0)).run(signal, close, adv)
    costly = Backtester(cfg, cost_model=CostModel(commission_bps=10.0),
                        slippage_model=FixedBpsSlippage(bps=50.0)).run(signal, close, adv)

    assert costly.result.total_return < free.result.total_return


def test_engine_applies_execution_lag(market):
    """A perfect-foresight signal must not produce a free lunch, because the
    engine lags it before trading."""
    close, adv = market
    tomorrow = close.shift(-1) / close - 1.0    # tomorrow's return, known today

    run = Backtester(BacktestConfig(rebalance_freq="D", max_names=5)).run(
        signal=tomorrow, close=close, adv=adv
    )
    # With the lag applied this is yesterday's return - no longer clairvoyant.
    assert run.result.sharpe < 8.0


# ------------------------------------------------------------ walk-forward

def test_splits_never_overlap():
    dates = pd.bdate_range("2015-01-01", periods=2000)
    splits = make_splits(dates, train_years=3, test_years=1)
    assert len(splits) > 0
    for s in splits:
        assert s.train_end < s.test_start


def test_splits_move_forward_in_time():
    dates = pd.bdate_range("2015-01-01", periods=2000)
    splits = make_splits(dates, train_years=3, test_years=1)
    for a, b in zip(splits, splits[1:]):
        assert b.test_start > a.test_start

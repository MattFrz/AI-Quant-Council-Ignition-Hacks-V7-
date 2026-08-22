"""Run a backtest end to end from the cached data. Lane A smoke test.

    python scripts/run_backtest.py                     # 12-1 momentum, monthly
    python scripts/run_backtest.py --signal reversal   # short-term reversal
    python scripts/run_backtest.py --walk-forward      # out-of-sample only
    python scripts/run_backtest.py --freq W-FRI --names 5 --max-pos 15

Exercises the whole lane: load -> universe -> signal -> leakage guard ->
engine -> metrics -> benchmark -> risk. If this prints numbers, Lane A works.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from backend.config import settings  # noqa: E402
from data.pipelines.prices import (  # noqa: E402
    average_dollar_volume,
    daily_returns,
    load_prices,
    to_wide,
)
from quant.backtest.benchmark import compare  # noqa: E402
from quant.backtest.engine import Backtester, BacktestConfig  # noqa: E402
from quant.backtest.leakage_guards import assert_causal, enforce_execution_lag  # noqa: E402
from quant.backtest.walk_forward import make_splits, run_walk_forward  # noqa: E402
from quant.risk.exposures import exposure_report  # noqa: E402
from quant.risk.liquidity import capacity, liquidation_horizon  # noqa: E402
from quant.risk.metrics import build_risk_metrics, rolling_volatility  # noqa: E402


def make_signal(close: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Placeholder signals so Lane A is testable before Lane B lands.

    These are NOT the alpha model - Nalin's quant/alpha/composite.py is. They
    exist so the engine can be exercised and so there is a baseline to beat.
    """
    if kind == "momentum":
        return close.pct_change(252, fill_method=None) - close.pct_change(21, fill_method=None)
    if kind == "reversal":
        return -close.pct_change(5, fill_method=None)
    if kind == "lowvol":
        return -close.pct_change(fill_method=None).rolling(60).std()
    raise ValueError(f"unknown signal '{kind}'")


def pct(x) -> str:
    return "n/a" if x is None else f"{x:+.2%}"


def main() -> int:
    p = argparse.ArgumentParser(description="Run a Lane A backtest")
    p.add_argument("--signal", default="momentum", choices=["momentum", "reversal", "lowvol"])
    p.add_argument("--freq", default="ME", help="rebalance: D, W-FRI, ME")
    p.add_argument("--names", type=int, default=8, help="max positions")
    p.add_argument("--max-pos", type=float, default=20.0, help="max position %%")
    p.add_argument("--walk-forward", action="store_true")
    p.add_argument("--train-years", type=float, default=2.0)
    p.add_argument("--test-years", type=float, default=1.0)
    args = p.parse_args()

    try:
        panel = load_prices()
    except FileNotFoundError as exc:
        print(f"\n{exc}\n")
        return 1

    close = to_wide(panel, "adj_close")
    volume = to_wide(panel, "volume")
    adv = average_dollar_volume(close, volume)
    rets = daily_returns(close)
    vol = rolling_volatility(rets, 60) / (252 ** 0.5)

    bench_ticker = settings.benchmark_ticker
    if bench_ticker not in close.columns:
        print(f"warning: benchmark {bench_ticker} not in cache - running without it")
        bench = None
    else:
        bench = rets[bench_ticker].dropna()
        close = close.drop(columns=[bench_ticker])
        adv = adv.drop(columns=[bench_ticker])
        vol = vol.drop(columns=[bench_ticker])
        rets = rets.drop(columns=[bench_ticker])

    print(f"\n{'='*66}")
    print(f"  universe {close.shape[1]} names   "
          f"{close.index[0].date()} to {close.index[-1].date()}   "
          f"{len(close):,} days")
    print(f"{'='*66}")

    signal = make_signal(close, args.signal)

    assert_causal(enforce_execution_lag(signal, 1), rets, args.signal)
    print(f"  leakage guard   PASS  ({args.signal} is causal)")

    cfg = BacktestConfig(
        rebalance_freq=args.freq,
        max_names=args.names,
        max_position_pct=args.max_pos,
        strategy_name=args.signal,
    )

    run = Backtester(cfg).run(
        signal=signal, close=close, adv=adv, volatility=vol, benchmark_returns=bench
    )
    r = run.result

    print(f"\n  IN-SAMPLE ({args.freq} rebalance, top {args.names}, {args.max_pos:.0f}% cap)")
    print(f"    annualized      {pct(r.annualized_return)}")
    print(f"    excess vs {bench_ticker:<5} {pct(r.excess_return)}")
    print(f"    sharpe          {r.sharpe:.2f}")
    print(f"    sortino         {r.sortino:.2f}")
    print(f"    volatility      {pct(r.volatility)}")
    print(f"    max drawdown    {pct(r.max_drawdown)}")
    print(f"    turnover        {r.turnover:.2f}x")
    print(f"    win rate        {r.win_rate:.1%}")
    print(f"    trades          {r.n_trades}")
    print(f"    cost drag       {run.cost_drag:.2%} of capital")

    if args.walk_forward:
        splits = make_splits(close.index, args.train_years, args.test_years)
        if not splits:
            print("\n  not enough history for walk-forward")
        else:
            wf = run_walk_forward(
                signal_fn=lambda _train_dates: signal,
                close=close, adv=adv, volatility=vol, benchmark_returns=bench,
                config=cfg, train_years=args.train_years, test_years=args.test_years,
                strategy_name=f"{args.signal}_wf",
            )
            print(f"\n  OUT-OF-SAMPLE ({len(splits)} walk-forward splits)")
            print(f"    annualized      {pct(wf.annualized_return)}")
            print(f"    excess vs {bench_ticker:<5} {pct(wf.excess_return)}")
            print(f"    sharpe          {wf.sharpe:.2f}")
            print(f"    max drawdown    {pct(wf.max_drawdown)}")
            print("    NOTE: signal_fn ignores train dates here, so this validates")
            print("          the machinery, not a fitted model. Real validation")
            print("          arrives when Lane B fits weights on train only.")

    if bench is not None:
        c = compare(run.returns, bench)
        print(f"\n  VS BENCHMARK")
        print(f"    beta            {c['beta']:.2f}")
        print(f"    CAPM alpha      {pct(c['capm_alpha'])}")
        print(f"    info ratio      {c['information_ratio']:.2f}")
        if c["beta"] > 1.1:
            print(f"    ^ beta > 1.1: much of the return is market exposure, not alpha")

    final_w = run.weights.iloc[-1]
    final_w = final_w[final_w.abs() > 1e-6]
    if len(final_w):
        rm = build_risk_metrics(
            run.returns, bench, final_w, {}, rets[final_w.index],
            position_notional=cfg.initial_capital * 0.05,
            adv_usd=float(adv.iloc[-1].median()),
        )
        er = exposure_report(final_w)
        print(f"\n  RISK")
        print(f"    beta            {rm.beta:.2f}" if rm.beta else "    beta            n/a")
        print(f"    VaR 95          {pct(rm.var_95)}")
        print(f"    CVaR 95         {pct(rm.cvar_95)}")
        print(f"    avg correlation {rm.avg_correlation:.2f}" if rm.avg_correlation else "")
        print(f"    risk band       {rm.risk_band.value}")
        print(f"    positions       {er['n_positions']} "
              f"(effective {er['effective_positions']:.1f})")
        print(f"    concentration   {er['concentration']:.1%}")
        horizon = liquidation_horizon(final_w, adv.iloc[-1], cfg.initial_capital)
        if horizon:
            print(f"    liquidate book  {horizon:.2f} days")
        cap = capacity(adv.iloc[-1])
        if cap:
            print(f"    capacity        ${cap/1e6:,.0f}M")

    print(f"\n{'='*66}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

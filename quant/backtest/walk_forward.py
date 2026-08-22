"""Walk-forward validation. Step A14 - the file that answers "did you overfit?"

Rolling train/test splits. Weights are fitted on train and applied, untouched,
to test. The stitched test returns are the only performance number worth
showing a judge; everything computed in-sample is a fitting diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd

from backend.core.logging import get_logger
from data.schemas.backtest_result import BacktestResult, BacktestWindow
from quant.backtest.engine import Backtester, BacktestConfig, BacktestRun
from quant.backtest.leakage_guards import assert_window_clean
from quant.backtest.metrics import build_result

log = get_logger(__name__)


@dataclass
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_window(self) -> BacktestWindow:
        return BacktestWindow(
            train_start=self.train_start.date(),
            train_end=self.train_end.date(),
            test_start=self.test_start.date(),
            test_end=self.test_end.date(),
        )

    def __str__(self) -> str:
        return (f"train {self.train_start.date()}..{self.train_end.date()} | "
                f"test {self.test_start.date()}..{self.test_end.date()}")


def make_splits(
    dates: pd.DatetimeIndex,
    train_years: float = 3.0,
    test_years: float = 1.0,
    step_years: Optional[float] = None,
    anchored: bool = False,
) -> List[Split]:
    """Rolling (or anchored) walk-forward splits.

    anchored=True keeps train_start fixed and grows the window; False slides it.
    """
    dates = pd.DatetimeIndex(sorted(dates))
    if len(dates) == 0:
        return []

    step_years = step_years or test_years
    train_td = pd.Timedelta(days=int(365.25 * train_years))
    test_td = pd.Timedelta(days=int(365.25 * test_years))
    step_td = pd.Timedelta(days=int(365.25 * step_years))

    splits: List[Split] = []
    origin = dates[0]
    train_start = origin

    while True:
        train_end = train_start + train_td
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + test_td

        if test_start > dates[-1]:
            break
        test_end = min(test_end, dates[-1])

        if len(dates[(dates >= test_start) & (dates <= test_end)]) < 20:
            break

        assert_window_clean(train_end, test_start)
        splits.append(Split(
            train_start=origin if anchored else train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))

        train_start = train_start + step_td
        if not anchored and train_start + train_td >= dates[-1]:
            break
        if anchored and train_end + step_td >= dates[-1]:
            break

    return splits


def run_walk_forward(
    signal_fn: Callable[[pd.DatetimeIndex], pd.DataFrame],
    close: pd.DataFrame,
    adv: pd.DataFrame,
    volatility: Optional[pd.DataFrame] = None,
    benchmark_returns: Optional[pd.Series] = None,
    config: Optional[BacktestConfig] = None,
    train_years: float = 3.0,
    test_years: float = 1.0,
    strategy_name: str = "walk_forward",
) -> BacktestResult:
    """`signal_fn` receives the TRAIN dates and must return a signal frame.

    It may fit anything it likes on those dates. The engine then evaluates it on
    the test dates only. Nothing from the test window reaches signal_fn.
    """
    config = config or BacktestConfig()
    splits = make_splits(close.index, train_years, test_years)

    if not splits:
        raise ValueError(
            f"not enough history for {train_years}y train + {test_years}y test "
            f"({len(close.index)} days available)"
        )

    log.info("walk-forward: %d splits", len(splits))

    test_returns: List[pd.Series] = []
    test_weights: List[pd.DataFrame] = []
    total_trades = 0
    engine = Backtester(config=config)

    for split in splits:
        log.info("  %s", split)
        train_dates = close.index[
            (close.index >= split.train_start) & (close.index <= split.train_end)
        ]
        signal = signal_fn(train_dates)

        test_mask = (close.index >= split.test_start) & (close.index <= split.test_end)
        test_dates = close.index[test_mask]

        run = engine.run(
            signal=signal.reindex(index=test_dates),
            close=close.loc[test_dates],
            adv=adv.loc[test_dates],
            volatility=volatility.loc[test_dates] if volatility is not None else None,
            window=split.to_window(),
        )
        test_returns.append(run.returns)
        test_weights.append(run.weights)
        total_trades += len(run.fills)

    stitched = pd.concat(test_returns).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    weights = pd.concat(test_weights).sort_index()
    weights = weights[~weights.index.duplicated(keep="first")].fillna(0.0)

    window = BacktestWindow(
        train_start=splits[0].train_start.date(),
        train_end=splits[0].train_end.date(),
        test_start=splits[0].test_start.date(),
        test_end=splits[-1].test_end.date(),
    )

    return build_result(
        strategy_name=strategy_name,
        returns=stitched,
        weights=weights,
        window=window,
        universe_size=int(close.shape[1]),
        benchmark_returns=benchmark_returns,
        n_trades=total_trades,
        commission_bps=engine.costs.commission_bps,
        slippage_model=engine.slippage.name,
        max_adv_participation=config.max_adv_participation,
        is_walk_forward=True,
        notes=f"{len(splits)} rolling splits, {train_years}y train / {test_years}y test, out-of-sample only",
    )

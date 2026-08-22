"""Backtest endpoints. Step 3.3 (Nalin)."""
from __future__ import annotations

from datetime import date as Date
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.core.logging import get_logger
from data.schemas.backtest_result import BacktestResult, BacktestWindow
from quant.api import compute_metrics, run_backtest as api_run_backtest
from quant.backtest.engine import BacktestConfig
from quant.factors.base import Panel, load_panel
from quant.signals.generation import SignalEngine

log = get_logger(__name__)
router = APIRouter(prefix="/backtest", tags=["backtest"])

CACHE_MISSING = (
    "No price cache. Run:  python scripts/seed_data.py"
)


# --------------------------------------------------------------- shared state

@lru_cache(maxsize=1)
def get_panel() -> Panel:
    """The price panel, loaded once per process.

    risk.py imports this too — one cache, one disk read. When Matt's
    quant/api.py facade lands this should point at it instead.
    """
    try:
        return load_panel()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"{CACHE_MISSING} ({exc})") from exc


@lru_cache(maxsize=4)
def get_engine(train_frac: float = 0.6, fundamentals: bool = False) -> SignalEngine:
    """A fitted SignalEngine. Cached because fitting is the slow part."""
    panel = get_panel()
    engine = SignalEngine.default(fundamentals=fundamentals,
                                  events=fundamentals)  # events need fundamentals too
    return engine.fit(panel, train_frac=train_frac, method="ic")


def daily_signal(engine: SignalEngine, panel: Panel) -> pd.DataFrame:
    """Composite alpha on the daily grid.

    The engine scores on rebalance dates only. The backtester aligns signal and
    prices by INTERSECTION, so handing it a monthly frame silently collapses the
    run to ~60 days and annualizes it into a nonsense number — 3900% in testing.
    Forward-filling onto the trading-day index is what keeps that honest.
    """
    return engine.scores().reindex(panel.adj_close.index).ffill()


# ------------------------------------------------------------------- payloads

class BacktestRequest(BaseModel):
    """Defined here rather than in api/schemas.py — that file is the frozen
    Phase 1 contract and this is a route-local input."""

    start: Optional[Date] = None
    end: Optional[Date] = None
    rebalance_freq: str = Field("ME", description="D | W-FRI | ME")
    max_names: int = Field(15, ge=1, le=100)
    train_frac: float = Field(0.6, gt=0.0, lt=1.0)
    test_only: bool = Field(
        False, description="Restrict to the window the weights never saw"
    )
    include_fundamentals: bool = False
    strategy_name: str = "composite_alpha"


class FactorScore(BaseModel):
    factor: str
    category: str
    mean_ic: Optional[float] = None
    t_stat: Optional[float] = None
    p_value: Optional[float] = None
    hit_rate: Optional[float] = None
    significant: bool = False


class ScoreboardResponse(BaseModel):
    """The evidence behind the alpha model."""

    horizon_days: int
    n_periods: int
    factors: List[FactorScore] = Field(default_factory=list)
    n_significant: int = 0
    caveat: Optional[str] = None


# -------------------------------------------------------------------- helpers

def _honesty_note(engine: SignalEngine, panel: Panel, test_dates) -> Optional[str]:
    """Attach the caveat to the result so the number cannot travel without it.

    BacktestResult.notes is a frozen field and this is what it is for. A Sharpe
    quoted without the significance of the signal underneath it is the exact
    thing the plan's 'report the real numbers' rule exists to stop.
    """
    try:
        oos = engine.evaluate(panel, dates=test_dates, label="composite")
    except Exception:  # noqa: BLE001 - a missing caveat must not fail the run
        return None

    if oos.p_value is None or not (oos.p_value == oos.p_value):  # NaN check
        return None
    if oos.p_value < 0.05:
        return (f"Out-of-sample IC {oos.mean_ic:+.4f} (t={oos.t_stat:+.2f}, "
                f"p={oos.p_value:.3f}) over {oos.n_periods} periods.")
    return (f"CAVEAT: the underlying signal is NOT statistically significant "
            f"out of sample — IC {oos.mean_ic:+.4f}, t={oos.t_stat:+.2f}, "
            f"p={oos.p_value:.3f} over {oos.n_periods} periods. Returns below "
            f"are real arithmetic on a signal that has not been shown to predict.")


def _slice(frame, start: Optional[Date], end: Optional[Date]):
    if start is not None:
        frame = frame.loc[pd.Timestamp(start):]
    if end is not None:
        frame = frame.loc[:pd.Timestamp(end)]
    return frame


# --------------------------------------------------------------------- routes

@router.post("", response_model=BacktestResult)
@router.post("/", response_model=BacktestResult, include_in_schema=False)
def run_backtest(request: BacktestRequest) -> BacktestResult:
    """Backtest the composite alpha model. Returns the frozen 1.7 schema."""
    panel = get_panel()
    engine = get_engine(request.train_frac, request.include_fundamentals)

    signal = daily_signal(engine, panel)
    close = panel.adj_close
    adv = (panel.adj_close * panel.volume).rolling(20).mean()

    train, test = engine.train_test_dates(request.train_frac)
    start = request.start
    if request.test_only and start is None and len(test):
        start = test[0].date()

    signal, close, adv = (_slice(f, start, request.end) for f in (signal, close, adv))
    if close.empty:
        raise HTTPException(400, "No trading days in the requested window.")

    benchmark = None
    from backend.config import settings
    if settings.benchmark_ticker in close.columns:
        benchmark = close[settings.benchmark_ticker].pct_change()

    config = BacktestConfig(
        rebalance_freq=request.rebalance_freq,
        max_names=request.max_names,
        strategy_name=request.strategy_name,
    )
    # Pass the real split. Without it the engine defaults to a degenerate window
    # where train_end == test_start, and BacktestWindow.is_clean() is false —
    # which is exactly the "did you overfit?" question, answered wrong.
    window = None
    if len(train) and len(test):
        window = BacktestWindow(
            train_start=train[0].date(), train_end=train[-1].date(),
            test_start=test[0].date(), test_end=test[-1].date(),
        )

    # Through quant/api.py, the shared entry point, so this route and Zain's
    # C20 validator provably run the same code path.
    end_stamp = close.index[-1]
    span_years = max((end_stamp - close.index[0]).days / 365.25, 0.1)
    try:
        run = api_run_backtest(
            universe=list(close.columns),
            factor_scores=signal,
            as_of=end_stamp,
            lookback_years=span_years,
            config=config,
            window=window,
            benchmark_returns=benchmark,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc

    result = compute_metrics(run)
    note = _honesty_note(engine, panel, test)
    if note:
        result.notes = note if not result.notes else f"{result.notes} | {note}"

    log.info("backtest %s: ann %.2f%% sharpe %.2f over %d days",
             request.strategy_name, result.annualized_return * 100,
             result.sharpe, len(run.equity))
    return result


@router.get("/scoreboard", response_model=ScoreboardResponse)
def factor_scoreboard(
    horizon: int = 21,
    include_fundamentals: bool = False,
) -> ScoreboardResponse:
    """Per-factor information coefficients — where the weights come from."""
    panel = get_panel()
    engine = get_engine(0.6, include_fundamentals)
    board = engine.scoreboard(panel)

    factors = [
        FactorScore(
            factor=row["factor"],
            category=row.get("category", ""),
            mean_ic=_clean(row.get("mean_ic")),
            t_stat=_clean(row.get("t_stat")),
            p_value=_clean(row.get("p_value")),
            hit_rate=_clean(row.get("hit_rate")),
            significant=bool(row.get("p_value") is not None
                             and row.get("p_value") == row.get("p_value")
                             and row.get("p_value") < 0.05),
        )
        for _, row in board.iterrows()
    ]
    n_sig = sum(f.significant for f in factors)
    caveat = None
    if n_sig == 0 and factors:
        caveat = ("No factor is statistically significant at 5%. Treat the "
                  "composite as unproven, not as validated alpha.")

    return ScoreboardResponse(
        horizon_days=horizon,
        n_periods=int(board["n"].max()) if len(board) else 0,
        factors=factors,
        n_significant=n_sig,
        caveat=caveat,
    )


@router.get("/weights")
def fitted_weights(train_frac: float = 0.6, include_fundamentals: bool = False) -> dict:
    """The fitted composite weights and the train window they came from."""
    engine = get_engine(train_frac, include_fundamentals)
    fitted = engine.fitted
    if fitted is None:
        raise HTTPException(503, "Model is not fitted.")
    return {
        "method": fitted.method,
        "horizon_days": fitted.horizon,
        "train_start": str(fitted.train_start),
        "train_end": str(fitted.train_end),
        "n_train_periods": fitted.n_train_periods,
        "embargo_days": fitted.embargo_days,
        "weights": fitted.weights,
    }


def _clean(value):
    """NaN is not JSON. Return None instead of emitting invalid JSON."""
    if value is None:
        return None
    try:
        return None if value != value else float(value)
    except (TypeError, ValueError):
        return None

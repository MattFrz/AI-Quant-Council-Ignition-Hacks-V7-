"""Execution simulation, backed by the C++ order book.

Everything here is computed by the pybind11 extension in cpp/, not by Python:
the book is a real price-time-priority structure, and the fills come from
walking its actual depth. The endpoint exists so the execution layer can be
demonstrated rather than described.

If the extension is not built, this returns 503 and says so. It is optional at
runtime everywhere else in the codebase (quant/backtest/slippage.py falls back
to the analytic model), and pretending to have run it would be worse than
admitting it is missing.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/execution", tags=["execution"])


class BookLevel(BaseModel):
    price: float
    shares: int


class ExecutionOutcome(BaseModel):
    mode: str
    requested: int
    filled: int
    avg_price: float
    arrival_mid: float
    slippage_bps: float
    levels_consumed: int
    complete: bool
    note: str


class SimulationRequest(BaseModel):
    shares: int = Field(2500, ge=1, le=1_000_000)
    side: str = Field("BUY", pattern="^(BUY|SELL)$")
    slices: int = Field(4, ge=1, le=50)
    levels: int = Field(5, ge=1, le=20)
    level_shares: int = Field(1000, ge=1, le=100_000)
    tick: float = Field(0.01, gt=0, le=1.0)
    #: Shares that trade at our limit price before we are filled. A passive
    #: order does not fill because it exists, it fills because the queue in
    #: front of it traded first.
    queue_volume: int = Field(300, ge=0, le=1_000_000)


class SimulationResponse(BaseModel):
    available: bool
    bids: List[BookLevel]
    asks: List[BookLevel]
    mid: float
    spread_bps: float
    capacity_5bps: int
    outcomes: List[ExecutionOutcome]
    engine: str


def _load():
    try:
        import aqc_exec  # noqa: PLC0415 - optional native extension
    except ImportError as exc:  # pragma: no cover - depends on build env
        raise HTTPException(
            status_code=503,
            detail=(
                "The C++ execution extension is not built in this deployment. "
                "Build it with: pip install ./cpp/bindings"
            ),
        ) from exc
    return aqc_exec


def _build_book(ax, req: SimulationRequest):
    """A symmetric ladder around 100.00, so the numbers are easy to check by eye."""
    book = ax.OrderBook()
    ref = 1
    for i in range(req.levels):
        book.add(ref, ax.Side.BUY, 100.00 - i * req.tick, req.level_shares)
        ref += 1
        book.add(ref, ax.Side.SELL, 100.00 + req.tick + i * req.tick, req.level_shares)
        ref += 1
    return book


@router.post("", response_model=SimulationResponse)
@router.post("/", response_model=SimulationResponse, include_in_schema=False)
def simulate(req: SimulationRequest) -> SimulationResponse:
    ax = _load()
    side = ax.Side.BUY if req.side == "BUY" else ax.Side.SELL

    quote = _build_book(ax, req).top()
    sim = ax.ExecutionSimulator()
    outcomes: List[ExecutionOutcome] = []

    def record(result, note: str) -> None:
        outcomes.append(
            ExecutionOutcome(
                mode=result.mode,
                requested=result.requested,
                filled=result.filled,
                avg_price=result.avg_price,
                arrival_mid=result.arrival_mid,
                slippage_bps=result.slippage_bps,
                levels_consumed=result.levels_consumed,
                complete=result.complete,
                note=note,
            )
        )

    # Each order runs against a FRESH book. Reusing one would mean the second
    # order trades against depth the first already consumed, which measures
    # something other than what the labels claim.
    record(
        sim.market_order(_build_book(ax, req), side, req.shares),
        "Crosses the spread immediately. Walks real depth, so cost grows with size.",
    )
    record(
        sim.sliced_order(_build_book(ax, req), side, req.shares, req.slices),
        f"Split into {req.slices} child orders. Usually beats one large order.",
    )
    record(
        sim.limit_order(_build_book(ax, req), side, req.shares, req.queue_volume, False),
        f"Posted passively. Only {req.queue_volume:,} shares traded at our price, "
        "so the queue ahead was never cleared.",
    )
    record(
        sim.limit_order(_build_book(ax, req), side, req.shares, req.shares * 10, True),
        "Posted passively and the price moved through us. Filled, but adversely: "
        "getting picked off is charged as a cost, not booked at our limit.",
    )

    book = _build_book(ax, req)
    return SimulationResponse(
        available=True,
        bids=[BookLevel(price=p, shares=s) for p, s in book.depth(ax.Side.BUY, req.levels)],
        asks=[BookLevel(price=p, shares=s) for p, s in book.depth(ax.Side.SELL, req.levels)],
        mid=quote.mid(),
        spread_bps=quote.spread_bps(),
        capacity_5bps=ax.capacity_within(book, side, 5.0),
        outcomes=outcomes,
        engine="aqc_exec (C++, pybind11)",
    )


@router.get("/status")
def status() -> dict:
    """Whether the native extension is present, without raising."""
    try:
        import aqc_exec  # noqa: PLC0415

        return {"available": True, "engine": "aqc_exec (C++, pybind11)"}
    except ImportError:
        return {"available": False, "engine": None}

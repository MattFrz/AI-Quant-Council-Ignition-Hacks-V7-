"""Event types for the backtest loop. Step A7.

The engine consumes these in timestamp order. Keeping them as explicit objects
rather than implicit DataFrame operations is what makes the loop auditable -
you can dump the event log and see exactly what the strategy did and when.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

import pandas as pd


class EventType(str, Enum):
    MARKET = "market"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    REBALANCE = "rebalance"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @classmethod
    def from_quantity(cls, qty: float) -> "Side":
        return cls.BUY if qty >= 0 else cls.SELL


@dataclass
class Event:
    timestamp: pd.Timestamp
    type: EventType


@dataclass
class MarketEvent(Event):
    """A new bar is available. Prices as of this timestamp are now knowable."""
    prices: Dict[str, float] = field(default_factory=dict)

    def __init__(self, timestamp: pd.Timestamp, prices: Dict[str, float]):
        super().__init__(timestamp, EventType.MARKET)
        self.prices = prices


@dataclass
class SignalEvent(Event):
    """Target weights produced by the alpha model, already execution-lagged."""
    weights: Dict[str, float] = field(default_factory=dict)

    def __init__(self, timestamp: pd.Timestamp, weights: Dict[str, float]):
        super().__init__(timestamp, EventType.SIGNAL)
        self.weights = weights


@dataclass
class OrderEvent(Event):
    ticker: str = ""
    side: Side = Side.BUY
    quantity: float = 0.0
    target_notional: float = 0.0
    #: Set when the order was cut down to respect the ADV participation cap.
    liquidity_capped: bool = False

    def __init__(
        self,
        timestamp: pd.Timestamp,
        ticker: str,
        quantity: float,
        target_notional: float,
        liquidity_capped: bool = False,
    ):
        super().__init__(timestamp, EventType.ORDER)
        self.ticker = ticker
        self.side = Side.from_quantity(quantity)
        self.quantity = quantity
        self.target_notional = target_notional
        self.liquidity_capped = liquidity_capped


@dataclass
class FillEvent(Event):
    ticker: str = ""
    side: Side = Side.BUY
    quantity: float = 0.0
    fill_price: float = 0.0
    reference_price: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0

    def __init__(
        self,
        timestamp: pd.Timestamp,
        ticker: str,
        quantity: float,
        fill_price: float,
        reference_price: float,
        commission: float,
        slippage_cost: float,
    ):
        super().__init__(timestamp, EventType.FILL)
        self.ticker = ticker
        self.side = Side.from_quantity(quantity)
        self.quantity = quantity
        self.fill_price = fill_price
        self.reference_price = reference_price
        self.commission = commission
        self.slippage_cost = slippage_cost

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.fill_price

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage_cost

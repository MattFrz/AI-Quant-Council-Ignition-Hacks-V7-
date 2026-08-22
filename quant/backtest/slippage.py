"""Market impact / slippage. Step A10.

Deliberately narrow interface: Phase 4 swaps a C++ order-book simulator in
behind `SlippageModel` without the engine changing. Per plan step 4.7 the pure
Python path must keep working, so a failed C++ build degrades to a slightly less
impressive number rather than a broken demo.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.core.logging import get_logger

log = get_logger(__name__)


class SlippageModel(ABC):
    """Return the cost, in basis points, of executing `notional` in one name."""

    name: str = "base"

    @abstractmethod
    def impact_bps(
        self,
        notional: float,
        adv_usd: float,
        volatility: Optional[float] = None,
    ) -> float:
        ...

    def cost(self, notional: float, adv_usd: float, volatility: Optional[float] = None) -> float:
        return abs(notional) * self.impact_bps(notional, adv_usd, volatility) / 1e4

    def fill_price(
        self,
        reference_price: float,
        quantity: float,
        adv_usd: float,
        volatility: Optional[float] = None,
    ) -> float:
        """Buys fill above the reference, sells below. Always against you."""
        notional = abs(quantity) * reference_price
        bps = self.impact_bps(notional, adv_usd, volatility)
        direction = 1.0 if quantity >= 0 else -1.0
        return reference_price * (1.0 + direction * bps / 1e4)


@dataclass
class ParticipationRateSlippage(SlippageModel):
    """Square-root market impact.

        impact_bps = coefficient * volatility_bps * sqrt(participation)

    Participation is the trade as a fraction of that day's dollar volume. The
    square root is the standard empirical shape - impact grows sublinearly with
    size. Defaults are conservative on purpose.
    """

    coefficient: float = 1.0
    default_volatility: float = 0.02   # 2% daily
    min_bps: float = 0.5
    max_bps: float = 300.0
    name: str = "participation_rate"

    def impact_bps(
        self,
        notional: float,
        adv_usd: float,
        volatility: Optional[float] = None,
    ) -> float:
        if not adv_usd or not np.isfinite(adv_usd) or adv_usd <= 0:
            return self.max_bps

        participation = abs(notional) / adv_usd
        vol = volatility if volatility and np.isfinite(volatility) else self.default_volatility
        vol_bps = vol * 1e4

        bps = self.coefficient * vol_bps * math.sqrt(participation)
        return float(np.clip(bps, self.min_bps, self.max_bps))


@dataclass
class FixedBpsSlippage(SlippageModel):
    """Flat assumption. Useful as a sanity baseline, not for the headline run."""

    bps: float = 5.0
    name: str = "fixed_bps"

    def impact_bps(self, notional: float, adv_usd: float, volatility: Optional[float] = None) -> float:
        return self.bps


def cpp_available() -> bool:
    """Is the compiled execution simulator importable?"""
    try:
        import aqc_exec  # noqa: F401
        return True
    except ImportError:
        return False


def get_slippage_model(use_cpp: bool = False, **kwargs) -> SlippageModel:
    """Factory. Step 4.7.

    The C++ path is opt-in and always degrades. A compiler problem on the demo
    laptop must cost us a more precise number, never a working demo - so an
    import failure logs once and returns the pure-Python model.
    """
    if use_cpp:
        try:
            import aqc_exec  # noqa: F401

            log.info("slippage: using C++ order-book execution simulator")
            return OrderBookSlippage(**kwargs)
        except ImportError as exc:
            log.warning(
                "slippage: aqc_exec not built (%s) - falling back to the "
                "participation-rate model. Run scripts/build_cpp.sh to enable.",
                exc,
            )

    return ParticipationRateSlippage(**kwargs)


@dataclass
class OrderBookSlippage(SlippageModel):
    """Impact measured by walking a reconstructed book instead of assuming it.

    Needs a book to walk. When one has been supplied for the symbol - via
    `attach_book()` after an ITCH replay - impact comes from real resting
    depth. With no book attached it delegates to the participation model
    rather than guessing, so a partially-populated run stays coherent.
    """

    coefficient: float = 1.0
    default_volatility: float = 0.02
    min_bps: float = 0.5
    max_bps: float = 300.0
    name: str = "cpp_orderbook"

    def __post_init__(self) -> None:
        self._books: dict = {}
        self._fallback = ParticipationRateSlippage(
            coefficient=self.coefficient,
            default_volatility=self.default_volatility,
            min_bps=self.min_bps,
            max_bps=self.max_bps,
        )

    def attach_book(self, ticker: str, book) -> None:
        """Register a replayed book for a symbol."""
        self._books[ticker.upper()] = book

    def impact_bps(
        self,
        notional: float,
        adv_usd: float,
        volatility: Optional[float] = None,
        ticker: Optional[str] = None,
    ) -> float:
        book = self._books.get((ticker or "").upper())
        if book is None:
            return self._fallback.impact_bps(notional, adv_usd, volatility)

        import aqc_exec

        quote = book.top()
        if not quote.valid() or quote.mid() <= 0:
            return self._fallback.impact_bps(notional, adv_usd, volatility)

        shares = max(1, int(abs(notional) / quote.mid()))
        side = aqc_exec.Side.BUY if notional >= 0 else aqc_exec.Side.SELL

        result = aqc_exec.ExecutionSimulator().market_order(book, side, shares)
        if result.filled == 0:
            return self.max_bps

        bps = abs(result.slippage_bps)
        # An order that could not be filled from displayed depth would have to
        # wait or pay up; charging only the walked portion would understate it.
        if not result.complete:
            bps = max(bps, self.max_bps * 0.5)

        return float(min(max(bps, self.min_bps), self.max_bps))


DEFAULT_SLIPPAGE_MODEL = ParticipationRateSlippage()

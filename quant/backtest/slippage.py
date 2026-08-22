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


def get_slippage_model(use_cpp: bool = False, **kwargs) -> SlippageModel:
    """Factory. Phase 4 wires the C++ simulator in here behind the flag.

    If the extension is missing or fails to import we log and fall back, because
    a compiler problem on the demo laptop must never take the demo down.
    """
    if use_cpp:
        try:
            from cpp.bindings import execution_sim  # type: ignore  # noqa: F401

            log.info("slippage: using C++ execution simulator")
            return _CppSlippageAdapter(**kwargs)  # pragma: no cover - Phase 4
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "slippage: C++ simulator unavailable (%s) - falling back to "
                "participation-rate model", exc,
            )

    return ParticipationRateSlippage(**kwargs)


class _CppSlippageAdapter(SlippageModel):  # pragma: no cover - Phase 4 placeholder
    """Thin wrapper over the pybind11 module. Implemented in step 4.7."""

    name = "cpp_orderbook"

    def impact_bps(self, notional: float, adv_usd: float, volatility: Optional[float] = None) -> float:
        raise NotImplementedError("Phase 4: wire cpp/bindings/pybind_module.cpp")


DEFAULT_SLIPPAGE_MODEL = ParticipationRateSlippage()

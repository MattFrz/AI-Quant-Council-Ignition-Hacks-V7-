from __future__ import annotations
from dataclasses import dataclass

from backend.research.thesis.decomposer import ThesisCriteria
from backend.config import settings

# NOTE: build_universe (A6) does NOT filter by sector or keyword - its four
# stages are listing, liquidity, market cap, and data completeness only.
# sector/theme/key_entities from the thesis aren't inputs to the universe
# screen itself; they're carried separately for use elsewhere (e.g. as a
# display label, or as input to Nalin's factor scoring later in the
# pipeline) - don't try to force them into ScreenParams for A6.


@dataclass
class ScreenParams:
    """Matches build_universe's actual kwargs - not a passthrough object,
    build_universe takes these as individual arguments, not a ScreenParams
    instance. Kept as a dataclass here for readability at the call site;
    unpack with **params.__dict__ or pass fields individually."""
    min_market_cap: float | None
    min_adv_usd: float | None
    min_days: int
    max_size: int | None


# Rough market-cap-tier hints a thesis's theme/text might imply. This is the
# only part of ThesisCriteria that maps onto anything build_universe accepts
# - sector and key_entities genuinely have no A6 parameter to land in, so
# they're intentionally NOT used here (see note above). If this list stays
# empty in practice, that's a sign this heuristic isn't worth keeping -
# don't be afraid to delete it after a few real thesis runs show it never
# fires usefully.
_SMALL_CAP_HINTS = ("small-cap", "small cap", "micro-cap", "emerging")
_LARGE_CAP_HINTS = ("mega-cap", "large-cap", "large cap", "blue-chip")


def criteria_to_screen_params(criteria: ThesisCriteria) -> ScreenParams:
    """
    Derives a market-cap floor override from the thesis's theme text when it
    clearly implies a tier; otherwise falls back to config defaults.
    liquidity/data-completeness params always come from settings - nothing
    in ThesisCriteria maps onto those, so there's no thesis-driven override
    for min_adv_usd or min_days.
    """
    theme_lower = (criteria.theme or "").lower()

    min_market_cap = settings.min_market_cap
    if any(hint in theme_lower for hint in _SMALL_CAP_HINTS):
        min_market_cap = min(settings.min_market_cap, 300_000_000)  # loosen the floor
    elif any(hint in theme_lower for hint in _LARGE_CAP_HINTS):
        min_market_cap = max(settings.min_market_cap, 50_000_000_000)  # raise the floor

    return ScreenParams(
        min_market_cap=min_market_cap,
        min_adv_usd=settings.min_adv_usd,
        min_days=500,  # matches build_universe's own default; override here if a thesis implies otherwise
        max_size=settings.universe_size,
    )


def call_build_universe(criteria: ThesisCriteria, panel, profiles):
    """
    Convenience wrapper showing the actual call shape against A6 - panel and
    profiles are the DataFrames Matt's pipeline produces, not something this
    function builds itself. Import build_universe at the call site:
        from quant.universe.builder import build_universe
    """
    params = criteria_to_screen_params(criteria)
    from quant.universe.builder import build_universe  # local import to avoid a hard dep at module load
    return build_universe(
        panel=panel,
        profiles=profiles,
        min_market_cap=params.min_market_cap,
        min_adv_usd=params.min_adv_usd,
        min_days=params.min_days,
        max_size=params.max_size,
    )
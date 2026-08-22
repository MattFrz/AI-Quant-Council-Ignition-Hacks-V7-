from __future__ import annotations
from dataclasses import dataclass, field

from backend.research.thesis.decomposer import ThesisCriteria

# Rough sector -> market cap band defaults, used only when the thesis doesn't
# imply a tighter range. Tune these with Matt once universe/filters.py (A5) exists.
_DEFAULT_MIN_MARKET_CAP = 2_000_000_000  # $2B floor, matches a typical liquid-universe cutoff


@dataclass
class ScreenParams:
    sector: str | None
    market_cap_min: float | None
    market_cap_max: float | None
    keywords: list[str] = field(default_factory=list)


def criteria_to_screen_params(criteria: ThesisCriteria) -> ScreenParams:
    """
    Pure mapping, no LLM call. Converts structured thesis criteria into the
    screen parameters Matt's quant/universe/builder.py (A6) consumes.

    IMPORTANT TODO: confirm field names below against A6's actual input signature
    before wiring this into the pipeline — don't assume, ask Matt.
    """
    keywords = [criteria.theme] + criteria.key_entities

    return ScreenParams(
        sector=criteria.sector,
        market_cap_min=_DEFAULT_MIN_MARKET_CAP if criteria.sector else None,
        market_cap_max=None,
        keywords=keywords,
    )